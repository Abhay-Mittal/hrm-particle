"""Autoregressive fixed-width particle rollouts and exact PPO replay.

The implementation is model-wrapper agnostic.  A compatible model accepts the
usual causal-LM arguments plus ``particle_z`` and ``particle_mask`` and returns
``logits``, ``terminal_state``, ``prompt_summary`` and (optionally)
``past_key_values``.  :class:`ParticleHrmForCausalLM` in this project implements
that contract, while tests use a tiny CPU-only dummy.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Callable, List, Literal, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from .cache import CacheCloneError, repeat_cache_batch
from .prompting import build_prefixlm_batch, format_hrm_prompt
from .verifier import ExactArithmeticVerifier, VerificationResult


@dataclass(frozen=True)
class RolloutExample:
    prompt: str
    answer: str
    example_id: str = ""
    metadata: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class RolloutConfig:
    k: int = 4
    latent_dim: int = 64
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 1.0
    # In ``causal_prefix`` mode these fixed response tokens are type 0 (causal),
    # while the official formatted prompt is type 1 (bidirectional prefix).
    response_prefix: str = "\nSolution:\n"
    first_token_mode: Literal[
        "causal_prefix", "shared_prefill", "branch_recompute"
    ] = "causal_prefix"
    generation_token_type_id: int = 0
    condition: str = "synth,cot"
    use_cache: bool = True
    compute_reference_logprobs: bool = True
    # Training uses one exact greedy z=0 anchor.  Matched ordinary-sampling
    # evaluation sets this false so all K zero-latent branches are sampled.
    anchor_greedy: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError("k must include one anchor and at least one explorer")
        if self.latent_dim <= 0 or self.max_new_tokens <= 0:
            raise ValueError("latent_dim and max_new_tokens must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must lie in (0, 1]")
        if self.first_token_mode not in {
            "causal_prefix",
            "shared_prefill",
            "branch_recompute",
        }:
            raise ValueError("unknown first_token_mode")
        if self.first_token_mode == "causal_prefix" and not self.response_prefix:
            raise ValueError("causal_prefix mode requires a non-empty response_prefix")


@dataclass
class ParticleRollout:
    """A replay-complete batch.

    ``particle_mask`` controls internal H injection.  ``action_mask`` controls
    the actor loss.  They differ intentionally: fixed response-prefix tokens
    must be injected so the first sampled token depends on the particle, but
    those prefix tokens are not sampled actions and receive no policy loss.
    """

    model_input_ids: Tensor  # [B, K, sequence]
    attention_mask: Tensor  # [B, K, sequence]
    token_type_ids: Optional[Tensor]  # [B, K, sequence], PrefixLM boundary labels
    position_ids: Tensor  # [B, K, sequence], padding-invariant absolute positions
    particle_mask: Tensor  # [B, K, sequence]
    particle_z: Tensor  # [B, K, latent]
    action_ids: Tensor  # [B, K, time]
    generated_mask: Tensor  # [B, K, time], includes the first sampled token
    action_mask: Tensor  # [B, K, time], trainable adapter actions only
    action_positions: Tensor  # [B, K, time], position whose logits predict action
    old_logprobs: Tensor  # [B, K, time]
    rewards: Tensor  # [B, K]
    prompt_summary: Tensor  # [B, K, hidden]
    # State that produced EOS; for truncation, state after the final sampled token.
    terminal_states: Tensor  # [B, K, hidden]
    response_texts: List[List[str]]
    example_ids: List[str]
    references: List[str]
    verification: List[List[VerificationResult]]
    temperature: float
    top_p: float
    reference_logprobs: Optional[Tensor] = None
    # Optional strict binary correctness targets for Q.  Actor credit always
    # comes from ``rewards``; when absent, Q retains the legacy rewards target.
    q_labels: Optional[Tensor] = None

    @property
    def batch_size(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def k(self) -> int:
        return int(self.rewards.shape[1])

    def to(self, device: torch.device | str) -> "ParticleRollout":
        updates = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                updates[field.name] = value.to(device)
        return replace(self, **updates)


@dataclass(frozen=True)
class ScoredActions:
    logprobs: Tensor
    raw_logprobs: Tensor
    output: Any


def _get_output(output: Any, name: str, default: Any = None) -> Any:
    if isinstance(output, Mapping):
        return output.get(name, default)
    return getattr(output, name, default)


def _terminal_state(output: Any) -> Tensor:
    state = _get_output(output, "terminal_state")
    if state is not None:
        return state
    state = _get_output(output, "last_hidden_state")
    if state is not None:
        return state[:, -1]
    hidden_states = _get_output(output, "hidden_states")
    if hidden_states:
        return hidden_states[-1][:, -1]
    raise RuntimeError("model output must expose terminal_state or a last hidden state")


def _prompt_summary(output: Any) -> Tensor:
    summary = _get_output(output, "prompt_summary")
    return summary if summary is not None else _terminal_state(output)


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> List[int]:
    encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if isinstance(encoded, Tensor):
        encoded = encoded.tolist()
    return [int(token) for token in encoded]


def _encode_prompt(tokenizer: Any, text: str) -> tuple[List[int], Optional[List[int]]]:
    """Tokenize a complete PrefixLM prompt and retain boundary labels if given."""

    if callable(tokenizer):
        try:
            encoded = tokenizer(text, add_special_tokens=True, return_attention_mask=False)
        except (TypeError, NotImplementedError):
            encoded = None
        if isinstance(encoded, Mapping) and "input_ids" in encoded:
            ids = encoded["input_ids"]
            types = encoded.get("token_type_ids")
            if isinstance(ids, Tensor):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if isinstance(types, Tensor):
                types = types.tolist()
            if types and isinstance(types[0], list):
                types = types[0]
            return [int(token) for token in ids], (
                [int(token) for token in types] if types is not None else None
            )
    return _encode(tokenizer, text, add_special_tokens=True), None


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return str(tokenizer.decode(list(ids), skip_special_tokens=True)).strip()


def _top_p_logits(logits: Tensor, top_p: float) -> Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > top_p
    # Keep the first token whose inclusion crosses the threshold.
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove_original = torch.zeros_like(remove).scatter(-1, sorted_indices, remove)
    return logits.masked_fill(remove_original, torch.finfo(logits.dtype).min)


def sampling_logprobs(logits: Tensor, temperature: float, top_p: float) -> Tensor:
    # Log-probabilities and multinomial probabilities stay fp32 even when the
    # 1B backbone runs in bf16; PPO ratios are too sensitive for bf16 log-softmax.
    scaled = logits.float() / temperature
    return torch.log_softmax(_top_p_logits(scaled, top_p), dim=-1)


def normalized_gaussian_latents(
    batch_size: int,
    k: int,
    latent_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tensor:
    """Sample unit-RMS explorer codes and an exactly zero anchor code."""

    latents = torch.randn(
        batch_size,
        k,
        latent_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    latents = latents / latents.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    latents[:, 0].zero_()
    return latents


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _call_particle_model(
    model: nn.Module,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    particle_z: Optional[Tensor],
    particle_mask: Optional[Tensor],
    use_cache: bool,
    token_type_ids: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    past_key_values: Any = None,
    prompt_summary: Optional[Tensor] = None,
) -> Any:
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        particle_z=particle_z,
        particle_mask=particle_mask,
        use_cache=use_cache,
        past_key_values=past_key_values,
        return_dict=True,
        return_q=False,
    )
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if prompt_summary is not None:
        # Cached one-token forwards cannot reconstruct the clean prompt summary.
        kwargs["prompt_summary"] = prompt_summary
    return model(**kwargs)


def score_actions(
    model: nn.Module,
    rollout: ParticleRollout,
    *,
    particle_z: Optional[Tensor] = None,
) -> ScoredActions:
    """Teacher-force a rollout with the exact stored latent and injection mask."""

    b, k, sequence_length = rollout.model_input_ids.shape
    time = rollout.action_ids.shape[-1]
    latents = rollout.particle_z if particle_z is None else particle_z
    if latents.shape != rollout.particle_z.shape:
        raise ValueError("particle_z override must match stored particle_z shape")

    output = _call_particle_model(
        model,
        input_ids=rollout.model_input_ids.reshape(b * k, sequence_length),
        attention_mask=rollout.attention_mask.reshape(b * k, sequence_length),
        particle_z=latents.reshape(b * k, -1),
        particle_mask=rollout.particle_mask.reshape(b * k, sequence_length),
        token_type_ids=(
            rollout.token_type_ids.reshape(b * k, sequence_length)
            if rollout.token_type_ids is not None
            else None
        ),
        position_ids=rollout.position_ids.reshape(b * k, sequence_length),
        use_cache=False,
        prompt_summary=rollout.prompt_summary.reshape(b * k, -1),
    )
    logits = _get_output(output, "logits")
    if logits is None or logits.ndim != 3:
        raise RuntimeError("model output must expose logits [batch, sequence, vocab]")
    positions = rollout.action_positions.reshape(b * k, time)
    gather_positions = positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    action_logits = logits.gather(1, gather_positions)
    all_logprobs = sampling_logprobs(action_logits, rollout.temperature, rollout.top_p)
    chosen = all_logprobs.gather(
        -1, rollout.action_ids.reshape(b * k, time).unsqueeze(-1)
    ).squeeze(-1)
    # Untruncated behavior distribution for reference KL.  Temperature must
    # match sampling; only nucleus truncation is omitted.
    raw = torch.log_softmax(action_logits.float() / rollout.temperature, dim=-1).gather(
        -1, rollout.action_ids.reshape(b * k, time).unsqueeze(-1)
    ).squeeze(-1)
    return ScoredActions(
        logprobs=chosen.reshape(b, k, time),
        raw_logprobs=raw.reshape(b, k, time),
        output=output,
    )


class ParticleRolloutEngine:
    """Generate one fixed anchor and ``K-1`` query-conditioned explorers."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        verifier: Optional[ExactArithmeticVerifier] = None,
        config: Optional[RolloutConfig] = None,
        latent_sampler: Optional[Callable[..., Tensor]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier or ExactArithmeticVerifier()
        self.config = config or RolloutConfig()
        self.latent_sampler = latent_sampler or normalized_gaussian_latents
        self._generators: dict[str, torch.Generator] = {}

    def _generator(self, device: torch.device, stream: str) -> torch.Generator:
        if stream not in {"tokens", "latents"}:
            raise ValueError(f"unknown RNG stream {stream!r}")
        generator_device: torch.device | str = (
            device if device.type in {"cpu", "cuda"} else "cpu"
        )
        key = f"{device}|{stream}"
        if key not in self._generators:
            generator = torch.Generator(device=generator_device)
            offset = 0 if stream == "tokens" else 1_000_003
            generator.manual_seed(self.config.seed + offset)
            self._generators[key] = generator
        return self._generators[key]

    def rng_state_dict(self) -> dict[str, Tensor]:
        """Return persistent rollout RNG states for exact checkpoint resume."""

        return {key: generator.get_state() for key, generator in self._generators.items()}

    def load_rng_state_dict(self, states: Mapping[str, Tensor]) -> None:
        self._generators.clear()
        for key, state in states.items():
            # ``cpu`` without a stream is accepted for pre-split POC checkpoints.
            device_name = key.split("|", 1)[0]
            restored_device = torch.device(device_name)
            generator_device = (
                restored_device if restored_device.type in {"cpu", "cuda"} else "cpu"
            )
            generator = torch.Generator(device=generator_device)
            generator.set_state(state.cpu())
            self._generators[str(key)] = generator

    @torch.no_grad()
    def generate(self, examples: Sequence[RolloutExample]) -> ParticleRollout:
        if not examples:
            raise ValueError("at least one rollout example is required")
        config = self.config
        device = _model_device(self.model)
        token_generator = self._generator(device, "tokens")
        latent_generator = self._generator(device, "latents")
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id is required")
        if pad_id is None:
            pad_id = eos_id

        batch_size = len(examples)
        if config.first_token_mode == "causal_prefix":
            prefix_batch = build_prefixlm_batch(
                self.tokenizer,
                [example.prompt for example in examples],
                condition=config.condition,
                response_prefix=config.response_prefix,
                device=device,
            )
            base_ids = prefix_batch.input_ids
            base_attention = prefix_batch.attention_mask
            base_token_types = prefix_batch.token_type_ids
            base_particle_mask = prefix_batch.particle_mask
            base_position_ids = prefix_batch.position_ids
        else:
            # Diagnostic modes omit the causal response prefix.  They still use
            # the official HRM prompt formatter and explicit PrefixLM labels.
            prompt_ids = [
                _encode(
                    self.tokenizer,
                    format_hrm_prompt(example.prompt, config.condition),
                    add_special_tokens=False,
                )
                for example in examples
            ]
            max_prompt = max(map(len, prompt_ids))
            base_ids = torch.full(
                (batch_size, max_prompt), int(pad_id), dtype=torch.long, device=device
            )
            base_attention = torch.zeros_like(base_ids, dtype=torch.bool)
            base_token_types = torch.zeros_like(base_ids)
            base_particle_mask = torch.zeros_like(base_ids, dtype=torch.bool)
            for row, ids in enumerate(prompt_ids):
                start = max_prompt - len(ids)
                base_ids[row, start:] = torch.tensor(ids, device=device)
                base_attention[row, start:] = True
                base_token_types[row, start:] = 1
            base_position_ids = base_attention.long().cumsum(dim=-1).sub(1).clamp_min(0)
        initial_length = base_ids.shape[1]
        model_config = getattr(self.model, "config", None)
        max_positions = getattr(model_config, "max_position_embeddings", None)
        if isinstance(max_positions, int) and initial_length + config.max_new_tokens > max_positions:
            longest = max(
                int(base_attention[row].sum()) for row in range(base_attention.shape[0])
            )
            raise ValueError(
                "prompt plus configured response exceeds the model context window: "
                f"padded_prefix={initial_length}, longest_prefix={longest}, "
                f"max_new_tokens={config.max_new_tokens}, context={max_positions}; "
                "filter this record during data preparation rather than truncating it silently"
            )

        # Contiguous branch layout: (prompt0,z0..zK-1), (prompt1,z0..zK-1), ...
        input_ids = base_ids.repeat_interleave(config.k, dim=0)
        attention_mask = base_attention.repeat_interleave(config.k, dim=0)
        particle_mask = base_particle_mask.repeat_interleave(config.k, dim=0)
        token_type_ids = base_token_types.repeat_interleave(config.k, dim=0)
        position_ids = base_position_ids.repeat_interleave(config.k, dim=0)

        parameter_dtype = next(self.model.parameters()).dtype
        particle_z = self.latent_sampler(
            batch_size,
            config.k,
            config.latent_dim,
            device=device,
            dtype=parameter_dtype,
            generator=latent_generator,
        )
        if particle_z.shape != (batch_size, config.k, config.latent_dim):
            raise ValueError("latent_sampler returned an unexpected shape")
        if not torch.equal(particle_z[:, 0], torch.zeros_like(particle_z[:, 0])):
            raise ValueError("latent_sampler must return an exactly zero particle-zero anchor")
        flat_z = particle_z.reshape(batch_size * config.k, config.latent_dim)

        was_training = self.model.training
        self.model.eval()
        fixed_summary: Optional[Tensor] = None
        if config.first_token_mode == "causal_prefix":
            assert prefix_batch.response_prefix_length > 0
            prompt_stop = initial_length - prefix_batch.response_prefix_length
            # Compute the official bidirectional prompt once, with no adapter at
            # all.  This supplies both a clean query state and a reusable cache.
            clean_output = _call_particle_model(
                self.model,
                input_ids=base_ids[:, :prompt_stop],
                attention_mask=base_attention[:, :prompt_stop],
                token_type_ids=base_token_types[:, :prompt_stop],
                position_ids=base_position_ids[:, :prompt_stop],
                particle_z=None,
                particle_mask=None,
                use_cache=config.use_cache,
            )
            clean_summary = _prompt_summary(clean_output)
            fixed_summary = clean_summary.repeat_interleave(config.k, dim=0)
            clean_past = _get_output(clean_output, "past_key_values")
            repeated_past = None
            if config.use_cache and clean_past is not None:
                try:
                    repeated_past = repeat_cache_batch(clean_past, config.k)
                except CacheCloneError:
                    # Tiny/dummy and older legacy caches may not support batch
                    # repeat.  A K-batched full recomputation is slower but exact.
                    repeated_past = None
            if repeated_past is not None:
                output = _call_particle_model(
                    self.model,
                    input_ids=input_ids[:, prompt_stop:],
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids[:, prompt_stop:],
                    position_ids=position_ids[:, prompt_stop:],
                    particle_z=flat_z,
                    particle_mask=particle_mask[:, prompt_stop:],
                    use_cache=True,
                    past_key_values=repeated_past,
                    prompt_summary=fixed_summary,
                )
            else:
                output = _call_particle_model(
                    self.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    particle_z=flat_z,
                    particle_mask=particle_mask,
                    use_cache=config.use_cache,
                    prompt_summary=fixed_summary,
                )
        elif config.first_token_mode == "branch_recompute":
            # A separate clean pass computes a query summary with no particle
            # intervention.  The complete prompt is then recomputed per branch,
            # with injection only at its final position.  This is the expensive
            # but PrefixLM-correct way to branch the very first sampled token.
            clean_output = _call_particle_model(
                self.model,
                input_ids=base_ids,
                attention_mask=base_attention,
                token_type_ids=base_token_types,
                position_ids=base_attention.long().cumsum(dim=-1).sub(1).clamp_min(0),
                particle_z=None,
                particle_mask=None,
                use_cache=False,
            )
            fixed_summary = _prompt_summary(clean_output).repeat_interleave(config.k, dim=0)
            particle_mask[:, -1] = True
            output = _call_particle_model(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                particle_z=flat_z,
                particle_mask=particle_mask,
                use_cache=config.use_cache,
                prompt_summary=fixed_summary,
            )
        else:
            # Clean K-batched prefill.  Token one is unperturbed and explicitly
            # masked from PPO; particle injection starts when token one is fed.
            output = _call_particle_model(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                particle_z=None,
                particle_mask=None,
                use_cache=config.use_cache,
            )
        summary = fixed_summary if fixed_summary is not None else _prompt_summary(output)
        if summary.shape[0] != batch_size * config.k:
            raise RuntimeError("prompt_summary has an unexpected batch dimension")
        past = _get_output(output, "past_key_values")

        active = torch.ones(batch_size * config.k, dtype=torch.bool, device=device)
        action_ids_steps: List[Tensor] = []
        generated_mask_steps: List[Tensor] = []
        action_mask_steps: List[Tensor] = []
        old_logprob_steps: List[Tensor] = []
        latest_producer_state = _terminal_state(output).clone()

        for step in range(config.max_new_tokens):
            logits = _get_output(output, "logits")
            if logits is None:
                raise RuntimeError("model output must expose logits")
            next_logits = logits[:, -1]
            logprobs = sampling_logprobs(next_logits, config.temperature, config.top_p)
            branch_ids = torch.arange(batch_size * config.k, device=device) % config.k
            anchor = branch_ids == 0

            sampled = torch.multinomial(
                logprobs.exp(), num_samples=1, generator=token_generator
            ).squeeze(-1)
            if config.anchor_greedy:
                sampled[anchor] = next_logits[anchor].argmax(dim=-1)
            sampled = torch.where(active, sampled, torch.full_like(sampled, int(pad_id)))
            chosen_logprobs = logprobs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            chosen_logprobs = torch.where(active, chosen_logprobs, torch.zeros_like(chosen_logprobs))

            producer = _terminal_state(output)
            latest_producer_state[active] = producer[active]
            was_active = active.clone()
            action_ids_steps.append(sampled)
            generated_mask_steps.append(was_active)
            # In shared-prefill mode token one came from a completely clean
            # prompt state.  It has no adapter gradient path and is explicitly
            # excluded from PPO.  Its input activates the particle for token 2.
            trainable_step = config.first_token_mode != "shared_prefill" or step > 0
            action_mask_steps.append(was_active & trainable_step)
            old_logprob_steps.append(chosen_logprobs)

            active = active & sampled.ne(int(eos_id))
            input_ids = torch.cat((input_ids, sampled[:, None]), dim=1)
            attention_mask = torch.cat((attention_mask, was_active[:, None]), dim=1)
            particle_mask = torch.cat((particle_mask, was_active[:, None]), dim=1)
            if token_type_ids is not None:
                next_types = torch.full_like(
                    sampled[:, None], int(config.generation_token_type_id)
                )
                token_type_ids = torch.cat((token_type_ids, next_types), dim=1)
            position_ids = attention_mask.long().cumsum(dim=-1).sub(1).clamp_min(0)
            if not bool(active.any()) or step + 1 == config.max_new_tokens:
                break

            if config.use_cache and past is not None:
                model_ids = sampled[:, None]
                model_particle_mask = was_active[:, None]
                model_token_types = (
                    token_type_ids[:, -1:] if token_type_ids is not None else None
                )
                model_position_ids = position_ids[:, -1:]
                model_past = past
            else:
                model_ids = input_ids
                model_particle_mask = particle_mask
                model_token_types = token_type_ids
                model_position_ids = position_ids
                model_past = None
            output = _call_particle_model(
                self.model,
                input_ids=model_ids,
                attention_mask=attention_mask,
                token_type_ids=model_token_types,
                position_ids=model_position_ids,
                particle_z=flat_z,
                particle_mask=model_particle_mask,
                use_cache=config.use_cache,
                past_key_values=model_past,
                prompt_summary=summary,
            )
            past = _get_output(output, "past_key_values")

        if bool(active.any()):
            # Truncated sequences have no EOS-producing state.  Run the final
            # sampled token once so Q sees that token's H2 state rather than the
            # preceding producer state.  EOS branches retain the state that
            # produced EOS, matching the paper's terminal-state definition.
            final_output = _call_particle_model(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                particle_z=flat_z,
                particle_mask=particle_mask,
                use_cache=False,
                prompt_summary=summary,
            )
            final_state = _terminal_state(final_output)
            latest_producer_state[active] = final_state[active]

        flat_action_ids = torch.stack(action_ids_steps, dim=1)
        flat_generated_mask = torch.stack(generated_mask_steps, dim=1)
        flat_action_mask = torch.stack(action_mask_steps, dim=1)
        flat_old_logprobs = torch.stack(old_logprob_steps, dim=1)
        time = flat_action_ids.shape[1]
        positions = (
            torch.arange(time, device=device)[None, :] + (initial_length - 1)
        ).expand(batch_size * config.k, -1)

        response_texts: List[List[str]] = []
        verification: List[List[VerificationResult]] = []
        rewards = torch.zeros(batch_size, config.k, device=device)
        for batch_index, example in enumerate(examples):
            texts: List[str] = []
            results: List[VerificationResult] = []
            for branch in range(config.k):
                flat_index = batch_index * config.k + branch
                valid_ids = flat_action_ids[flat_index][flat_generated_mask[flat_index]].tolist()
                text = _decode(self.tokenizer, valid_ids)
                result = self.verifier.verify(text, example.answer)
                texts.append(text)
                results.append(result)
                rewards[batch_index, branch] = float(result.correct)
            response_texts.append(texts)
            verification.append(results)

        rollout = ParticleRollout(
            model_input_ids=input_ids.reshape(batch_size, config.k, -1),
            attention_mask=attention_mask.reshape(batch_size, config.k, -1),
            token_type_ids=(
                token_type_ids.reshape(batch_size, config.k, -1)
                if token_type_ids is not None
                else None
            ),
            position_ids=position_ids.reshape(batch_size, config.k, -1),
            particle_mask=particle_mask.reshape(batch_size, config.k, -1),
            particle_z=particle_z,
            action_ids=flat_action_ids.reshape(batch_size, config.k, time),
            generated_mask=flat_generated_mask.reshape(batch_size, config.k, time),
            action_mask=flat_action_mask.reshape(batch_size, config.k, time),
            action_positions=positions.reshape(batch_size, config.k, time),
            old_logprobs=flat_old_logprobs.reshape(batch_size, config.k, time),
            rewards=rewards,
            prompt_summary=summary.reshape(batch_size, config.k, -1),
            terminal_states=latest_producer_state.reshape(batch_size, config.k, -1),
            response_texts=response_texts,
            example_ids=[example.example_id for example in examples],
            references=[example.answer for example in examples],
            verification=verification,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        if config.compute_reference_logprobs:
            # z=0 exactly disables the adapter, giving the frozen clean reference
            # without maintaining a duplicate 1B checkpoint in memory.
            clean_z = torch.zeros_like(rollout.particle_z)
            reference = score_actions(
                self.model, rollout, particle_z=clean_z
            ).raw_logprobs.detach()
            rollout.reference_logprobs = reference
        if was_training:
            self.model.train()
        return rollout


__all__ = [
    "ParticleRollout",
    "ParticleRolloutEngine",
    "RolloutConfig",
    "RolloutExample",
    "ScoredActions",
    "normalized_gaussian_latents",
    "sampling_logprobs",
    "score_actions",
]
