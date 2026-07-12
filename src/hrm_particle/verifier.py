"""Safe, exact verification for the arithmetic proof-of-concept.

The verifier deliberately accepts only a small arithmetic language.  It never
calls :func:`eval`, never imports names from an expression, and places limits on
input size, AST depth, and exponent size.  Accepted values are converted to
``fractions.Fraction`` so comparisons of integers, decimals, and ratios are
exact rather than floating point comparisons.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Optional


_FINAL_MARKER_RE = re.compile(
    r"(?:final\s+answer|answer|ans)\s*(?:is|=|:)?\s*(.+)", re.IGNORECASE
)
_LATEX_FRAC_RE = re.compile(
    r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}"
)
_NUMBERISH_RE = re.compile(
    r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"(?:\s*/\s*[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)?"
    r"\s*%?"
)


class UnsafeArithmeticExpression(ValueError):
    """Raised when an answer is outside the verifier's arithmetic language."""


@dataclass(frozen=True)
class VerificationResult:
    """Structured verifier output useful for audits and dataset filtering."""

    correct: bool
    predicted: Optional[Fraction]
    expected: Optional[Fraction]
    predicted_text: str
    expected_text: str
    error: Optional[str] = None


def _strip_balanced_box(text: str) -> Optional[str]:
    """Return the content of the final ``\boxed{...}``, including nesting."""

    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for match in reversed(starts):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index]
    return None


def extract_answer_text(text: object) -> str:
    """Extract the most likely final arithmetic answer from model text.

    Explicit ``\boxed{}`` and ``Final answer:`` forms take precedence.  As a
    conservative fallback, only the final numeric/fraction-like span is used.
    """

    raw = str(text).strip()
    boxed = _strip_balanced_box(raw)
    if boxed is not None:
        return boxed.strip()

    marker_matches = list(_FINAL_MARKER_RE.finditer(raw))
    if marker_matches:
        candidate = marker_matches[-1].group(1).strip().splitlines()[0].strip()
        return candidate

    # Preserve a standalone arithmetic expression in full; taking only its
    # final number would turn ``(2+3)/10`` into ``10`` and could also bypass the
    # exponent/depth safety limits.
    if re.fullmatch(r"[\d\s.,eE+\-*/^()%]+", raw):
        return raw
    # Executable-looking syntax and malformed LaTeX must reach the safe parser
    # intact and be rejected, not be reduced to a coincidental trailing number.
    if any(token in raw for token in ("__", "'", '"', "[", "]", "\\", "(")):
        return raw

    candidates = list(_NUMBERISH_RE.finditer(raw))
    return candidates[-1].group(0).strip() if candidates else raw


def _replace_simple_latex_fractions(text: str) -> str:
    # Repeated substitution handles simple nested fractions from the inside out.
    previous = None
    current = text
    for _ in range(12):
        if current == previous:
            break
        previous = current
        current = _LATEX_FRAC_RE.sub(r"((\1)/(\2))", current)
    return current


def _prepare_expression(text: object) -> tuple[str, bool]:
    value = extract_answer_text(text)
    value = _replace_simple_latex_fractions(value)
    value = value.replace("\u2212", "-").replace("\u00d7", "*").replace("\u00f7", "/")
    value = value.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    value = value.replace("^", "**").replace("$", "").strip()
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = value.rstrip(". ,;!\t\r\n")
    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1].strip()
    if len(value) > 256:
        raise UnsafeArithmeticExpression("answer expression is too long")
    if not value:
        raise UnsafeArithmeticExpression("empty answer")
    return value, is_percent


def _number_to_fraction(node: ast.Constant, source: str) -> Fraction:
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsafeArithmeticExpression("only numeric literals are allowed")
    if isinstance(value, int):
        return Fraction(value)
    # Preserve every decimal digit from the original token rather than the
    # binary float stored by ``ast``.
    token = ast.get_source_segment(source, node)
    try:
        return Fraction(Decimal(token if token is not None else str(value)))
    except InvalidOperation as exc:
        raise UnsafeArithmeticExpression("invalid numeric literal") from exc


def _eval_ast(node: ast.AST, source: str, *, depth: int = 0) -> Fraction:
    if depth > 24:
        raise UnsafeArithmeticExpression("arithmetic expression is too deeply nested")

    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, source, depth=depth + 1)
    if isinstance(node, ast.Constant):
        return _number_to_fraction(node, source)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_ast(node.operand, source, depth=depth + 1)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left, source, depth=depth + 1)
        right = _eval_ast(node.right, source, depth=depth + 1)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise UnsafeArithmeticExpression("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 12:
                raise UnsafeArithmeticExpression("exponent must be an integer with magnitude <= 12")
            if left == 0 and right < 0:
                raise UnsafeArithmeticExpression("division by zero")
            result = left ** right.numerator
            if result.numerator.bit_length() > 4096 or result.denominator.bit_length() > 4096:
                raise UnsafeArithmeticExpression("arithmetic result is too large")
            return result
        raise UnsafeArithmeticExpression("operator is not allowed")
    # Names, calls, attributes, containers, comprehensions, and subscripts all
    # arrive here and are rejected.
    raise UnsafeArithmeticExpression(f"syntax {type(node).__name__} is not allowed")


def normalize_numeric(text: object) -> Fraction:
    """Normalize an arithmetic answer to an exact :class:`Fraction`.

    Examples: ``"0.5"``, ``"1/2"``, ``"50%"`` and ``"\\frac{1}{2}"``
    all normalize to ``Fraction(1, 2)``.
    """

    expression, is_percent = _prepare_expression(text)
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        # Decimal provides a clearer path for values ast may reject, while still
        # accepting no executable syntax.
        try:
            result = Fraction(Decimal(expression))
        except (InvalidOperation, ValueError, ZeroDivisionError) as decimal_exc:
            raise UnsafeArithmeticExpression("invalid arithmetic answer") from decimal_exc
    else:
        result = _eval_ast(tree, expression)
    return result / 100 if is_percent else result


class ExactArithmeticVerifier:
    """Binary exact-answer verifier used as the sole actor reward source."""

    def verify(self, prediction: object, reference: object) -> VerificationResult:
        predicted_text = extract_answer_text(prediction)
        expected_text = extract_answer_text(reference)
        try:
            predicted = normalize_numeric(predicted_text)
            expected = normalize_numeric(expected_text)
        except (UnsafeArithmeticExpression, ArithmeticError, ValueError) as exc:
            return VerificationResult(
                correct=False,
                predicted=None,
                expected=None,
                predicted_text=predicted_text,
                expected_text=expected_text,
                error=str(exc),
            )
        return VerificationResult(
            correct=predicted == expected,
            predicted=predicted,
            expected=expected,
            predicted_text=predicted_text,
            expected_text=expected_text,
        )

    def __call__(self, prediction: object, reference: object) -> float:
        return float(self.verify(prediction, reference).correct)


__all__ = [
    "ExactArithmeticVerifier",
    "UnsafeArithmeticExpression",
    "VerificationResult",
    "extract_answer_text",
    "normalize_numeric",
]
