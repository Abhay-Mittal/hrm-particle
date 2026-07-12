from __future__ import annotations

import json

import pytest

import hrm_particle.v1_dependencies as dependencies


class Distribution:
    def __init__(self, payload: object | None) -> None:
        self.payload = payload

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return None if self.payload is None else json.dumps(self.payload)


def test_pinned_vcs_commit_reads_pep610_and_matches(monkeypatch) -> None:
    commit = "1" * 40
    monkeypatch.setattr(
        dependencies.importlib.metadata,
        "distribution",
        lambda _: Distribution({"vcs_info": {"commit_id": commit}}),
    )
    assert dependencies.installed_vcs_commit("open-r1") == commit
    assert dependencies.require_installed_vcs_commit("open-r1", commit) == commit
    with pytest.raises(RuntimeError, match="mismatch"):
        dependencies.require_installed_vcs_commit("open-r1", "2" * 40)


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"vcs_info": {}}, {"vcs_info": {"commit_id": "main"}}],
)
def test_pinned_vcs_commit_rejects_unpinned_installs(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        dependencies.importlib.metadata,
        "distribution",
        lambda _: Distribution(payload),
    )
    with pytest.raises(RuntimeError):
        dependencies.installed_vcs_commit("open-r1")
