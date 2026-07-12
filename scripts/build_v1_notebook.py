#!/usr/bin/env python3
"""Convert the checked-in percent-format V1 notebook source to .ipynb."""

from __future__ import annotations

from pathlib import Path
import hashlib
import re

import nbformat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks" / "HRM_Text_1B_Particle_V1.py"
DESTINATION = SOURCE.with_suffix(".ipynb")
MARKER = re.compile(r"^# %%(?: \[(markdown)\])?\s*$")


def _markdown(lines: list[str]) -> str:
    converted: list[str] = []
    for line in lines:
        if line == "#":
            converted.append("")
        elif line.startswith("# "):
            converted.append(line[2:])
        elif line.startswith("#"):
            converted.append(line[1:])
        elif not line.strip():
            converted.append("")
        else:
            raise ValueError(f"markdown cell contains a non-comment line: {line!r}")
    return "\n".join(converted).rstrip() + "\n"


def build() -> Path:
    raw = SOURCE.read_text(encoding="utf-8").splitlines()
    cells: list[tuple[str, list[str]]] = []
    kind: str | None = None
    current: list[str] = []
    for line in raw:
        match = MARKER.match(line)
        if match:
            if kind is not None:
                cells.append((kind, current))
            kind = "markdown" if match.group(1) else "code"
            current = []
        elif kind is None:
            if line.strip():
                raise ValueError("notebook source must begin with a cell marker")
        else:
            current.append(line)
    if kind is not None:
        cells.append((kind, current))

    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook_cells = []
    used_ids: set[str] = set()
    for index, (kind, lines) in enumerate(cells):
        source = (
            _markdown(lines)
            if kind == "markdown"
            else "\n".join(lines).rstrip() + "\n"
        )
        cell = (
            nbformat.v4.new_markdown_cell(source)
            if kind == "markdown"
            else nbformat.v4.new_code_cell(source)
        )
        cell_id = hashlib.sha256(f"{index}:{kind}:{source}".encode()).hexdigest()[:12]
        if cell_id in used_ids:
            raise AssertionError("deterministic notebook cell ID collision")
        used_ids.add(cell_id)
        cell["id"] = cell_id
        notebook_cells.append(cell)
    notebook["cells"] = notebook_cells
    nbformat.validate(notebook)
    nbformat.write(notebook, DESTINATION)
    return DESTINATION


if __name__ == "__main__":
    print(build())
