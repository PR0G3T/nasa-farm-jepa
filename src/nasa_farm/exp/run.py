"""Experiment registry + serialized, self-describing reports.

`run(name, spec)` dispatches to a registered experiment and wraps its
result in a `Report` carrying the full spec, wall time and environment —
every saved artifact replays and cites itself. `flat`/`latex` flatten the
nested metrics into paper-ready rows so results flow into manuscripts
without hand transcription.

New experiments register a `Spec → dict` callable, the same way forcing
providers register into `forcing.provide` — the CLI and reports pick them
up with zero wiring.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from nasa_farm.exp.spec import Spec

_REGISTRY: dict[str, Callable[[Spec], dict]] = {}


def register(name: str):
    """`@register("name")` on a `Spec → dict` experiment runner."""
    def deco(fn: Callable[[Spec], dict]) -> Callable[[Spec], dict]:
        _REGISTRY[name] = fn
        return fn
    return deco


def experiments() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


@dataclass(frozen=True, slots=True)
class Report:
    name: str
    spec: dict
    results: dict
    seconds: float
    meta: dict

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dataclasses.asdict(self), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Report":
        return cls(**json.loads(Path(path).read_text()))

    def flat(self) -> Iterator[tuple[str, object]]:
        """Dotted-key leaves of the results tree (scalars only)."""
        def walk(d: dict, pre: str) -> Iterator[tuple[str, object]]:
            for k, v in d.items():
                key = f"{pre}.{k}" if pre else str(k)
                if isinstance(v, dict):
                    yield from walk(v, key)
                elif not isinstance(v, list) and v is not None:
                    yield key, v
        yield from walk(self.results, "")

    def latex(self) -> str:
        """Booktabs rows of every scalar result — paste into a longtable."""
        rows = ["\\begin{longtable}{lr}", "\\toprule",
                "Metric & Value \\\\", "\\midrule", "\\endfirsthead",
                "\\toprule Metric & Value \\\\ \\midrule \\endhead"]
        for k, v in self.flat():
            s = f"{v:.4g}" if isinstance(v, float) else str(v)
            rows.append(f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & {s} \\\\")
        rows += ["\\bottomrule", "\\end{longtable}"]
        return "\n".join(rows)


def run(name: str, spec: Spec) -> Report:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown experiment {name!r}; have {experiments()}")
    t0 = time.perf_counter()
    results = _REGISTRY[name](spec)
    meta = {"python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "argv": sys.argv[1:]}
    return Report(name=name, spec=spec.as_dict(), results=results,
                  seconds=round(time.perf_counter() - t0, 3), meta=meta)
