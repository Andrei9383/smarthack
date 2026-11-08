"""Train one registered model: python train.py --model NAME [model flags...]

Also: python train.py --preset NAME[@sN] [extra flags]  (named configs in PRESETS; @sN adds --seed N --tag sN,
so ensemble members get distinct artifact names: pvnet.sN.onnx)

Each REGISTRY target is imported lazily and exposes `main(argv: list[str] | None = None) -> None`; everything after
--model NAME is passed through unchanged.

  pvnet       M1  PV forecast                 demandnet   M3  zone demand
  loadnet     M2  household load forecast     windpower   M5  wind power forecast
  sitehealth  M4  PV site health              windfailure M7  wind component failure
  windnbm     M6  wind normal behaviour       gridtwin    M8  feeder grid twin
  battery     I4  data-v2 battery behaviour   judge       J3  Judge validity models
"""

from __future__ import annotations

import importlib
import sys

REGISTRY: dict[str, str] = {
    "pvnet": "wattsentry.model.forecast:main_pvnet",
    "loadnet": "wattsentry.model.forecast:main_loadnet",
    "sitehealth": "wattsentry.model.sitehealth:main",
    "windnbm": "wattsentry.model.wind.train:main",
    "demandnet": "wattsentry.demand.train:main",
    "windpower": "wattsentry.model.windpower.train:main",
    "windfailure": "wattsentry.model.windfailure.train:main",
    "gridtwin": "wattsentry.gridtwin.run:main",
    "battery": "wattsentry.model.battery.train:main",
    "judge": "wattsentry.judge.train:main",
}


# Named configs (one `lab submit --grid p=a,b,c` runs several in parallel); extra flags after --preset are appended.
# pv_now / load / sh / pv_fc trained the committed models/, wind is the promoted M6 config.
PRESETS: dict[str, list[str]] = {
    "pv_now": ["--model", "pvnet", "--epochs", "150", "--lr", "1e-3", "--max-minutes", "20"],
    "pv_fc": ["--model", "pvnet", "--mode", "forecast", "--epochs", "150", "--lr", "1e-3", "--max-minutes", "20"],
    "load": ["--model", "loadnet", "--epochs", "100", "--lr", "1e-3", "--max-minutes", "20"],
    "sh": ["--model", "sitehealth", "--epochs", "200", "--patience", "15", "--max-minutes", "25"],
    "wind": ["--model", "windnbm", "--epochs", "40", "--normal-select", "label_free"],
}


class RegistryError(SystemExit):
    pass


def resolve(name: str):
    if name not in REGISTRY:
        raise RegistryError(f"train.py: unknown model {name!r}; known: {', '.join(sorted(REGISTRY))}")
    mod_name, fn_name = REGISTRY[name].split(":")
    return getattr(importlib.import_module(mod_name), fn_name)


def split_argv(argv: list[str]) -> tuple[str | None, list[str]]:
    rest, name, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--model" and i + 1 < len(argv):
            name, i = argv[i + 1], i + 2
            continue
        if a.startswith("--model="):
            name, i = a.split("=", 1)[1], i + 1
            continue
        rest.append(a)
        i += 1
    return name, rest


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--preset" in argv:
        i = argv.index("--preset")
        pname = argv[i + 1] if i + 1 < len(argv) else ""
        base, _, seed = pname.partition("@s")  # "pv_now@s2" = preset pv_now with --seed 2 --tag s2
        if base not in PRESETS or (seed and not seed.isdigit()):
            raise RegistryError(f"train.py: --preset needs one of {', '.join(PRESETS)} (optionally NAME@sN)")
        extra = ["--seed", seed, "--tag", f"s{seed}"] if seed else []
        argv = PRESETS[base] + extra + argv[:i] + argv[i + 2:]
    name, rest = split_argv(argv)
    if name is None:
        print(__doc__)
        raise RegistryError(2)
    resolve(name)(rest)


if __name__ == "__main__":
    main()
