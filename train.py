"""Train one registered model: python train.py --model NAME [model flags...]

Registry (lazy imports, so a card's module is only imported when its model is asked for). Each target exposes
`main(argv: list[str] | None = None) -> None`; everything after --model NAME is passed through unchanged
(lab substitutes grid placeholders like `--lr {lr}` before this process starts).

  pvnet       M1  sentinel.model.forecast:main_pvnet      (#14)
  loadnet     M2  sentinel.model.forecast:main_loadnet    (#14)
  sitehealth  M4  sentinel.model.sitehealth:main          (#14)
  windnbm     M6  sentinel.model.wind.train:main          (#41)
  demandnet   M3  sentinel.demand.train:main              (#44)
  windpower   M5  sentinel.model.windpower.train:main     (#42)
  windfailure M7  sentinel.model.windfailure.train:main   (#43)
  gridtwin    M8  sentinel.gridtwin.train:main            (#13)
"""

from __future__ import annotations

import importlib
import sys

REGISTRY: dict[str, str] = {
    "pvnet": "sentinel.model.forecast:main_pvnet",
    "loadnet": "sentinel.model.forecast:main_loadnet",
    "sitehealth": "sentinel.model.sitehealth:main",
    "windnbm": "sentinel.model.wind.train:main",
    "demandnet": "sentinel.demand.train:main",
    "windpower": "sentinel.model.windpower.train:main",
    "windfailure": "sentinel.model.windfailure.train:main",
    "gridtwin": "sentinel.gridtwin.train:main",
}


class RegistryError(SystemExit):
    pass


def resolve(name: str):
    if name not in REGISTRY:
        raise RegistryError(f"train.py: unknown model {name!r}; known: {', '.join(sorted(REGISTRY))}")
    target = REGISTRY[name]
    mod_name, fn_name = target.split(":")
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        if e.name and (mod_name == e.name or mod_name.startswith(e.name + ".")):
            raise RegistryError(f"train.py: model {name!r} is registered to {target}, but module {e.name!r} "
                                f"does not exist yet (its card has not landed on this branch)") from e
        raise
    fn = getattr(mod, fn_name, None)
    if not callable(fn):
        raise RegistryError(f"train.py: {mod_name} has no callable {fn_name}(argv)")
    return fn


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
    name, rest = split_argv(argv)
    if name is None:
        print(__doc__)
        raise RegistryError(2)
    resolve(name)(rest)


if __name__ == "__main__":
    main()
