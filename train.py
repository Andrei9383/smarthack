"""Train one registered model: python train.py --model NAME [model flags...]

Also: python train.py --preset NAME[@sN] [extra flags]  (named configs in PRESETS for one-submit grids;
@sN adds --seed N --tag sN, so ensemble members get distinct artifact names: pvnet.sN.onnx)

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
  judge       J3  sentinel.judge.train:main               (#148, Judge validity models; team/judge-design.md §7)
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
    "judge": "sentinel.judge.train:main",
}


# Named configs, so one `lab submit --grid preset=a,b,c` can run different models / settings in parallel
# (a second submit on the same card cancels its running runs). Extra flags after --preset are appended.
PRESETS: dict[str, list[str]] = {
    "pv_now": ["--model", "pvnet", "--epochs", "150", "--lr", "1e-3", "--max-minutes", "20"],
    "pv_now_lr2e3": ["--model", "pvnet", "--epochs", "150", "--lr", "2e-3", "--max-minutes", "20"],
    "pv_now_sn0": ["--model", "pvnet", "--epochs", "150", "--lr", "1e-3", "--max-minutes", "20", "--stat-noise", "0"],
    "pv_now_sn1": ["--model", "pvnet", "--epochs", "150", "--lr", "1e-3", "--max-minutes", "20", "--stat-noise", "1.0"],
    "load_sn0": ["--model", "loadnet", "--epochs", "100", "--lr", "1e-3", "--max-minutes", "20", "--stat-noise", "0"],
    "pv_now_nores": ["--model", "pvnet", "--epochs", "150", "--lr", "1e-3", "--max-minutes", "20", "--no-residual",
                     "--pointwise", "0"],
    "pv_fc": ["--model", "pvnet", "--mode", "forecast", "--epochs", "150", "--lr", "1e-3", "--max-minutes", "20"],
    "load": ["--model", "loadnet", "--epochs", "100", "--lr", "1e-3", "--max-minutes", "20"],
    "sh": ["--model", "sitehealth", "--epochs", "200", "--patience", "15", "--max-minutes", "25"],
    # #140 experiment: 2x batch, half the steps (same samples per epoch); the step is launch-bound, so ~2x samples/s
    "pv_now_b512": ["--model", "pvnet", "--epochs", "150", "--lr", "1e-3", "--max-minutes", "20", "--batch", "512",
                    "--steps-per-epoch", "150"],
    # M6 as promoted on #99 (train#593/594): label-free normal selection, 40 epochs
    "wind": ["--model", "windnbm", "--epochs", "40", "--normal-select", "label_free"],
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
