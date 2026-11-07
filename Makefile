# Prosumer Sentinel: clean checkout -> every number the presentation shows.
#
#   make reproduce        setup + tests + simulator + baselines + eval + demand + uncertainty + replay, then a provenance summary
#   make help             list the targets and variables
#
# Offline after `make setup` (uv / pip download the Python wheels once): every raw input is committed in data/raw/.
# Outputs go to $(OUT) (gitignored). Committed files (demo/, models/) are never overwritten unless you ask (PROMOTE=1).
# Steps whose code has not reached main yet are skipped with the card that brings them, so the target stays green.

SHELL       := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PY_VERSION   ?= 3.12
VENV         ?= .venv
PYTHON       ?= $(VENV)/bin/python
OUT          ?= .lab-out/reproduce
LOGS         := $(OUT)/logs
# Seed for replay, demand and the uncertainty demo (the simulator keeps its own defaults: fleet 7, faults 0).
SEED         ?= 0
# FULL=1: fault injector on the full 600-site fleet (~4 min) instead of the quick 120-site fleet (~15 s).
FULL         ?= 0
# REPLAY_WORLD: mock (same as lab.toml [runs.replay]) | sim (the #11 fleet that `make sim` builds).
REPLAY_WORLD ?= mock
REPLAY_DAYS  ?= 7
# REPLAY_OUT may point at a committed file (demo/sentinel.sqlite, #29) only with PROMOTE=1.
REPLAY_OUT   ?= $(OUT)/sentinel.sqlite
PROMOTE      ?= 0
# EVAL_ARGS: extra eval.py flags, e.g. --quick (150-site fleet) or --drills (#28).
EVAL_ARGS    ?=
TRAIN_LOCAL  ?= 0
# MODEL: pvnet | loadnet | sitehealth | windnbm | demandnet | windpower | windfailure | gridtwin (train.py registry)
MODEL        ?= pvnet
TRAIN_ARGS   ?=
CARD         ?= 14
# APP_DB: sqlite file for `make app`; empty = demo/sentinel.sqlite, else demo/fixtures.sqlite.
APP_DB       ?=
comma        := ,

STAMP := $(VENV)/.reqs-stamp

# $(call step,name,command): run, tee to $(LOGS)/name.log, record wall time (read back by `summary`).
define step
@mkdir -p $(LOGS); printf '\n==> %s\n' "$(1)"; t0=$$(date +%s); \
	{ $(2) ; } 2>&1 | tee $(LOGS)/$(1).log; \
	echo "$$(( $$(date +%s) - t0 ))" > $(LOGS)/$(1).time
endef

# $(call skip,name,message): log a skipped step so the summary shows it.
define skip
@mkdir -p $(LOGS); printf '\n==> %s\n' "$(1)"; echo "skipped: $(2)" | tee $(LOGS)/$(1).log; echo "-" > $(LOGS)/$(1).time
endef

.PHONY: help reproduce setup test sim baselines eval demand uncertainty replay train app fixtures summary clean clean-venv

help:
	@echo "Prosumer Sentinel"
	@echo "  make reproduce   setup, test, sim, baselines, eval, demand, uncertainty, replay, summary (~8 min)"
	@echo "  make setup       $(VENV) with Python $(PY_VERSION) + requirements.txt (uv, else python$(PY_VERSION) -m venv)"
	@echo "  make test        pytest tests/"
	@echo "  make sim         RO prosumer fleet (#11) + PV fault injector (#12)        -> $(OUT)/fleet, $(OUT)/faults"
	@echo "  make baselines   eval.py --baselines: pvlib, persistence, LightGBM, CUSUM  -> $(OUT)/baselines"
	@echo "  make eval        eval.py: model test-set metrics by label (#16)            -> $(OUT)/eval"
	@echo "  make demand      M3 DemandNet on real RO load (#44)                        -> $(OUT)/demand"
	@echo "  make uncertainty conformal + P10 commitment demo (#27)                    -> $(OUT)/uncertainty.json"
	@echo "  make replay      baseline vs Sentinel test week through the agent (#17)    -> $(REPLAY_OUT)"
	@echo "  make train       how training runs on the lab; TRAIN_LOCAL=1 MODEL=pvnet trains here"
	@echo "  make app         npm ci + build + start the web app (APP_DB=<sqlite> to pick the DB)"
	@echo "  make summary     re-print the provenance table of the last reproduce"
	@echo "  make clean       remove $(OUT) (make clean-venv also removes $(VENV))"
	@echo "Variables: OUT SEED FULL REPLAY_WORLD REPLAY_DAYS REPLAY_OUT PROMOTE EVAL_ARGS TRAIN_LOCAL MODEL TRAIN_ARGS APP_DB"

reproduce:
	@start=$$(date +%s); \
	$(MAKE) --no-print-directory setup test sim baselines eval demand uncertainty replay; \
	mkdir -p $(LOGS); echo "$$(( $$(date +%s) - start ))" > $(LOGS)/total.time; \
	$(MAKE) --no-print-directory summary

# ---------------------------------------------------------------------------------------------------------- setup
setup: $(STAMP)

$(STAMP): requirements.txt
	@if command -v uv >/dev/null 2>&1; then \
	  [ -x $(PYTHON) ] || uv venv -q --python $(PY_VERSION) $(VENV); \
	  uv pip install -q --python $(PYTHON) -r requirements.txt; \
	else \
	  echo "uv not found; using python$(PY_VERSION) -m venv + pip (slower; https://docs.astral.sh/uv/ to install uv)"; \
	  [ -x $(PYTHON) ] || python$(PY_VERSION) -m venv $(VENV); \
	  $(PYTHON) -m pip install -q -r requirements.txt; \
	fi
	@$(PYTHON) -c 'import sys; v = sys.version_info; assert v[:2] == (3, 12), f"need Python 3.12, got {v}"; print("venv ok:", sys.version.split()[0])'
	@touch $(STAMP)

# ----------------------------------------------------------------------------------------------------------- test
test: setup
	$(call step,test,$(PYTHON) -m pytest -q tests)

# ------------------------------------------------------------------------------------------------------------ sim
# Full fleet (600 sites, 15 feeders, 2022-2025, ~30 s, ~0.7 GB) for `REPLAY_WORLD=sim`; the fault injector runs on
# the quick fleet by default (120 sites, 2024) because the full one takes minutes. Both are deterministic (seeded).
sim: setup
	$(call step,sim,\
	  $(PYTHON) -m sentinel.sim.fleet --no-plots --out $(OUT)/fleet && \
	  if [ "$(FULL)" = 1 ]; then F=$(OUT)/fleet; \
	  else F=$(OUT)/fleet-quick; $(PYTHON) -m sentinel.sim.fleet --quick --no-plots --out $$F; fi && \
	  $(PYTHON) -m sentinel.sim.faults --fleet $$F --out $(OUT)/faults)

# ------------------------------------------------------------------------------------------------------ baselines
# The numbers the models must beat (PV / load / fault: #15; wind WindMon rules: #24). eval.py builds the 600-site
# test fleet + injected faults once (~4 min) under $(OUT)/cache; `eval` reuses it.
baselines: setup
ifneq ($(wildcard eval.py),)
	$(call step,baselines,mkdir -p $(OUT)/baselines && \
	  SENTINEL_CACHE=$(abspath $(OUT))/cache LAB_OUTPUT_DIR=$(OUT)/baselines $(PYTHON) eval.py --baselines $(EVAL_ARGS))
else ifneq ($(wildcard sentinel/eval/baselines/pv/run.py),)
	$(call step,baselines,mkdir -p $(OUT)/baselines && \
	  SENTINEL_CACHE=$(abspath $(OUT))/cache $(PYTHON) -m sentinel.eval.baselines.pv.run --seed $(SEED) \
	    --out $(OUT)/baselines && \
	  if [ ! -d sentinel/eval/baselines/wind ]; then echo "note: wind baselines not on main yet (#24)"; fi)
else
	$(call skip,baselines,PV baselines not on main yet (#15))
endif

# ----------------------------------------------------------------------------------------------------------- eval
# Model test-set metrics grouped by label. Scores models/*.onnx (#29) when present, else the MockPredictor (a
# pipeline check, not a result: eval.py prints which predictor it loaded).
eval: setup
ifeq ($(wildcard eval.py),)
	$(call skip,eval,eval.py not on main yet (#16); model numbers in the README cite lab train runs (#14 / #41))
else
	$(call step,eval,mkdir -p $(OUT)/eval && \
	  SENTINEL_CACHE=$(abspath $(OUT))/cache LAB_OUTPUT_DIR=$(OUT)/eval $(PYTHON) eval.py $(EVAL_ARGS))
endif

# --------------------------------------------------------------------------------------------------------- demand
demand: setup
	$(call step,demand,$(PYTHON) -m sentinel.demand --seed $(SEED) --out $(OUT)/demand)

# ---------------------------------------------------------------------------------------------------- uncertainty
uncertainty: setup
	$(call step,uncertainty,$(PYTHON) -m sentinel.eval.uncertainty --seed $(SEED) \
	  --calibration-out $(OUT)/calibration --report $(OUT)/uncertainty.json)

# --------------------------------------------------------------------------------------------------------- replay
replay: setup
	@if git ls-files --error-unmatch "$(REPLAY_OUT)" >/dev/null 2>&1 && [ "$(PROMOTE)" != 1 ]; then \
	  echo "refusing to overwrite committed $(REPLAY_OUT); rerun with PROMOTE=1 if you mean it"; exit 1; fi
	@if [ "$(REPLAY_WORLD)" = sim ] && [ ! -f $(OUT)/fleet/sites.parquet ]; then \
	  echo "REPLAY_WORLD=sim needs the fleet: run 'make sim' first"; exit 1; fi
	$(call step,replay,\
	  mkdir -p $$(dirname $(REPLAY_OUT)) && \
	  SENTINEL_DATA=$(abspath $(OUT))/fleet $(PYTHON) replay.py --policy baseline$(comma)sentinel \
	    --world $(REPLAY_WORLD) --days $(REPLAY_DAYS) --seed $(SEED) --out $(REPLAY_OUT))

# ---------------------------------------------------------------------------------------------------------- train
# Training runs on the lab's GPU workers, not here: models fit a 4 GB GPU; final runs used the 12 GB RTX 5070.
train:
	@echo "Training runs on the lab (lab.toml [runs.train] / [runs.train_cpu]); results come back to the card:"
	@echo "  lab submit $(CARD) --run train --cmd \".venv/bin/python train.py --model $(MODEL)\""
	@echo "  lab submit 14 --run train --cmd \".venv/bin/python train.py --preset {p}\" --grid p=pv_now@s0,pv_now@s1,load,sh"
	@echo "  lab submit 44 --run train_cpu --cmd \".venv/bin/python train.py --model demandnet\""
	@echo "Local (needs requirements-train.txt, i.e. torch): make train TRAIN_LOCAL=1 MODEL=pvnet TRAIN_ARGS='--max-minutes 5'"
ifeq ($(TRAIN_LOCAL),1)
ifeq ($(wildcard train.py),)
	$(call skip,train,train.py not on main yet (#14 merges it; #41 adds windnbm))
else
	@$(MAKE) --no-print-directory setup
	@if command -v uv >/dev/null 2>&1; then uv pip install -q --python $(PYTHON) -r requirements-train.txt; \
	  else $(PYTHON) -m pip install -q -r requirements-train.txt; fi
	$(call step,train-$(MODEL),mkdir -p $(OUT)/train/$(MODEL) && \
	  LAB_OUTPUT_DIR=$(OUT)/train/$(MODEL) $(PYTHON) train.py --model $(MODEL) $(TRAIN_ARGS))
endif
endif

# ------------------------------------------------------------------------------------------------------------ app
# The app reads demo/sentinel.sqlite (promoted replay, #29), else demo/fixtures.sqlite (#30), unless APP_DB is set.
app:
	@[ -d node_modules ] || npm ci --no-audit --no-fund
	@if [ -z "$(APP_DB)" ] && [ ! -f demo/sentinel.sqlite ] && [ ! -f demo/fixtures.sqlite ]; then \
	  python3 demo/fixtures/make_fixtures.py; fi
	npm run build
	@echo "serving on http://localhost:$${PORT:-3000}  (DB: $(if $(APP_DB),$(abspath $(APP_DB)),demo/sentinel.sqlite else demo/fixtures.sqlite))"
	$(if $(APP_DB),DATABASE_URL=file:$(abspath $(APP_DB)),) npm start

fixtures:
	python3 demo/fixtures/make_fixtures.py

# -------------------------------------------------------------------------------------------------------- summary
# Every number printed by reproduce, with the step that produced it, its label and the card / lab run it backs.
summary:
	@L=$(LOGS); [ -d $$L ] || { echo "no logs in $$L: run make reproduce"; exit 1; }; \
	label() { case $$1 in \
	  test) echo "n/a: unit + contract tests";; \
	  sim) echo "synthetic RO (driven by real RO ERA5 in data/raw); #11, #12";; \
	  baselines) echo "label per block below (synthetic_RO test 2025 / non_RO PVDAQ / real_RO); #15 via eval.py";; \
	  eval) echo "label per block below; ONNX from models/ if present, else MockPredictor = pipeline check; #16";; \
	  demand) echo "real RO: Energy-Charts/ENTSO-E RO load + ERA5 temp, test 2025; #44";; \
	  uncertainty) echo "synthetic (MockPredictor truth): method check, not a model result; #27";; \
	  replay) echo "synthetic RO; components marked real vs FALLBACK below; #17 (lab [runs.replay])";; \
	  *) echo "";; esac; }; \
	echo; echo "================ make reproduce: where every number came from ================"; \
	for s in test sim baselines eval demand uncertainty replay; do \
	  [ -f $$L/$$s.log ] || continue; t=$$(cat $$L/$$s.time 2>/dev/null || echo "?"); \
	  if grep -q '^skipped:' $$L/$$s.log; then st="SKIPPED"; else st="ran in $${t}s"; fi; \
	  printf '\n[%s] %s  |  %s\n' "$$s" "$$st" "$$(label $$s)"; \
	  printf '    log: %s\n' "$$L/$$s.log"; \
	  grep -E '^skipped:' $$L/$$s.log | sed 's/^/    /' || true; \
	  case $$s in \
	    test) grep -E '[0-9]+ passed' $$L/$$s.log | tail -1 | sed 's/^/    /' || true;; \
	    sim) grep -E "^wrote " $$L/$$s.log | sed "s/^/    /" || true;; \
	    replay) grep -E '^\[agent\] (world|FALLBACK)|^# replay:' $$L/$$s.log | sed 's/^/    /' || true;; \
	  esac; \
	  if grep -q '^== ' $$L/$$s.log; then grep -E '^predictors:' $$L/$$s.log | sed 's/^/    /' || true; \
	    sed -n '/^== /,/^eval report:/p' $$L/$$s.log | sed 's/^/    /'; \
	  else grep -E '^LAB_METRIC ' $$L/$$s.log | sed 's/^LAB_METRIC /    /' || true; fi; \
	done; \
	echo; echo "outputs: $(OUT)/  (replay DB: $(REPLAY_OUT))"; \
	[ -f $$L/total.time ] && echo "total: $$(cat $$L/total.time)s" || true; \
	echo "Model numbers (M1/M2/M4/M6) come from lab train runs until #29 commits the ONNX weights for eval.py: README.md#results"

# ---------------------------------------------------------------------------------------------------------- clean
clean:
	rm -rf $(OUT)

clean-venv: clean
	rm -rf $(VENV)
