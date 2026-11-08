# WattSentry: from a clean checkout to every number the app shows.
#
#   make reproduce    setup, test, sim, eval, demand, uncertainty, replay, replay-demo, then a summary
#   make app          build and serve the web app
#   make help         list all targets and variables
#
# Offline after `make setup`: every raw input is committed in data/raw/. Outputs go to $(OUT) (gitignored).
# Committed files in demo/ and models/ are never overwritten unless PROMOTE=1.

SHELL         := /bin/bash
.SHELLFLAGS   := -eu -o pipefail -c
.DEFAULT_GOAL := help

PY_VERSION   ?= 3.12
VENV         ?= .venv
PYTHON       ?= $(VENV)/bin/python
OUT          ?= .lab-out/reproduce
LOGS         := $(OUT)/logs
CACHE        := $(abspath $(OUT))/cache
SEED         ?= 0
# 1: fault injector on the full 600-site fleet (~4 min) instead of the quick one
FULL         ?= 0
# mock | sim (sim needs the caches `make replay-demo` builds)
REPLAY_WORLD ?= mock
REPLAY_DAYS  ?= 7
REPLAY_OUT   ?= $(OUT)/wattsentry.sqlite
DEMO_OUT     ?= $(OUT)/demo/wattsentry.sqlite
# 0: reproduce skips replay-demo
DEMO         ?= 1
PROMOTE      ?= 0
# e.g. --quick for a 150-site smoke test
EVAL_ARGS    ?=
# see train.py --help
MODEL        ?= pvnet
TRAIN_ARGS   ?=
# sqlite file for `make app`; default demo/wattsentry.sqlite
APP_DB       ?=
comma        := ,
STAMP        := $(VENV)/.reqs-stamp
STEPS        := test sim baselines eval demand uncertainty replay replay-demo

# $(call step,name,command): run, tee to $(LOGS)/name.log and record the wall time.
define step
@mkdir -p $(LOGS); printf '\n==> %s\n' "$(1)"; t0=$$(date +%s); \
	{ $(2) ; } 2>&1 | tee $(LOGS)/$(1).log; \
	echo "$$(( $$(date +%s) - t0 ))" > $(LOGS)/$(1).time
endef

# Refuse to overwrite a committed file unless PROMOTE=1.
define guard
@if git ls-files --error-unmatch "$(1)" >/dev/null 2>&1 && [ "$(PROMOTE)" != 1 ]; then \
	  echo "refusing to overwrite committed $(1); rerun with PROMOTE=1"; exit 1; fi
endef

.PHONY: help reproduce setup test sim baselines eval demand uncertainty replay replay-demo train app app-600 \
        fixtures summary clean clean-venv

help:
	@echo "WattSentry"
	@echo "  make reproduce    setup + $(STEPS) + summary (DEMO=0 skips replay-demo)"
	@echo "  make setup        $(VENV) with Python $(PY_VERSION) + requirements.txt"
	@echo "  make test         pytest"
	@echo "  make sim          prosumer fleet + PV fault injector        -> $(OUT)/fleet, $(OUT)/faults"
	@echo "  make baselines    pvlib, persistence, LightGBM, CUSUM      -> $(OUT)/baselines"
	@echo "  make eval         models/ against every baseline + drills  -> $(OUT)/eval"
	@echo "  make demand       DemandNet on real Romanian load          -> $(OUT)/demand"
	@echo "  make uncertainty  conformal calibration + P10 commitment   -> $(OUT)/uncertainty.json"
	@echo "  make replay       baseline vs WattSentry on a mock world   -> $(REPLAY_OUT)"
	@echo "  make replay-demo  the demo replay with models/             -> $(DEMO_OUT)"
	@echo "  make train        train one model locally (MODEL=pvnet, needs requirements-train.txt)"
	@echo "  make app          build and serve the web app (APP_DB=<sqlite> to pick the database)"
	@echo "  make app-600      the app on the 600-home replay"
	@echo "  make fixtures     mock database for the app (demo/fixtures.sqlite)"
	@echo "  make clean        remove $(OUT) (clean-venv also removes $(VENV))"
	@echo "Variables: OUT SEED FULL REPLAY_WORLD REPLAY_DAYS REPLAY_OUT DEMO DEMO_OUT PROMOTE EVAL_ARGS MODEL TRAIN_ARGS APP_DB"

reproduce:
	@start=$$(date +%s); \
	$(MAKE) --no-print-directory setup test sim eval demand uncertainty replay $(if $(filter 1,$(DEMO)),replay-demo,); \
	mkdir -p $(LOGS); echo "$$(( $$(date +%s) - start ))" > $(LOGS)/total.time; \
	$(MAKE) --no-print-directory summary

setup: $(STAMP)

$(STAMP): requirements.txt
	@if command -v uv >/dev/null 2>&1; then \
	  [ -x $(PYTHON) ] || uv venv -q --python $(PY_VERSION) $(VENV); \
	  uv pip install -q --python $(PYTHON) -r requirements.txt; \
	else \
	  [ -x $(PYTHON) ] || python$(PY_VERSION) -m venv $(VENV); \
	  $(PYTHON) -m pip install -q -r requirements.txt; \
	fi
	@touch $(STAMP)

test: setup
	$(call step,test,$(PYTHON) -m pytest -q tests)

# Full fleet (600 sites, 2022-2025) for REPLAY_WORLD=sim; the fault injector runs on the quick fleet unless FULL=1.
sim: setup
	$(call step,sim,\
	  $(PYTHON) -m wattsentry.sim.fleet --no-plots --out $(OUT)/fleet && \
	  if [ "$(FULL)" = 1 ]; then F=$(OUT)/fleet; \
	  else F=$(OUT)/fleet-quick; $(PYTHON) -m wattsentry.sim.fleet --quick --no-plots --out $$F; fi && \
	  $(PYTHON) -m wattsentry.sim.faults --fleet $$F --out $(OUT)/faults)

# `make eval` prints the same baselines next to the models, so reproduce skips this target.
baselines: setup
	$(call step,baselines,mkdir -p $(OUT)/baselines && \
	  WATTSENTRY_CACHE=$(CACHE) LAB_OUTPUT_DIR=$(OUT)/baselines $(PYTHON) eval.py --baselines $(EVAL_ARGS))

eval: setup
	$(call step,eval,mkdir -p $(OUT)/eval && \
	  WATTSENTRY_CACHE=$(CACHE) LAB_OUTPUT_DIR=$(OUT)/eval \
	  $(PYTHON) eval.py --predictor onnx$(comma)baselines --models models --drills $(EVAL_ARGS))

demand: setup
	$(call step,demand,$(PYTHON) -m wattsentry.demand --seed $(SEED) --out $(OUT)/demand)

uncertainty: setup
	$(call step,uncertainty,$(PYTHON) -m wattsentry.eval.uncertainty --seed $(SEED) \
	  --calibration-out $(OUT)/calibration --report $(OUT)/uncertainty.json)

replay: setup
	$(call guard,$(REPLAY_OUT))
	@if [ "$(REPLAY_WORLD)" = sim ] && ! ls $(CACHE)/data-v*/windows/index.json >/dev/null 2>&1; then \
	  echo "REPLAY_WORLD=sim needs the data caches: run 'make replay-demo' first"; exit 1; fi
	$(call step,replay,mkdir -p $$(dirname $(REPLAY_OUT)) && \
	  WATTSENTRY_CACHE=$(CACHE) $(PYTHON) replay.py --policy baseline$(comma)sentinel \
	    --world $(REPLAY_WORLD) --days $(REPLAY_DAYS) --seed $(SEED) --out $(REPLAY_OUT))

# Builds the data windows, wind parks and grid twin into $(CACHE), then replays the demo week with models/.
replay-demo: setup
	$(call guard,$(DEMO_OUT))
	$(call step,replay-demo,export WATTSENTRY_CACHE=$(CACHE) && \
	  $(PYTHON) -m wattsentry.data.windows && \
	  $(PYTHON) -m wattsentry.sim.wind --no-plots && \
	  $(PYTHON) -m wattsentry.gridtwin.run && \
	  mkdir -p $$(dirname $(DEMO_OUT)) && \
	  $(PYTHON) replay.py --world sim --policy baseline$(comma)sentinel --out $(DEMO_OUT))

train: setup
	@if command -v uv >/dev/null 2>&1; then uv pip install -q --python $(PYTHON) -r requirements-train.txt; \
	  else $(PYTHON) -m pip install -q -r requirements-train.txt; fi
	$(call step,train-$(MODEL),mkdir -p $(OUT)/train/$(MODEL) && \
	  LAB_OUTPUT_DIR=$(OUT)/train/$(MODEL) $(PYTHON) train.py --model $(MODEL) $(TRAIN_ARGS))

app:
	@[ -d node_modules ] || npm ci --no-audit --no-fund
	@if [ -z "$(APP_DB)" ] && [ ! -f demo/wattsentry.sqlite ] && [ ! -f demo/fixtures.sqlite ]; then \
	  python3 demo/fixtures/make_fixtures.py; fi
	npm run build
	$(if $(APP_DB),DATABASE_URL=file:$(abspath $(APP_DB)),) npm start

demo/wattsentry-homes600.sqlite: demo/wattsentry-homes600.sqlite.gz
	python3 -c 'import gzip,shutil; shutil.copyfileobj(gzip.open("$<"), open("$@","wb"))'

app-600: demo/wattsentry-homes600.sqlite
	$(MAKE) app APP_DB=demo/wattsentry-homes600.sqlite

fixtures:
	python3 demo/fixtures/make_fixtures.py

# Every metric printed by the last reproduce, per step.
summary:
	@[ -d $(LOGS) ] || { echo "no logs in $(LOGS): run make reproduce"; exit 1; }
	@for s in $(STEPS); do \
	  f=$(LOGS)/$$s.log; [ -f $$f ] || continue; \
	  printf '\n[%s] %ss  (%s)\n' "$$s" "$$(cat $(LOGS)/$$s.time 2>/dev/null || echo ?)" "$$f"; \
	  case $$s in \
	    test) grep -E '[0-9]+ passed' $$f | tail -1 | sed 's/^/    /' || true;; \
	    sim)  grep -E '^wrote ' $$f | sed 's/^/    /' || true;; \
	    *) if grep -q '^== ' $$f; then sed -n '/^== /,/^eval report:/p' $$f | sed 's/^/    /'; \
	       else grep -E '^LAB_METRIC ' $$f | sed 's/^LAB_METRIC /    /' || true; fi;; \
	  esac; \
	done
	@[ -f $(LOGS)/total.time ] && echo && echo "total: $$(cat $(LOGS)/total.time)s" || true

clean:
	rm -rf $(OUT)

clean-venv: clean
	rm -rf $(VENV)
