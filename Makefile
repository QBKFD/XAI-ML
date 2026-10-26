PYTHON ?= python

.PHONY: install test test-fast lint benchmark tune figures all

install:            ## package, dev tools and the three legacy rule learners
	$(PYTHON) -m pip install -e ".[dev]"
	PYTHON=$(PYTHON) bash scripts/install_legacy_deps.sh

test:               ## full test suite (~15 s)
	$(PYTHON) -m pytest

test-fast:          ## skip the tests that fit MDL and IDS
	$(PYTHON) -m pytest -m "not slow"

lint:
	ruff check src scripts tests
	ruff format --check src scripts tests

benchmark:          ## 11 datasets x 7 models x 5 splits (~30 min, mostly IDS)
	$(PYTHON) scripts/run_benchmark.py

tune:               ## RuleOpt grid search and IDS lambda search (IDS takes hours)
	$(PYTHON) scripts/tune.py ruleopt
	$(PYTHON) scripts/tune.py ids

figures:
	$(PYTHON) scripts/make_figures.py

all: benchmark figures
