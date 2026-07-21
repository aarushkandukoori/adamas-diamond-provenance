PYTHON ?= python3.12
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
export PATH := $(HOME)/Library/TinyTeX/bin/universal-darwin:/Library/TeX/texbin:$(PATH)

.PHONY: all sim paper verify clean

all: sim paper verify

sim:
	cd $(REPO_ROOT) && $(PYTHON) src/adamas_simulation.py

paper:
	cd $(REPO_ROOT)/paper && pdflatex -interaction=nonstopmode adamas.tex
	cd $(REPO_ROOT)/paper && pdflatex -interaction=nonstopmode adamas.tex
	cd $(REPO_ROOT)/paper && pdflatex -interaction=nonstopmode adamas.tex

verify:
	cd $(REPO_ROOT) && $(PYTHON) scripts/verify.py

clean:
	rm -f $(REPO_ROOT)/paper/*.aux $(REPO_ROOT)/paper/*.log $(REPO_ROOT)/paper/*.out \
		$(REPO_ROOT)/paper/*.toc $(REPO_ROOT)/paper/*.synctex.gz
	find $(REPO_ROOT) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
