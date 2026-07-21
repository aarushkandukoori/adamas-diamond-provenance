# ADAMAS: Diamond Provenance via Physical Cryptographic Binding

[![reproduce](https://github.com/aarushkandukoori/adamas-diamond-provenance/actions/workflows/reproduce.yml/badge.svg)](https://github.com/aarushkandukoori/adamas-diamond-provenance/actions/workflows/reproduce.yml)

Distributed ledgers give diamond provenance systems tamper-evident records, but they do not bind those records to the physical stones they describe. ADAMAS closes that binding gap by deriving a cryptographic key from a stone's crystallographic defect structure—an alignment-free inclusion embedding for screening, plus a registration-and-vault stage over the inclusion point set—so that possession of the physical stone is a prerequisite for every provenance operation. This repository ships the paper sources, a pure NumPy/Matplotlib evaluation harness, committed numerical results, and a bit-exact verification workflow.

## Findings

| Quantity | Value |
|---|---|
| `eer_learned` | 0.01881666666666667 |
| `eer_pca` | 0.08875 |
| `stageA_best_key_bits` | −124.28434129972646 |
| `pl_gate_acc` | 0.9978750000000001 |
| `lab_admitted` / `lab_n` | 200 / 200 |
| `G_for_128` | 22 |
| `vault_choice` | C=10000, D=8, sec=90.16602947289002, unlock=0.9699112511799476 |

## Quickstart

```bash
git clone https://github.com/aarushkandukoori/adamas-diamond-provenance.git
cd adamas-diamond-provenance
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all
```

`make all` runs the simulation (`make sim`), builds the paper with `pdflatex` ×3 (`make paper`), and checks headline numbers (`make verify`). Simulation runtime is ~65 seconds single-threaded.

## Reproducibility

The harness is pure NumPy + Matplotlib (no SciPy, no scikit-learn). A single RNG seed is fixed at process start:

```python
RNG = np.random.default_rng(20260721)
```

Dependencies are pinned in `requirements.txt` to `numpy==2.4.4` and `matplotlib==3.10.8` on **Python 3.12**. LAPACK version differences across BLAS backends can shift eigendecomposition sign conventions in the PCA/LDA stage, so exact reproduction assumes the pinned stack. CI re-runs `scripts/verify.py` on every push and pull request and asserts the headline values above within absolute tolerance `1e-9`.

## Repo layout

```
README.md
LICENSE                 # MIT
CITATION.cff
.gitignore
requirements.txt
Makefile
paper/
  adamas.tex            # embedded thebibliography (no .bib)
  adamas.pdf
src/
  adamas_simulation.py  # writes results/ and results/figures/
results/
  results.json
  figures/*.pdf
scripts/
  verify.py
.github/workflows/
  reproduce.yml
```

## Status of the evaluation

All numbers in this repository come from a generative model over synthetic stones and are predictions about a measurement campaign, not measurements of physical diamonds.
