# Rung surrogates

Code for *Rank-Preserving Synthetic Returns for Linear Signal Strategies: A Ladder of Invariances
for Robustness Testing* (N. Sambelashvili).

A **rung** is a class of trading strategies plus the set of synthetic data sets on which every
strategy in the class ranks exactly as it does on history. `synthgen.py` draws synthetic return
paths on a chosen rung, while also pinning the mean and covariance matrix and restoring fat tails
and volatility clustering.

## Quick start

```python
import numpy as np
from synthgen import RungGenerator

R = ...                                   # (T, N) array of daily returns
gen = RungGenerator(R, "diagonal",        # full | symmetric | factor | diagonal | pooled
                    tail="rank",          # rank | student_t | radial | none
                    vol_beta="auto",      # volatility-clustering strength, calibrated
                    seed=1)
paths, info = gen.generate(20)            # list of (T, N) synthetic paths
print(gen.pinned_error(paths[0]))         # ~1e-15: rung statistics are exact
```

Mixed rungs across frequency bands:

```python
spec = [{"rung": "full",   "band": (0.00, 0.02)},
        {"rung": "factor", "band": (0.02, 0.20), "K": 5},
        {"rung": "pooled", "band": (0.20, 0.50)}]
gen = RungGenerator(R, spec)
```

| Rung | Strategies whose rankings are preserved | Pinned |
|---|---|---|
| full | all linear signal strategies | every lagged cross-moment |
| symmetric | reciprocal strategies (XS momentum/reversal, trend, factor momentum) | symmetric part |
| factor | strategies inside K factor portfolios | factor lagged cross-moments |
| diagonal | each asset traded on its own past | each asset's autocovariances |
| pooled | the same own-past rule for every asset | average autocovariance |

Every rung also pins the mean vector and covariance matrix, so static Sharpe rankings are kept too.
Exactness holds for circular backtests (the sample treated as a loop); see the paper, Section 8.

## Files

| File | Purpose |
|---|---|
| `synthgen.py` | rung generator, diagnostics |
| `baselines.py` | block bootstrap, VAR(1), GARCH-FHS, phase randomization, multivariate IAAFT |
| `experiments.py` | rank-preservation and realism scores for every generator |
| `edge_profile.py` | edge-profile experiment (momentum and reversal strategies) |
| `run_dataset.py` | Sigma-pinned paths used in the path figures |
| `toy_example.py` | two-asset toy market used to explain the rungs |
| `make_*.py` | figures and LaTeX tables, generated directly from experiment outputs |
| `reproduce.sh` | runs everything in order |
| `paper/` | LaTeX source of the paper |

## Reproducing the paper

```bash
pip install -r requirements.txt
# put prices.csv in data/ (see data/README.md)
./reproduce.sh
```
