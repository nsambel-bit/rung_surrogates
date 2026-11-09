#!/usr/bin/env bash
# Reproduces every table and figure of the paper from data/prices.csv.
# Runtimes are for a 2-core machine.
set -euo pipefail
python run_dataset.py            # Sigma-pinned paths for the path figures      (~30 min)
python experiments.py etf 20     # rungs + baselines, 9 sector ETFs               (~4 min)
python experiments.py all 20     # rungs + baselines, 97 instruments              (~50 min)
python edge_profile.py etf 50    # edge profile, ETFs                             (~5 min)
python edge_profile.py all 20    # edge profile, 97 instruments                   (~50 min)
python toy_example.py            # toy two-asset table
python make_tables.py            # all LaTeX tables -> paper/tables
PAPER=1 python make_figs.py
PAPER=1 python make_fig_baselines.py
PAPER=1 python make_fig_edge.py all
PAPER=1 python make_fig_edge.py etf    # figures -> paper/figs
cd paper && latexmk -pdf main.tex
