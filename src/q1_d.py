# -*- coding: utf-8 -*-

# Question 1d
import pandas as pd
from pathlib import Path

# set the path
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_q2"
FIG_DIR.mkdir(exist_ok=True)

data = pd.read_csv(ROOT / "data" / "iowa_yield_05_10.csv")
