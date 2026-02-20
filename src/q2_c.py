# -*- coding: utf-8 -*-

# Question 2c
import pandas as pd
from pathlib import Path

from q2_a import function_q2_a

output_q2_a = function_q2_a()

# set the path
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures_q2"
FIG_DIR.mkdir(exist_ok=True)

data = pd.read_excel(ROOT / "data" / "cds_spatialweights.xlsx")

# Project paths
ROOT = Path(__file__).resolve().parents[1]

