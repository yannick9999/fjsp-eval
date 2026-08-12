"""Shared configuration for the model comparison analysis and plotting scripts.

Data layout (relative to this script):
    {METHOD_DIR}/test/seed{0,1,2}/{folder}_{mode}/*.xlsx   DRL results
        sheets "makespan" and "solve_time"
        DAN: fixed filename "results.xlsx"
        others: timestamped "test_results_*.xlsx" (most recent one is used)
    benchmarks/{BASELINE}/{folder}.csv                      baseline results

Instance names in the xlsx/csv files may carry ".fjs" ("abz5.fjs") or not
("abz5"); the loaders in analyze.py normalize this by stripping it.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR / "benchmarks"
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Cached bootstrap/analysis results, written by analyze.py and read by plot.py
ANALYSIS_CACHE = SCRIPT_DIR / "analysis_cache.pkl"

# DRL methods to compare. Lei has its result folder created but not filled in
# yet; add it here once test data has landed (analyze.py handles missing
# files gracefully, but including a method with zero data just floods the
# run with "missing" warnings).
METHODS = ["dan", "song", "sagc", "edsp"]
METHOD_LABELS = {"dan": "DAN", "song": "Song", "sagc": "SAGC", "edsp": "EDSP", "lei": "Lei"}
METHOD_COLORS = {
    "dan": "#4C72B0",     # blue
    "song": "#DD8452",    # orange
    "sagc": "#55A868",    # green
    "edsp": "#C44E52",    # red, reserved
    "lei": "#8172B2",     # purple, reserved
}

# Top-level folder holding each method's data, relative to this script.
# Layout under it: {METHOD_DIR}/test/seed{s}/{folder}_{mode}/*.xlsx
METHOD_DIRS = {"dan": "DAN", "song": "Song", "sagc": "SAGC", "edsp": "EDSP", "lei": "Lei"}

# Seeds. Per-method override below for methods with fewer seeds available.
SEEDS = [0, 1, 2]
METHOD_SEEDS = {"edsp": [0]}

# Inference modes. Same method color, distinguished by hatching in bar charts.
# Per-method override below for methods without sampling data yet.
MODES = ["greedy", "sample"]
MODE_LABELS = {"greedy": "Greedy", "sample": "Sampling"}
MODE_HATCHES = {"greedy": "", "sample": "///"}
METHOD_MODES = {"edsp": ["greedy"]}


def method_seeds(method: str) -> list[int]:
    return METHOD_SEEDS.get(method, SEEDS)


def method_modes(method: str) -> list[str]:
    return METHOD_MODES.get(method, MODES)

# Synthetic instance groups. 20x10 appears in both, jobs x machines.
SYNTHETIC_MACHINES = ["20x5", "20x10", "20x20", "20x30"]   # jobs fixed at 20
SYNTHETIC_JOBS = ["10x10", "20x10", "30x10", "40x10",
                  "50x10", "100x10", "200x10"]             # machines fixed at 10

# Numeric x-axis values for the efficiency (solve-time) line plots, aligned
# index-for-index with SYNTHETIC_MACHINES / SYNTHETIC_JOBS above.
SYNTHETIC_MACHINES_X = [5, 10, 20, 30]
SYNTHETIC_JOBS_X = [10, 20, 30, 40, 50, 100, 200]

# Hurink datasets
HURINK_DATASETS = ["edata", "rdata", "vdata"]
HURINK_LABELS = {"edata": "Edata", "rdata": "Rdata", "vdata": "Vdata"}

# Brandimarte, plotted per instance
BRANDIMARTE = "Mk"
BRANDIMARTE_INSTANCES = [f"Mk{i:02d}" for i in range(1, 11)]

# Display label -> folder name (results) resp. CSV base name (benchmarks).
# Brandimarte is the only case where the two differ, handled via
# RESULT_FOLDER_MAP and BENCH_FILE_MAP below.
SIZE_FOLDER_MAP = {
    "10x10": "1010",
    "20x5": "2005",
    "20x10": "2010",
    "20x20": "2020",
    "20x30": "2030",
    "30x10": "3010",
    "40x10": "4010",
    "50x10": "5010",
    "100x10": "10010",
    "200x10": "20010",
}
SIZE_FOLDER_MAP.update({d: d for d in HURINK_DATASETS})

RESULT_FOLDER_MAP = dict(SIZE_FOLDER_MAP)
RESULT_FOLDER_MAP[BRANDIMARTE] = "Mk"

BENCH_FILE_MAP = dict(SIZE_FOLDER_MAP)
BENCH_FILE_MAP[BRANDIMARTE] = "brandimarte"

# All non-Brandimarte sizes, each bootstrapped exactly once (20x10 shared
# between the two synthetic groups).
ALL_SIZES = list(dict.fromkeys(SYNTHETIC_MACHINES + SYNTHETIC_JOBS)) + HURINK_DATASETS

# Baselines (dispatching rules + CP-SAT)
BASELINES = ["MWR", "SPT", "MOR", "FIFO", "CPSAT"]
# The pure dispatching rules, without CP-SAT, used for the "best dispatching
# rule" pseudo-baseline (best rule per instance, then aggregated).
DISPATCHING_RULES = ["MWR", "SPT", "MOR", "FIFO"]
BASELINE_COLORS = {
    "CPSAT": "#9A7090",
    "BestDR": "#2E2E2E",
}
BASELINE_LABELS = {"CPSAT": "CP-SAT", "BestDR": "Best DR"}

# Baseline bars shown in the IQM charts, in this order (before the methods)
IQM_BASELINE_KEYS = ["CPSAT", "BestDR"]

# The efficiency (solve-time) plots compare DRL methods only, no baselines:
# CP-SAT hits its 3600s time limit (status FEASIBLE, not OPTIMAL) on almost
# every size above 10x10, so its "solve time" is just the cutoff, not a
# converged runtime, and would dominate a shared axis with DRL inference
# times (fractions of a second). Dispatching rules are near-instant and
# equally uninformative on the same scale.

# rliable bootstrap replications, at least 2000 for final reporting
BOOTSTRAP_REPS = 100


def combo_key(method: str, mode: str) -> str:
    """Composite key identifying a (method, mode) series, e.g. 'dan__greedy'."""
    return f"{method}__{mode}"


def split_combo_key(key: str) -> tuple[str, str]:
    """Inverse of combo_key."""
    method, mode = key.split("__", 1)
    return method, mode