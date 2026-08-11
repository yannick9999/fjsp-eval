"""Loads the model comparison result data, runs the rliable bootstrap
analysis for the IQM and solve-time bar/line charts, and writes a cache for
plot.py.

Run this whenever the underlying data changes. Run plot.py (no
recomputation) whenever only the plot styling should change.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rliable import library as rly
from rliable import metrics

from common import (
    ALL_SIZES,
    ANALYSIS_CACHE,
    BASELINES,
    BENCH_FILE_MAP,
    BENCHMARKS_DIR,
    BOOTSTRAP_REPS,
    BRANDIMARTE,
    BRANDIMARTE_INSTANCES,
    DISPATCHING_RULES,
    METHOD_DIRS,
    METHODS,
    MODES,
    RESULT_FOLDER_MAP,
    SCRIPT_DIR,
    SEEDS,
    combo_key,
)


# Data loading

def _strip_ext(names: pd.Series) -> pd.Series:
    return names.astype(str).str.replace(r"\.fjs$", "", regex=True)


def _drl_result_file(method: str, size: str, seed: int, mode: str) -> Path | None:
    """Resolves the result xlsx for a method/size/seed/mode, or None if missing.

    DAN writes a fixed "results.xlsx"; the other methods write a timestamped
    "test_results_*.xlsx" per run, so the most recent match is used.
    """
    folder = RESULT_FOLDER_MAP[size]
    result_dir = SCRIPT_DIR / METHOD_DIRS[method] / "test" / f"seed{seed}" / f"{folder}_{mode}"
    if method == "dan":
        excel = result_dir / "results.xlsx"
        return excel if excel.exists() else None
    matches = sorted(result_dir.glob("test_results_*.xlsx"))
    return matches[-1] if matches else None


def load_drl_test_results(method: str, size: str, seed: int, mode: str
                          ) -> tuple[dict[str, float], dict[str, float]] | None:
    """Loads test makespans and solve times per instance for a method, size, seed, mode.

    Both live as separate sheets in the same results file, so it's read once.

    Returns:
        (makespans, solve_times), each {instance_name: value}, or None if the
        file is missing.
    """
    excel = _drl_result_file(method, size, seed, mode)
    if excel is None:
        return None
    makespan_df = pd.read_excel(excel, sheet_name="makespan")
    time_df = pd.read_excel(excel, sheet_name="solve_time")
    makespans = dict(zip(_strip_ext(makespan_df.iloc[:, 0]), makespan_df.iloc[:, 1].astype(float)))
    solve_times = dict(zip(_strip_ext(time_df.iloc[:, 0]), time_df.iloc[:, 1].astype(float)))
    return makespans, solve_times


def load_benchmark_makespans(rule: str, size: str) -> dict[str, float] | None:
    """Loads baseline makespans from CSV, stripping the .fjs extension so the
    instance names match the xlsx result files.

    Returns:
        Dict {instance_name: makespan} or None if the file is missing.
    """
    csv = BENCHMARKS_DIR / rule / f"{BENCH_FILE_MAP[size]}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    return dict(zip(_strip_ext(df["instance_name"]), df["makespan"].astype(float)))


# Score matrices

def get_baseline_makespans(size: str) -> dict[str, dict[str, float]]:
    """Collects all available baseline makespans for a size.

    Returns:
        Dict {baseline_name: {instance_name: makespan}}
    """
    result = {}
    for b in BASELINES:
        m = load_benchmark_makespans(b, size)
        if m is not None:
            result[b] = m
        else:
            print(f"  [warn] Baseline {b} missing for {size}")
    return result


def compute_c_best(baseline_data: dict[str, dict[str, float]]) -> dict[str, float]:
    """Computes C_best per instance as the minimum over all baselines."""
    if not baseline_data:
        return {}
    all_instances = set()
    for b_data in baseline_data.values():
        all_instances.update(b_data.keys())
    c_best = {}
    for inst in all_instances:
        values = [b_data[inst] for b_data in baseline_data.values() if inst in b_data]
        if values:
            c_best[inst] = min(values)
    return c_best


def compute_c_best_dr(baseline_data: dict[str, dict[str, float]]) -> dict[str, float]:
    """Computes the best-dispatching-rule makespan per instance (CP-SAT excluded)."""
    dr_data = {b: d for b, d in baseline_data.items() if b in DISPATCHING_RULES}
    return compute_c_best(dr_data)


def build_score_matrix(method: str, size: str, c_best: dict[str, float],
                       mode: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Builds the normalized score matrix and the solve-time matrix for a
    method, size, mode.

    Score = C_best / C_method (higher = better). Solve time is in seconds,
    unnormalized.

    Returns:
        (score matrix, time matrix, list of instance_names), all shape
        (num_seeds, num_instances) resp. matching the matrix columns.
    """
    per_seed_makespans = []
    per_seed_times = []
    for s in SEEDS:
        loaded = load_drl_test_results(method, size, s, mode)
        if loaded is None:
            print(f"  [warn] Test data missing: {method} seed{s} {size} {mode}")
            return np.array([]), np.array([]), []
        makespans, times = loaded
        per_seed_makespans.append(makespans)
        per_seed_times.append(times)

    # Common instances present in all seeds AND in c_best
    common = set(per_seed_makespans[0].keys())
    for d in per_seed_makespans[1:]:
        common &= set(d.keys())
    if c_best:
        common &= set(c_best.keys())
    instances = sorted(common)

    if not instances:
        return np.array([]), np.array([]), []

    score_matrix = np.zeros((len(SEEDS), len(instances)))
    time_matrix = np.zeros((len(SEEDS), len(instances)))
    for i, s in enumerate(SEEDS):
        for j, inst in enumerate(instances):
            score_matrix[i, j] = c_best[inst] / per_seed_makespans[i][inst]
            time_matrix[i, j] = per_seed_times[i][inst]
    return score_matrix, time_matrix, instances


def build_baseline_score(baseline_makespans: dict[str, float], c_best: dict[str, float],
                         instances: list[str]) -> np.ndarray:
    """Score array for a deterministic baseline (shape (1, num_instances))."""
    scores = np.array([c_best[i] / baseline_makespans[i] for i in instances if i in baseline_makespans])
    return scores.reshape(1, -1)


def build_scores_for_size(size: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Builds all (method, mode) score matrices, solve-time matrices, plus
    baseline score arrays for one size/dataset.

    Returns:
        (score_dict, time_dict, baseline_scores). All empty if data is missing.
    """
    baseline_data = get_baseline_makespans(size)
    c_best = compute_c_best(baseline_data)
    if not c_best:
        print(f"  [warn] No C_best for {size}, skipping")
        return {}, {}, {}

    score_dict = {}
    time_dict = {}
    instances_ref = None
    for method in METHODS:
        for mode in MODES:
            score_matrix, time_matrix, instances = build_score_matrix(method, size, c_best, mode)
            if score_matrix.size == 0:
                continue
            key = combo_key(method, mode)
            score_dict[key] = score_matrix
            time_dict[key] = time_matrix
            if instances_ref is None:
                instances_ref = instances
            iqm = metrics.aggregate_iqm(score_matrix)
            print(f"    {key}: {score_matrix.shape}, IQM={iqm:.4f}, mean solve_time={time_matrix.mean():.3f}s")

    baseline_scores = {}
    if instances_ref is not None:
        for b, b_data in baseline_data.items():
            arr = build_baseline_score(b_data, c_best, instances_ref)
            if arr.size:
                baseline_scores[b] = arr
        best_dr = compute_c_best_dr(baseline_data)
        arr = build_baseline_score(best_dr, c_best, instances_ref)
        if arr.size:
            baseline_scores["BestDR"] = arr

    return score_dict, time_dict, baseline_scores


# Bootstrap analysis

def _bootstrap_means(data_dict: dict[str, np.ndarray], aggregate_fn
                     ) -> tuple[list[str], dict[str, float], dict[str, tuple[float, float]]]:
    """Runs the rliable bootstrap for one size and unpacks the result.

    Returns:
        (method_keys, {key: point estimate}, {key: (ci_low, ci_high)})
    """
    point_estimates, cis = rly.get_interval_estimates(data_dict, aggregate_fn, reps=BOOTSTRAP_REPS)
    keys = list(data_dict.keys())
    means = {k: float(point_estimates[k][0]) for k in keys}
    cis_out = {k: (float(cis[k][0, 0]), float(cis[k][1, 0])) for k in keys}
    return keys, means, cis_out


def analyze_iqm_bars(score_dict_per_size: dict[str, dict[str, np.ndarray]],
                     baseline_scores_per_size: dict[str, dict[str, np.ndarray]],
                     sizes: list[str]) -> dict:
    """Bootstraps IQM + 95% CI per (method, mode), per size; plus baseline IQM points."""
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])
    result = {}
    n = len(sizes)

    for si, size in enumerate(sizes):
        score_dict = score_dict_per_size.get(size, {})
        if not score_dict:
            result[size] = None
            continue

        print(f"    bootstrap {size} ({si+1}/{n}) ...", end=" ", flush=True)
        t0 = time.time()
        methods, means, cis = _bootstrap_means(score_dict, iqm_fn)
        print(f"{time.time()-t0:.1f}s")

        baseline_scores = baseline_scores_per_size.get(size, {})
        baseline_iqm = {}
        for b, arr in baseline_scores.items():
            if arr.size:
                baseline_iqm[b] = float(metrics.aggregate_iqm(arr))

        result[size] = {"methods": methods, "means": means, "cis": cis, "baseline_iqm": baseline_iqm}

    return result


def analyze_solve_time_bars(time_dict_per_size: dict[str, dict[str, np.ndarray]],
                            sizes: list[str]) -> dict:
    """Bootstraps mean solve time (seconds) + 95% CI per (method, mode), per size.

    No baselines here, see the comment on IQM_BASELINE_KEYS in common.py for
    why CP-SAT isn't a fair comparison on this axis.
    """
    mean_fn = lambda x: np.array([np.mean(x)])
    result = {}
    n = len(sizes)

    for si, size in enumerate(sizes):
        time_dict = time_dict_per_size.get(size, {})
        if not time_dict:
            result[size] = None
            continue

        print(f"    bootstrap {size} ({si+1}/{n}) ...", end=" ", flush=True)
        t0 = time.time()
        methods, means, cis = _bootstrap_means(time_dict, mean_fn)
        print(f"{time.time()-t0:.1f}s")

        result[size] = {"methods": methods, "means": means, "cis": cis}

    return result


def build_brandimarte_per_instance() -> tuple[dict[str, dict[str, np.ndarray]],
                                              dict[str, dict[str, np.ndarray]]]:
    """Builds per-instance score dicts for Brandimarte, so that the IQM chart
    can show one bar group per Mk instance.

    Each "size" key is one instance (Mk01 ... Mk10). The method matrices have
    shape (num_seeds, 1), the bootstrap CI then reflects seed variation only.

    Returns:
        (score_dict_per_instance, baseline_scores_per_instance)
    """
    baseline_data = get_baseline_makespans(BRANDIMARTE)
    c_best = compute_c_best(baseline_data)
    best_dr = compute_c_best_dr(baseline_data)

    # Load all method data once (makespans only, Brandimarte has no efficiency plot)
    method_data = {}
    for method in METHODS:
        for mode in MODES:
            per_seed = []
            for s in SEEDS:
                loaded = load_drl_test_results(method, BRANDIMARTE, s, mode)
                if loaded is None:
                    print(f"  [warn] Test data missing: {method} seed{s} {BRANDIMARTE} {mode}")
                    per_seed = None
                    break
                makespans, _times = loaded
                per_seed.append(makespans)
            if per_seed is not None:
                method_data[combo_key(method, mode)] = per_seed

    score_dict_per_instance = {}
    baseline_scores_per_instance = {}
    for inst in BRANDIMARTE_INSTANCES:
        if inst not in c_best:
            print(f"  [warn] No baseline data for {inst}, skipping")
            continue

        score_dict = {}
        for key, per_seed in method_data.items():
            if all(inst in d for d in per_seed):
                col = np.array([[c_best[inst] / d[inst]] for d in per_seed])
                score_dict[key] = col
        score_dict_per_instance[inst] = score_dict

        baseline_scores = {}
        for b, b_data in baseline_data.items():
            if inst in b_data:
                baseline_scores[b] = np.array([[c_best[inst] / b_data[inst]]])
        if inst in best_dr:
            baseline_scores["BestDR"] = np.array([[c_best[inst] / best_dr[inst]]])
        baseline_scores_per_instance[inst] = baseline_scores

    return score_dict_per_instance, baseline_scores_per_instance


# Main

def main():
    print("=" * 70)
    print("Model Comparison Analysis")
    print("=" * 70)
    print(f"Script dir:  {SCRIPT_DIR}")
    print(f"Benchmarks:  {BENCHMARKS_DIR}")
    print(f"Cache out:   {ANALYSIS_CACHE}")
    print(f"Methods:     {METHODS}")
    print(f"Modes:       {MODES}")
    print(f"Seeds:       {SEEDS}")
    print(f"Sizes:       {ALL_SIZES}")
    print(f"Brandimarte: {BRANDIMARTE_INSTANCES}")
    print(f"Baselines:   {BASELINES}")
    print()

    t_start = time.time()

    # 1. Score + solve-time matrices for all sizes/datasets (20x10 shared
    # between the two synthetic groups, so it's loaded and bootstrapped
    # exactly once)
    print("[1/4] Loading data and building score/time matrices ...")
    score_dict_per_size = {}
    time_dict_per_size = {}
    baseline_scores_per_size = {}
    for size in ALL_SIZES:
        print(f"  Size {size}")
        score_dict, time_dict, baseline_scores = build_scores_for_size(size)
        if score_dict:
            score_dict_per_size[size] = score_dict
            time_dict_per_size[size] = time_dict
            baseline_scores_per_size[size] = baseline_scores

    # 2. Bootstrap IQM per size
    print("[2/4] IQM bootstrap per size ...")
    iqm = analyze_iqm_bars(score_dict_per_size, baseline_scores_per_size, ALL_SIZES)

    # 3. Bootstrap mean solve time per size (synthetic sizes only get plotted,
    # but bootstrapping is cheap so this covers all of ALL_SIZES uniformly)
    print("[3/4] Solve-time bootstrap per size ...")
    solve_time = analyze_solve_time_bars(time_dict_per_size, ALL_SIZES)

    # 4. Brandimarte per instance
    print("[4/4] IQM bootstrap per Brandimarte instance ...")
    mk_scores, mk_baselines = build_brandimarte_per_instance()
    iqm_brandimarte = analyze_iqm_bars(mk_scores, mk_baselines, BRANDIMARTE_INSTANCES)

    cache = {"iqm": iqm, "iqm_brandimarte": iqm_brandimarte, "solve_time": solve_time}
    with open(ANALYSIS_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nSaved analysis cache to {ANALYSIS_CACHE}")

    total_elapsed = time.time() - t_start
    print(f"All done in {total_elapsed:.1f}s. Run plot.py to (re)generate plots.")


if __name__ == "__main__":
    main()