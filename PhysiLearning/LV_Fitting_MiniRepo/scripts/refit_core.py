#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import matplotlib as mpl
    try:
        mpl.use("TkAgg")
    except Exception:
        mpl.use("Agg")
except Exception:
    mpl = None

from physilearning.evaluate import Evaluation
from physilearning.train import Trainer

CONFIG_FILE = REPO_ROOT / "config.yaml"
DATA_DIR = REPO_ROOT / "data" / "experimental_data_elias" / "2025_stem_paper_final_final"
REFERENCE_DIR = REPO_ROOT / "reference"
ORIGINAL_SAVED_PATH = REFERENCE_DIR / "original_saved.npy"
PARAMETER_ORDER = [
    "growth_rate_sus",
    "growth_rate_res",
    "competition",
    "random_death",
    "death_treat_sus",
    "ramp_time_up",
    "ramp_time_down",
    "carrying_capacity",
    "on_treat_threshold",
    "off_treat_threshold",
    "min_treat",
]
ORIGINAL_SEED = [0.07, 0.14, 2.0, 0.01, 0.06, 2.0, 1.0, 4.0, 6.5, 1.5, 0.01]
BOUNDS = [
    (0.01, 5.0),
    (0.01, 5.0),
    (0.01, 500.0),
    (0.00001, 5.0),
    (0.001, 5.0),
    (0.1, 10000.0),
    (0.1, 10000.0),
    (1.0, 1000.0),
    (0.1, 10000.0),
    (0.1, 10000.0),
    (0.001, 5.0),
]
EXPERIMENTS = [
    "Plate 1/Results/Results_G418_a2a4_B4_AT100-B4.txt",
    "Plate 1/Results/Results_G418_a2a4_B5_AT100-B5.txt",
    "Plate 1/Results/Results_G418_a2a4_C4_Pulse-C4.txt",
    "Plate 1/Results/Results_G418_a2a4_C6_Pulse-C6.txt",
    "Plate 1/Results/Results_G418_a2a4_D4_NT-D4.txt",
    "Plate 2/Results/Results_G418_a2a4_A1_CT-A1.txt",
    "Plate 2/Results/Results_G418_a2a4_A6_CT-A6.txt",
    "Plate 2/Results/Results_G418_a2a4_C1_AT100-C1.txt",
    "Plate 2/Results/Results_G418_a2a4_C2_AT100-C2.txt",
    "Plate 2/Results/Results_G418_a2a4_D2_Pulse-D2.txt",
    "Plate 2/Results/Results_G418_a2a4_D5_Pulse-D5.txt",
    "Plate 3/Results/Results_G418_a2a4_B2_CT-B2.txt",
    "Plate 3/Results/Results_G418_a2a4_B3_CT-B3.txt",
    "Plate 3/Results/Results_G418_a2a4_D2_AT100-D2.txt",
    "Plate 3/Results/Results_G418_a2a4_D4_AT100-D4.txt",
    "Plate 4/Results/Results_G418_a2a4_A4_Pulse-A4.txt",
    "Plate 4/Results/Results_G418_a2a4_A5_Pulse-A5.txt",
    "Plate 4/Results/Results_G418_a2a4_B1_NT-B1.txt",
    "Plate 4/Results/Results_G418_a2a4_B6_NT-B6.txt",
    "Plate 4/Results/Results_G418_a2a4_C2_CT-C2.txt",
    "Plate 4/Results/Results_G418_a2a4_C5_CT-C5.txt",
    "Plate 5/Results/Results_G418_a2a4_A2_AT100-A2.txt",
    "Plate 5/Results/Results_G418_a2a4_A5_AT100-A5.txt",
    "Plate 5/Results/Results_G418_a2a4_B3_Pulse-B3.txt",
    "Plate 5/Results/Results_G418_a2a4_B5_Pulse-B5.txt",
    "Plate 5/Results/Results_G418_a2a4_C1_NT-C1.txt",
    "Plate 5/Results/Results_G418_a2a4_C3_NT-C3.txt",
    "Plate 5/Results/Results_G418_a2a4_C4_NT-C4.txt",
    "Plate 5/Results/Results_G418_a2a4_C5_NT-C5.txt",
    "Plate 5/Results/Results_G418_a2a4_C6_NT-C6.txt",
    "Plate 5/Results/Results_G418_a2a4_D2_CT-D2.txt",
    "Plate 5/Results/Results_G418_a2a4_D5_CT-D5.txt",
]


@dataclass
class ExperimentRecord:
    rel_path: str
    treatment_type: str
    sus_exp: np.ndarray
    res_exp: np.ndarray


def load_original_saved() -> list[float]:
    return [float(v) for v in np.load(ORIGINAL_SAVED_PATH).tolist()]


def infer_treatment_type(name: str) -> str:
    if "NT" in name:
        return "NT"
    if "CT" in name:
        return "CT"
    if "Pulse" in name:
        return "Pulse"
    if "AT100" in name:
        return "AT100"
    raise ValueError(f"Unknown treatment type for {name}")


def get_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path, sep=r"\s+", names=["name", "sus", "res"], engine="python")
    data["day"] = data["name"].str.extract(r"(\d{2})$").astype(int)
    ini_tot = data["sus"].iloc[0] + data["res"].iloc[0]
    data["sus"] /= ini_tot
    data["res"] /= ini_tot
    return data["sus"].values[:9].astype(float), data["res"].values[:9].astype(float)


def load_experiments() -> list[ExperimentRecord]:
    rows = []
    for rel_path in EXPERIMENTS:
        path = DATA_DIR / rel_path
        sus_exp, res_exp = get_data(path)
        rows.append(
            ExperimentRecord(
                rel_path=rel_path,
                treatment_type=infer_treatment_type(path.name),
                sus_exp=sus_exp,
                res_exp=res_exp,
            )
        )
    return rows


def run_simulation(params: list[float], initial_condition: list[float], exp_data: list[np.ndarray], iterations: int, treatment_type: str) -> tuple[list[float], list[float]]:
    (
        growth_rate_sus,
        growth_rate_res,
        competition,
        random_death,
        death_treat_sus,
        ramp_time_up,
        ramp_time_down,
        carrying_capacity,
        on_treat_threshold,
        off_treat_threshold,
        min_treat,
    ) = params

    (REPO_ROOT / "Training").mkdir(exist_ok=True)
    train = Trainer(str(CONFIG_FILE))
    train.env_type = "LvEnv"
    train.setup_env()
    evaluation = Evaluation(train.env, config_file=str(CONFIG_FILE))

    sus, res = [], []
    evaluation.env.reset()
    lv_env = evaluation.env.env
    lv_env.initial_wt = initial_condition[0]
    lv_env.initial_res = initial_condition[1]
    lv_env.state = [lv_env.initial_wt, lv_env.initial_res, 0.0]
    lv_env.growth_rate = [growth_rate_sus, growth_rate_res]
    lv_env.death_rate_treat[0] = death_treat_sus
    lv_env.death_rate = [random_death, random_death]
    lv_env.competition[0] = competition
    lv_env.env_specific_params["ramp_time_up"] = ramp_time_up
    lv_env.env_specific_params["ramp_time_down"] = ramp_time_down
    lv_env.capacity = carrying_capacity
    lv_env.env_specific_params["min_death_during_treat"] = min_treat
    lv_env.on_treat_threshold = on_treat_threshold
    lv_env.off_treat_threshold = off_treat_threshold

    sus.append(lv_env.initial_wt)
    res.append(lv_env.initial_res)

    for i in range(iterations - 1):
        if treatment_type == "NT":
            action = 0
        elif treatment_type == "CT":
            action = 1
        elif treatment_type == "Pulse":
            action = 1 if i == 0 else 0
        elif treatment_type == "AT100":
            action = 1 if exp_data[0][i] + exp_data[1][i] >= exp_data[0][0] + exp_data[1][0] else 0
        obs, _, _, _, _ = evaluation.env.step(action)
        sus.append(float(obs[0]))
        res.append(float(obs[1]))

    return sus, res


def error_components(sen_simulation: np.ndarray, res_simulation: np.ndarray, sen_ground_truth: np.ndarray, res_ground_truth: np.ndarray) -> tuple[float, float]:
    sen_total_sum = float(np.mean(np.sqrt((sen_simulation[:] - sen_ground_truth[:]) ** 2)))
    res_total_sum = float(np.mean(np.sqrt((res_simulation[:] - res_ground_truth[:]) ** 2)))
    return sen_total_sum, res_total_sum


def evaluate_params(params: list[float], experiments: list[ExperimentRecord]) -> tuple[float, list[dict]]:
    results = []
    rows = []
    for record in experiments:
        sus_sim, res_sim = run_simulation(
            params,
            initial_condition=[float(record.sus_exp[0]), float(record.res_exp[0])],
            exp_data=[record.sus_exp, record.res_exp],
            iterations=len(record.sus_exp),
            treatment_type=record.treatment_type,
        )
        sus_sim_arr = np.asarray(sus_sim, dtype=float)
        res_sim_arr = np.asarray(res_sim, dtype=float)
        sen_total_sum, res_total_sum = error_components(sus_sim_arr, res_sim_arr, record.sus_exp, record.res_exp)
        objective = float(sen_total_sum + res_total_sum)
        combined_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        np.concatenate([sus_sim_arr, res_sim_arr])
                        - np.concatenate([record.sus_exp, record.res_exp])
                    ) ** 2
                )
            )
        )
        rows.append(
            {
                "experiment": record.rel_path,
                "treatment_type": record.treatment_type,
                "objective": objective,
                "combined_rmse": combined_rmse,
                "sus_rmse": float(np.sqrt(np.mean((sus_sim_arr - record.sus_exp) ** 2))),
                "res_rmse": float(np.sqrt(np.mean((res_sim_arr - record.res_exp) ** 2))),
            }
        )
        results.append(objective)
    return float(np.mean(results)), rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class FitMonitor:
    def __init__(self, label: str, outdir: Path, maxiter: int):
        self.label = label
        self.outdir = outdir
        self.progress_dir = outdir / "progress"
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.progress_dir / f"{label}.progress.log"
        self.status_path = self.progress_dir / f"{label}.status.json"
        self.maxiter = int(maxiter)
        self.eval_count = 0
        self.iter_count = 0
        self.best_loss = None
        self.best_params = None
        self.last_loss = None
        self.last_params = None
        self.started_at = time.time()
        self.progress_bar = tqdm(total=self.maxiter, desc=label, unit="iter", dynamic_ncols=True)

    def _write_log(self, message: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def _write_status(self, stage: str) -> None:
        payload = {
            "label": self.label,
            "stage": stage,
            "maxiter": self.maxiter,
            "nit": self.iter_count,
            "nfev": self.eval_count,
            "best_objective": self.best_loss,
            "last_objective": self.last_loss,
            "best_params": self.best_params,
            "last_params": self.last_params,
            "elapsed_seconds": float(time.time() - self.started_at),
            "updated_at": dt.datetime.now().isoformat(),
        }
        self.status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def on_eval(self, loss: float, params: np.ndarray) -> None:
        self.eval_count += 1
        self.last_loss = float(loss)
        self.last_params = [float(v) for v in params.tolist()]
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_params = [float(v) for v in params.tolist()]
            self._write_log(f"[{dt.datetime.now().isoformat()}] BEST nfev={self.eval_count} nit={self.iter_count} objective={loss:.12g}")
            self._write_status("running")

    def on_iteration(self, xk: np.ndarray) -> None:
        self.iter_count += 1
        self.last_params = [float(v) for v in xk.tolist()]
        self.progress_bar.update(1)
        if self.best_loss is not None:
            self.progress_bar.set_postfix(best=f"{self.best_loss:.6f}", nfev=self.eval_count)
        self._write_log(
            f"[{dt.datetime.now().isoformat()}] ITER nit={self.iter_count} nfev={self.eval_count} best={self.best_loss:.12g} last={self.last_loss:.12g}"
        )
        self._write_status("running")

    def finalize(self, result: opt.OptimizeResult) -> None:
        self.progress_bar.set_postfix(best=f"{float(result.fun):.6f}", nfev=int(result.nfev))
        self.progress_bar.close()
        payload = {
            "label": self.label,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "fun": float(result.fun),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "optimized_params": [float(v) for v in result.x.tolist()],
        }
        (self.progress_dir / f"{self.label}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._write_log(
            f"[{dt.datetime.now().isoformat()}] END success={bool(result.success)} status={int(result.status)} nit={int(result.nit)} nfev={int(result.nfev)} objective={float(result.fun):.12g}"
        )
        self._write_status("finished")


def fit_from_seed(seed: list[float], experiments: list[ExperimentRecord], outdir: Path, label: str, maxiter: int = 1000) -> opt.OptimizeResult:
    monitor = FitMonitor(label=label, outdir=outdir, maxiter=maxiter)

    def objective(x: np.ndarray) -> float:
        loss, _ = evaluate_params([float(v) for v in x.tolist()], experiments)
        monitor.on_eval(loss, x)
        return loss

    def callback(xk: np.ndarray) -> None:
        monitor.on_iteration(xk)

    result = opt.minimize(
        objective,
        np.asarray(seed, dtype=float),
        method="Nelder-Mead",
        callback=callback,
        options={"disp": False, "maxiter": int(maxiter)},
        bounds=BOUNDS,
    )
    monitor.finalize(result)
    return result
