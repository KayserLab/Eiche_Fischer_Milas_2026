#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import math
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP_ROOT = REPO_ROOT / "PhysiLearning" / "Bootstrapping"
FITTING_REPO = REPO_ROOT / "PhysiLearning" / "LV_Fitting_MiniRepo"
METHODS_BOOTSTRAP = BOOTSTRAP_ROOT / "outputs" / "bootstrap_n20_methods_seed_maxiter1000"
METHODS_PARAMS_PATH = REPO_ROOT / "PhysiLearning" / "LV_Agent_Training_MiniRepo" / "reference" / "fitted_lv_params.json"

for path in (FITTING_REPO / "scripts", FITTING_REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refit_core import DATA_DIR, PARAMETER_ORDER, evaluate_params, load_experiments, run_simulation  # noqa: E402

PARAMETER_LABELS = {
    "growth_rate_sus": "gamma_S / growth_rate_wt",
    "growth_rate_res": "gamma_R / growth_rate_mut",
    "competition": "c / competition_wt",
    "random_death": "delta / death_rate_wt/mut",
    "death_treat_sus": "D / treat_death_rate_wt",
    "ramp_time_up": "beta+ / ramp_time_up",
    "ramp_time_down": "beta- / ramp_time_down",
    "carrying_capacity": "K / carrying_capacity",
    "on_treat_threshold": "tau_on_lag / on_treat_threshold",
    "off_treat_threshold": "tau_off_lag / off_treat_threshold",
    "min_treat": "Delta_min / min_death_during_treat",
}


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_methods_vector() -> list[float]:
    params = json.loads(METHODS_PARAMS_PATH.read_text(encoding="utf-8"))
    return [
        float(params["growth_rate_wt"]),
        float(params["growth_rate_mut"]),
        float(params["competition_wt"]),
        float(params["death_rate_wt"]),
        float(params["treat_death_rate_wt"]),
        float(params["ramp_time_up"]),
        float(params["ramp_time_down"]),
        float(params["carrying_capacity"]),
        float(params["on_treat_threshold"]),
        float(params["off_treat_threshold"]),
        float(params["min_death_during_treat"]),
    ]


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "q025": float(np.quantile(arr, 0.025)),
        "q250": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "q750": float(np.quantile(arr, 0.75)),
        "q975": float(np.quantile(arr, 0.975)),
        "max": float(np.max(arr)),
    }


def objective_summary(rows: list[dict], label: str) -> list[dict]:
    out = []
    for key in ("bootstrap_objective", "full_data_objective", "nit", "nfev"):
        stats = summarize([float(row[key]) for row in rows])
        stats.update({"run": label, "quantity": key})
        out.append(stats)
    out.append(
        {
            "run": label,
            "quantity": "converged_count",
            "n": len(rows),
            "mean": sum(row["success"] == "True" for row in rows),
            "std": "",
            "min": "",
            "q025": "",
            "q250": "",
            "median": "",
            "q750": "",
            "q975": "",
            "max": "",
        }
    )
    return out


def build_parameter_summary(methods_vector: list[float], methods_rows: list[dict]) -> list[dict]:
    rows = []
    for idx, name in enumerate(PARAMETER_ORDER):
        methods_value = float(methods_vector[idx])
        methods_stats = summarize([float(row[name]) for row in methods_rows])
        rows.append(
            {
                "parameter": name,
                "label": PARAMETER_LABELS[name],
                "methods_value": methods_value,
                "methods_seed_q025": methods_stats["q025"],
                "methods_seed_median": methods_stats["median"],
                "methods_seed_q975": methods_stats["q975"],
                "methods_seed_min": methods_stats["min"],
                "methods_seed_max": methods_stats["max"],
                "methods_inside_methods_seed_q025_q975": methods_stats["q025"] <= methods_value <= methods_stats["q975"],
            }
        )
    return rows


def excel_cell(value) -> str:
    if value is None or value == "":
        return "<c/>"
    if isinstance(value, bool):
        return f'<c t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
        return f"<c><v>{float(value):.15g}</v></c>"
    return f'<c t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def sheet_xml(rows: list[dict]) -> str:
    if not rows:
        rows = [{"empty": ""}]
    headers = list(rows[0].keys())
    xml_rows = []
    xml_rows.append("<row>" + "".join(excel_cell(header) for header in headers) + "</row>")
    for row in rows:
        xml_rows.append("<row>" + "".join(excel_cell(row.get(header, "")) for header in headers) + "</row>")
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<sheetData>"
        + "".join(xml_rows)
        + "</sheetData></worksheet>"
    )


def write_xlsx(path: Path, sheets: dict[str, list[dict]]) -> None:
    sheet_names = list(sheets.keys())
    workbook_sheets = []
    workbook_rels = []
    content_overrides = [
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for idx, name in enumerate(sheet_names, start=1):
        safe_name = html.escape(name[:31], quote=True)
        workbook_sheets.append(f'<sheet name="{safe_name}" sheetId="{idx}" r:id="rId{idx}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            + "".join(content_overrides)
            + "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(workbook_sheets)
            + "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(workbook_rels)
            + "</Relationships>",
        )
        for idx, name in enumerate(sheet_names, start=1):
            archive.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(sheets[name]))


def load_days(record) -> np.ndarray:
    data = pd.read_csv(DATA_DIR / record.rel_path, sep=r"\s+", names=["name", "sus", "res"], engine="python")
    return data["name"].str.extract(r"(\d{2})$").astype(int).iloc[: len(record.sus_exp), 0].to_numpy(dtype=float)


def select_representative_records(methods_vector: list[float]) -> tuple[dict[str, object], list[dict]]:
    experiments = load_experiments()
    _, eval_rows = evaluate_params(methods_vector, experiments)
    by_treatment = defaultdict(list)
    for record, eval_row in zip(experiments, eval_rows):
        by_treatment[record.treatment_type].append((record, float(eval_row["objective"])))

    selected = {}
    selected_rows = []
    for treatment in ("AT100", "CT", "NT", "Pulse"):
        records = sorted(by_treatment[treatment], key=lambda item: item[1])
        objectives = np.asarray([objective for _, objective in records], dtype=float)
        median_objective = float(np.median(objectives))
        record, objective = min(records, key=lambda item: abs(item[1] - median_objective))
        selected[treatment] = record
        selected_rows.append(
            {
                "treatment_type": treatment,
                "representative_experiment": record.rel_path,
                "selection_rule": "closest to treatment median objective under methods parameters",
                "methods_objective": objective,
                "treatment_median_methods_objective": median_objective,
            }
        )
    return selected, selected_rows


def plot_dynamics(methods_vector: list[float], methods_rows: list[dict], output_path: Path) -> list[dict]:
    selected, selected_rows = select_representative_records(methods_vector)
    bootstrap_vectors = [
        [float(row[name]) for name in PARAMETER_ORDER]
        for row in methods_rows
    ]

    fig, axes = plt.subplots(4, 2, figsize=(11.5, 13.5), sharex=False)
    colors = {
        "experiment": "#171717",
        "methods": "#155e75",
        "band": "#f59e0b",
        "median": "#9a3412",
    }

    for row_idx, treatment in enumerate(("AT100", "CT", "NT", "Pulse")):
        record = selected[treatment]
        days = load_days(record)
        methods_sus, methods_res = run_simulation(
            methods_vector,
            initial_condition=[float(record.sus_exp[0]), float(record.res_exp[0])],
            exp_data=[record.sus_exp, record.res_exp],
            iterations=len(record.sus_exp),
            treatment_type=record.treatment_type,
        )

        boot_sus = []
        boot_res = []
        for vector in bootstrap_vectors:
            sus, res = run_simulation(
                vector,
                initial_condition=[float(record.sus_exp[0]), float(record.res_exp[0])],
                exp_data=[record.sus_exp, record.res_exp],
                iterations=len(record.sus_exp),
                treatment_type=record.treatment_type,
            )
            boot_sus.append(sus)
            boot_res.append(res)

        for col_idx, (cell_type, exp_values, methods_values, boot_values) in enumerate(
            (
                ("Sensitive cells", record.sus_exp, np.asarray(methods_sus, dtype=float), np.asarray(boot_sus, dtype=float)),
                ("Resistant cells", record.res_exp, np.asarray(methods_res, dtype=float), np.asarray(boot_res, dtype=float)),
            )
        ):
            ax = axes[row_idx, col_idx]
            lower = np.quantile(boot_values, 0.025, axis=0)
            upper = np.quantile(boot_values, 0.975, axis=0)
            median = np.quantile(boot_values, 0.5, axis=0)
            ax.fill_between(days, lower, upper, color=colors["band"], alpha=0.28, label="Bootstrap LV 95% band")
            ax.plot(days, median, color=colors["median"], linewidth=1.4, linestyle="--", label="Bootstrap LV median")
            ax.plot(days, methods_values, color=colors["methods"], linewidth=2.0, label="Methods LV")
            ax.plot(days, exp_values, color=colors["experiment"], marker="o", linewidth=1.7, markersize=4.2, label="Experiment")
            ax.set_title(f"{treatment} - {cell_type}", fontsize=11, weight="bold")
            ax.set_ylabel("Normalized density")
            ax.grid(True, alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_idx == 3:
                ax.set_xlabel("Day")
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(
        "LV Dynamics: Methods Parameters vs Methods-Seed Bootstrap Range",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return selected_rows


def main() -> int:
    os.chdir(FITTING_REPO)

    methods_vector = load_methods_vector()
    methods_rows = read_csv_rows(METHODS_BOOTSTRAP / "bootstrap_results.csv")

    parameter_summary = build_parameter_summary(methods_vector, methods_rows)
    objective_rows = objective_summary(methods_rows, "methods_seed")

    image_path = METHODS_BOOTSTRAP / "LV_methods_seed_bootstrap_dynamics.png"
    representative_rows = plot_dynamics(methods_vector, methods_rows, image_path)

    exact_methods_objective, _ = evaluate_params(methods_vector, load_experiments())
    converged = sum(row["success"] == "True" for row in methods_rows)
    outside = [
        row["parameter"]
        for row in parameter_summary
        if not row["methods_inside_methods_seed_q025_q975"]
    ]
    notes = [
        {
            "item": "methods_parameter_source",
            "value": str(METHODS_PARAMS_PATH),
        },
        {
            "item": "methods_seed_bootstrap_source",
            "value": str(METHODS_BOOTSTRAP),
        },
        {
            "item": "exact_methods_full_data_objective",
            "value": exact_methods_objective,
        },
        {
            "item": "bootstrap_convergence",
            "value": f"{converged} of {len(methods_rows)} optimizer runs converged; {len(methods_rows) - converged} reached maxiter.",
        },
        {
            "item": "methods_params_outside_q025_q975",
            "value": ", ".join(outside),
        },
        {
            "item": "trajectory_band",
            "value": "Per-timepoint q025-q975 across the 20 methods-seed bootstrap parameter vectors.",
        },
    ]

    xlsx_path = METHODS_BOOTSTRAP / "LV_methods_seed_bootstrap_summary.xlsx"
    write_xlsx(
        xlsx_path,
        {
            "parameter_summary": parameter_summary,
            "objective_summary": objective_rows,
            "representatives": representative_rows,
            "notes": notes,
        },
    )

    full_objectives = [float(row["full_data_objective"]) for row in methods_rows]
    boot_objectives = [float(row["bootstrap_objective"]) for row in methods_rows]
    full_stats = summarize(full_objectives)
    boot_stats = summarize(boot_objectives)
    doc_path = METHODS_BOOTSTRAP / "results_documentation.txt"
    doc_path.write_text(
        "\n".join(
            [
                "Methods-seed LV bootstrap results",
                "=================================",
                "",
                f"Output directory: {METHODS_BOOTSTRAP}",
                "Bootstrap unit: experimental replicate",
                "Sampling: stratified by treatment type, with replacement",
                f"Bootstrap replicates: {len(methods_rows)}",
                "Optimizer: Nelder-Mead through the validated LV fitting pipeline",
                "Initialization: methods / agent-training LV parameter vector",
                f"Optimizer convergence: {converged}/{len(methods_rows)} converged; {len(methods_rows) - converged}/{len(methods_rows)} reached maxiter.",
                "",
                "Objective summary",
                f"Exact methods-parameter full-data objective: {exact_methods_objective:.12g}",
                f"Bootstrap-objective median: {boot_stats['median']:.12g}",
                f"Bootstrap-objective q025-q975: {boot_stats['q025']:.12g} to {boot_stats['q975']:.12g}",
                f"Full-data objective median after bootstrap refits: {full_stats['median']:.12g}",
                f"Full-data objective q025-q975 after bootstrap refits: {full_stats['q025']:.12g} to {full_stats['q975']:.12g}",
                "",
                "Methods parameters relative to methods-seed bootstrap q025-q975 intervals",
                f"Inside interval: {len(PARAMETER_ORDER) - len(outside)}/{len(PARAMETER_ORDER)}",
                f"Outside interval: {', '.join(outside)}",
                "",
                "Trajectory figure",
                f"PNG: {image_path.name}",
                "Band definition: pointwise q025-q975 LV trajectory band across the 20 methods-seed bootstrap parameter vectors.",
                "Representative replicate per treatment: closest to treatment-specific median objective under the methods parameter vector.",
                "",
                "Spreadsheet",
                f"XLSX: {xlsx_path.name}",
                "Sheets: parameter_summary, objective_summary, representatives, notes.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {xlsx_path}")
    print(f"Wrote {image_path}")
    print(f"Wrote {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
