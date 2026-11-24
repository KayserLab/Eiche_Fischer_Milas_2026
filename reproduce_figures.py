#!/usr/bin/env python3
"""Regenerate Figure 1--4 and supplementary plots from frozen figure inputs.

Preview: python reproduce_figures.py --dry-run
Run:     python reproduce_figures.py --data-root "C:/path/to/Data"
Optional examples: --demo czi or --demo simulation.
The original full analysis requires --mode full and additional cohort data.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import threading

from shared.data_paths import DATA_DIRECTORY
from reproduction.scheduler import execute_plan


CONSOLE_LOCK = threading.RLock()


def configure_utf8_output():
    """Keep redirected Windows console output compatible with child UTF-8 logs."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors=getattr(stream, "errors", "strict"))

# Set DATA_DIRECTORY in shared/data_paths.py. The --data-root argument overrides it for one run.
# Independent plates/runs only. Reduce these if RAM or shared-drive I/O is limiting.
PLATE_WORKERS = 1
ANALYSIS_WORKERS = 1
# Independent, audited stages may overlap when CPU and RAM are available.
STAGE_WORKERS = 1
TOTAL_WORKER_BUDGET = min(6, os.cpu_count() or 1)
MIN_FREE_MEMORY_GB = 2
# Resume the interrupted run when using PyCharm's Run button.
# Set to None (or pass --from-step colony_masks) to start from the beginning.
# Otherwise use one of the quoted step names below, in execution order.
# The selected step and all later steps run; earlier prerequisites must exist.
# Explicit --from-step or --only arguments override this setting.
#
# Preparation:
#   "colony_masks"                 - prepare colony masks and stacks
#   "clone_tracking"               - track clones in mask stacks
#   "clean_sensitive"              - clean sensitive-cell masks
#   "escape_inputs"                - prepare experimental escape inputs
#   "simulation_overview"          - build simulation overview
#   "simulation_points"            - prepare simulation point clouds
#   "escape_margins"               - calculate escape margins
# Figure 1:
#   "fig1_assay"                   - Alamar Blue assay
#   "fig1_trajectories"             - selected well trajectories
#   "fig1_legend"                  - trajectory legend
#   "fig1_movie_frames"            - export fluorescence movie frames
# Figure 2:
#   "fig2_tte"                     - time to escape
#   "fig2_ttf"                     - time to failure
#   "fig2_ttf_vs_tte"               - compare failure and escape times
#   "fig2_examples"                - selected experimental escape margins
#   "fig2_examples_side_by_side"   - side-by-side escape-margin examples
#   "fig2_ratchet"                 - ratchet plot
# Figure 3:
#   "master_curves"                - compute master curves
#   "cell_size_density"            - compute cell size versus density
#   "fig3_master_120"              - master curves at 120 micrometres
#   "fig3_master_160"              - master curves at 160 micrometres
#   "fig3_cell_size"               - cell-size plot
#   "fig3_maps"                    - AT50 density maps
#   "fig3_trajectories"             - AT50 absolute-density trajectories
# Figure 4:
#   "policy_behavior"              - analyse RL policy behaviour
#   "fig4_tables"                  - prepare Figure 4 plot tables
#   "fig4_density"                 - normalised density plot
#   "fig4_lv_example"              - experimental versus LV example
#   "fig4_agent_examples"          - agent example trajectories
#   "fig4_policy"                  - policy-behaviour plot
#   "fig4_survival"                - Kaplan-Meier survival plot
#   "fig4_burden"                  - cumulative treatment and burden
#   "dpd_decisions"                - extract DPD policy decisions
#   "experimental_decisions"       - extract experimental policy decisions
#   "fig4_portrait"                - experimental DPD policy portrait
# Supplementary figures:
#   "suppl_initial_margin"         - initial escape-margin histogram
#   "suppl_escape_time"            - escape margin over time
#   "suppl_escape_burden"          - escape margin versus total burden
#   "suppl_resistant_maps"         - resistant-cell density maps
#   "suppl_relative_growth"        - relative-growth comparison
#   "suppl_all_ttf"                - failure times for all strategies
#   "suppl_km_atplus"              - ATplus simulation Kaplan-Meier plot
#   "matched_events"               - matched experimental/DPD events
#   "simulation_events"            - matched simulation events
#   "suppl_rank_agreement"         - rank consistency
#   "suppl_initial_conditions"     - experimental initial conditions
#   "suppl_mean_ranks"             - experimental cohort mean ranks
#   "suppl_day1_counts"            - day-1 cell counts by group
START_FROM_STEP = None
ROOT = Path(__file__).resolve().parent
POLICY = "PhysiLearning/LV_Agent_Training_MiniRepo/analysis/policy_distillation_clean/"
# Keep policy-decision CSVs alongside PhysiLearning, independently of DATA_DIRECTORY.
POLICY_DATA = "{data_root}/intermediate_analysis_data/policy_decisions/"
TRAINING = ROOT / "PhysiLearning" / "LV_Agent_Training_MiniRepo"
AGENT = ROOT / "PhysiLearning/Agent_used_for_experiment/2025_10_01_lv_dont_treat_1_load_best_reward"


@dataclass(frozen=True)
class Step:
    name: str
    group: str
    script: str
    requires: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    environment: str = "root"


def build_steps() -> list[Step]:
    steps = []

    def add(name, group, script, requires=(), args=(), environment="root"):
        steps.append(Step(name, group, script, tuple(requires), tuple(args), environment))

    add("colony_masks", "prepare", "analysis/basic/microscopy/Make_susrestotal_Colony_Masks.py",
        args=["--plate-workers", "{plate_workers}"])
    add("clone_tracking", "prepare", "analysis/basic/microscopy/single_clone_tracking.py", ["colony_masks"],
        ["--plate-workers", "{plate_workers}"])
    add("clean_sensitive", "prepare", "analysis/basic/escape/Escape_Remove_Sen_Artifacts.py", ["clone_tracking"],
        ["--plate-workers", "{plate_workers}"])
    add("escape_inputs", "prepare", "analysis/basic/escape/Prepare_Escape_Analysis_Inputs.py",
        ["clone_tracking", "clean_sensitive"], ["--workers", "{workers}"])
    add("simulation_overview", "prepare", "analysis/figure_2/build_simulation_overview.py",
        args=["--workers", "{workers}"])
    add("simulation_points", "prepare", "analysis/basic/density_maps/Prepare_Simulation_Pointclouds.py",
        ["simulation_overview"], ["--workers", "{workers}"])
    add("escape_margins", "prepare", "analysis/figure_2/Escape_Margin_Analysis.py",
        ["escape_inputs", "simulation_points"])

    for name, script in (
        ("fig1_assay", "c_alamar_blue_a2_a4_plot.py"),
        ("fig1_trajectories", "g_plot_selected_well_trajectories.py"),
        ("fig1_legend", "g_plot_selected_well_trajectories_legend.py"),
        ("fig1_movie_frames", "export_movie_fluorescence_composites.py"),
    ):
        add(name, "fig1", "figures/Figure_1/" + script)

    for name, script in (
        ("fig2_tte", "b_TTE CT & AT100.py"),
        ("fig2_ttf", "c_TTF CT & AT100.py"),
        ("fig2_ttf_vs_tte", "d_plot_ttf_vs_tte.py"),
        ("fig2_examples", "f_plot_selected_invitro_escape_margin.py"),
        ("fig2_examples_side_by_side", "f_plot_selected_invitro_escape_margin_side_by_side_174mm.py"),
        ("fig2_ratchet", "g_plot_ratchet.py"),
    ):
        add(name, "fig2", "figures/Figure_2/" + script, ["escape_margins"])

    add("master_curves", "fig3", "analysis/figure_3/HDF5_analysis/compute_master_curves.py",
        ["escape_margins"])
    add("cell_size_density", "fig3", "analysis/figure_3/HDF5_analysis/compute_cell_size_density.py")
    for radius in (120, 160):
        add(f"fig3_master_{radius}", "fig3", "figures/Figure_3/plot_hdf5_master_curves.py",
            ["master_curves"], ["--kernel-radius-um", str(radius)])
    add("fig3_cell_size", "fig3", "figures/Figure_3/b_plot_cell_size_vs_density.py", ["cell_size_density"])
    add("fig3_maps", "fig3", "figures/Figure_3/plot_at50_density_maps.py", ["escape_margins"])
    add("fig3_trajectories", "fig3", "figures/Figure_3/plot_at50_hdf5_absolute_density_trajectories.py",
        ["escape_margins"])

    add("policy_behavior", "fig4", "analysis/figure_4/run_rl_policy_behavior_analysis.py",
        args=["--mode", "both"], environment="policy")
    add("fig4_tables", "fig4", "analysis/figure_4/ef_Figure_4_prepare_plot_data.py",
        ["simulation_overview", "simulation_points"])
    add("fig4_density", "fig4", "figures/Figure_4/a_plot_area_average_total_density_vs_normalized_total_hdf5.py",
        ["escape_margins"], ["--kernel-radius-um", "120"])
    add("fig4_lv_example", "fig4", "figures/Figure_4/b_plot_plate5_a2_invitro_vs_lv_full_x_scale.py",
        ["escape_margins"])
    add("fig4_agent_examples", "fig4", "figures/Figure_4/d_Figure_4_example_trajectories.py",
        ["escape_margins"], environment="policy")
    add("fig4_policy", "fig4", "figures/Figure_4/d2_replot_rl_policy_behavior.py", ["policy_behavior"])
    add("fig4_survival", "fig4", "figures/Figure_4/e_Figure_4_Kaplan_Meier_in_vitro_and_in_silico.py",
        ["fig4_tables"])
    add("fig4_burden", "fig4",
        "figures/Figure_4/f_Figure_4_Cumulative_Treatment_average_burden_in_vitro_and_in_silico.py",
        ["fig4_tables"])
    add("dpd_decisions", "fig4", POLICY + "02_load_dpd_decisions.py",
        args=["--output", POLICY_DATA + "dpd_decisions.csv"], environment="policy")
    add("experimental_decisions", "fig4", POLICY + "02d_load_experimental_decisions.py",
        args=["--data-root", "{data_root}/exp_raw_data",
              "--output", POLICY_DATA + "experimental_decisions.csv"], environment="policy")
    add("fig4_portrait", "fig4", "figures/Figure_4/Figure_4_Experimental_DPD_Policy_Portrait.py",
        ["dpd_decisions", "experimental_decisions"],
        ["--output-dir", "{data_root}/figure_plots/Figure_4",
         "--experimental-csv", POLICY_DATA + "experimental_decisions.csv",
         "--dpd-csv", POLICY_DATA + "dpd_decisions.csv"])

    for name, script in (
        ("suppl_initial_margin", "Initial_Escape_Margin_Histogram.py"),
        ("suppl_escape_time", "Escape_Margin_Over_Time_By_Strategy.py"),
        ("suppl_escape_burden", "Escape_Margin_vs_Total_Burden_All_Replicates.py"),
        ("suppl_resistant_maps", "plot_at50_resistant_density_maps.py"),
    ):
        add(name, "suppl", "figures/Suppl_Figures/" + script, ["escape_margins"])
    add("suppl_relative_growth", "suppl", "figures/Suppl_Figures/RelativeGrowth_alpha2_alpha4_side_by_side.py")
    add("suppl_all_ttf", "suppl", "figures/Suppl_Figures/TTF_All_Strategies_InVitro_DPD_LV.py",
        ["simulation_points"], environment="policy")
    add("suppl_km_atplus", "suppl", "figures/Suppl_Figures/Kaplan_Meier_Simulation_ATplus.py",
        ["simulation_points", "escape_margins", "simulation_overview"], ["--hide-figure"])
    add("matched_events", "suppl", "figures/Suppl_Figures/TTF_TTE_Matched_InVitro_DPD.py",
        ["escape_margins", "simulation_overview"])
    add("simulation_events", "suppl", "figures/Suppl_Figures/plot_TTF_TTE_Matched_Simulation_Strategies.py",
        ["escape_margins", "simulation_overview"])
    for name, script in (
        ("suppl_rank_agreement", "TTF_TTE_Matched_Exp_Sim_Rank_Consistency.py"),
        ("suppl_initial_conditions", "TTF_TTE_Experimental_Cohort_Initial_Conditions.py"),
        ("suppl_mean_ranks", "TTF_TTE_Experimental_Cohort_Mean_Ranks.py"),
        ("suppl_day1_counts", "Day1_Cell_Counts_By_Experimental_Group.py"),
    ):
        add(name, "suppl", "figures/Suppl_Figures/" + script, ["matched_events", "simulation_events"])
    add('raw_image_demo', 'demo', 'analysis/basic/microscopy/raw_image_demo.py', environment='cellpose')
    add('raw_demo_report', 'demo', 'analysis/basic/microscopy/raw_demo_report.py', ['raw_image_demo'])
    add('manuscript_statistics', 'stats', 'verification/reproduce_manuscript_statistics.py',
        ['fig2_tte', 'fig2_ttf', 'fig2_ttf_vs_tte', 'fig3_master_120', 'fig3_master_160',
         'fig4_density', 'fig4_survival', 'fig4_burden'])
    return steps


def validate_steps(steps):
    seen = set()
    for step in steps:
        if step.name in seen:
            raise ValueError(f"Duplicate step: {step.name}")
        if set(step.requires) - seen:
            raise ValueError(f"{step.name}: prerequisites must occur earlier: {step.requires}")
        if any("legacy" in part.lower() for part in Path(step.script).parts):
            raise ValueError(f"Legacy entry point is not allowed: {step.script}")
        if not (ROOT / step.script).is_file():
            raise ValueError(f"Missing script: {step.script}")
        seen.add(step.name)


def select_steps(steps, groups=None, start=None, skip=()):
    by_name = {step.name: step for step in steps}
    selected = {step.name for step in steps if not groups or step.group in groups}

    def include_dependencies(name):
        for required in by_name[name].requires:
            if required not in selected:
                selected.add(required)
                include_dependencies(required)

    for name in list(selected):
        include_dependencies(name)
    result = [step for step in steps if step.name in selected]
    if start:
        names = [step.name for step in result]
        if start not in names:
            raise ValueError(f"--from-step is not in this plan: {start}")
        result = result[names.index(start):]
    unknown = set(skip) - set(by_name)
    if unknown:
        raise ValueError(f"Unknown --skip step(s): {sorted(unknown)}")
    result = [step for step in result if step.name not in skip]
    for step in result:
        removed = set(step.requires) & set(skip)
        if removed:
            raise ValueError(f"Cannot skip {sorted(removed)}: required by {step.name}. "
                             "Use --from-step to reuse previously generated upstream data.")
    return result


def command_for(step, interpreters, data_root, workers=None, plate_workers=None):
    replacements = {"{data_root}": str(data_root),
                    "{workers}": str(ANALYSIS_WORKERS if workers is None else workers),
                    "{plate_workers}": str(PLATE_WORKERS if plate_workers is None else plate_workers)}
    arguments = []
    for arg in step.args:
        for placeholder, value in replacements.items():
            arg = arg.replace(placeholder, value)
        arguments.append(arg)
    return [interpreters[step.environment], "-u", str(ROOT / 'reproduction/runtime.py'),
            step.name, str(ROOT / step.script), *arguments]


def child_environment(data_root, movie_dir=None, workers=None):
    env = os.environ.copy()
    # Discard inherited path overrides so the master directory is authoritative.
    # Do not clear numerical settings (kernel radius, worker count, etc.).
    for name in (
        "HDF5_ANALYSIS_DATA_DIR", "MAJA_ANALYSIS_DATA_DIR",
        "HDF5_SIMULATION_INPUT_DIR", "MAJA_SIMULATION_INPUT_DIR",
        "HDF5_EXPERIMENT_INPUT_DIR", "MAJA_EXPERIMENT_INPUT_DIR",
        "FLUX_OUTPUT_DIR", "EXP_FLUX_OUTPUT_DIR", "FLUX_FIGURES_DIR",
        "CORESEED_EXPERIMENT_PATH", "CORESEED_MOUNT_ROOT",
    ):
        env.pop(name, None)
    env.update(EICHE_FISCHER_DATA_ROOT=str(data_root), MPLBACKEND="Agg",
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               PYTHONFAULTHANDLER="1")
    env['PYTHONHASHSEED'] = '0'
    env['CELLPOSE_LOCAL_MODELS_PATH'] = str(data_root / 'Models/Cellpose')
    env['TORCH_HOME'] = str(data_root / 'tmp/torch_cache')
    env['MPLCONFIGDIR'] = str(data_root / 'tmp/matplotlib')
    cache = str(data_root / 'tmp/numba_cache')
    env['NUMBA_CACHE_DIR'] = '\\\\?\\' + cache if os.name == 'nt' else cache
    temporary = data_root / 'tmp/runtime_temp'
    temporary.mkdir(parents=True, exist_ok=True)
    env['TMP'] = env['TEMP'] = env['TMPDIR'] = str(temporary)
    env['PYTHONPATH'] = str(ROOT)
    env['PYTHONNOUSERSITE'] = '1'
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    env['EICHE_FISCHER_STRICT_IO'] = '1'
    env["EICHE_FISCHER_MOVIE_DIR"] = str(movie_dir or data_root / "figure_plots/Figure_1/Movies")
    if workers is not None:
        env["HDF5_N_JOBS"] = env["HDF5_EXPERIMENT_N_JOBS"] = str(workers)
    return env


def preflight(steps, interpreters, data_root, movie_dir):
    """Check external inputs only for an actual run; child scripts validate full cohorts."""
    errors = []
    if not data_root.is_dir():
        errors.append(f"Data directory does not exist: {data_root}")
    for environment in {step.environment for step in steps}:
        if not Path(interpreters[environment]).is_file():
            errors.append(f"Missing {environment} Python: {interpreters[environment]}")
    names = {step.name for step in steps}
    required = []
    if "escape_inputs" in names or "colony_masks" in names:
        required.append(data_root / "exp_raw_data")
    if "simulation_points" in names or "simulation_overview" in names:
        required.append(data_root / "sim_raw_data/data-DB")
    if "cell_size_density" in names:
        required.extend(data_root / "sim_raw_data/pressure_analysis/pressure_data/pressure_data" / f"{p}.h5"
                        for p in ("AT50", "AT100", "CT"))
    if "fig1_movie_frames" in names:
        folder = movie_dir or data_root / "figure_plots/Figure_1/Movies"
        required.extend(folder / f for f in ("D3_AT50-D3_zoom.tif", "A2_AT100-A2.tif", "A1_CT-A1.tif"))
    if any(step.environment == "policy" for step in steps):
        required.extend(data_root / 'Models/experimental_agent' / f for f in ("data", "policy.pth", "pytorch_variables.pth"))
    if "dpd_decisions" in names:
        required.append(data_root / "sim_raw_data/data-DB/data-DB_Agent/outputs")
    if "fig4_lv_example" in names:
        required.append(data_root / 'Fitting/PhysiLearning/LV_Agent_Training_MiniRepo/reference/fitted_lv_params.json')
    if "policy_behavior" in names:
        required.append(data_root / 'Models/bootstrap')
    if 'raw_image_demo' in names or 'raw_demo_report' in names:
        demo = data_root / 'Raw_Demo/Plate3_D4_AT100_tp018_day040'
        required.extend([demo / '20251010_d40_Plate3_MIPs.czi', demo / 'settings.json',
                         demo / 'Reference/corrected_padded.h5'])
        required.extend(data_root / 'Models/Cellpose' / name for name in (
            'a2_cyto3_Epi_1000_PhD', 'a4_cyto2_PhD_1000_new'))
    for path in required:
        if not path.exists():
            errors.append(f"Missing input: {path}")
    if errors:
        raise ValueError("Preflight failed before running any analyses:\n  " + "\n  ".join(errors))


def run_step(command, env, log_path, *, stop_event=None):
    """Stream merged stdout/stderr to the console and a per-step log."""
    with log_path.open("w", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1, start_new_session=(os.name != 'nt')) as process:
            finished = threading.Event()

            def stop_tree():
                if process.poll() is not None:
                    return
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    import signal
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            def watch_cancellation():
                while not finished.wait(.2):
                    if stop_event.is_set():
                        stop_tree()
                        return

            if stop_event is not None:
                threading.Thread(target=watch_cancellation, daemon=True).start()
            try:
                for line in process.stdout:
                    prefix = env.get('PIPELINE_STAGE_LABEL')
                    with CONSOLE_LOCK:
                        print(f'[{prefix}] {line}' if prefix else line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return process.wait()
            except KeyboardInterrupt:
                stop_tree()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                finished.set()


def desktop_directory():
    """Use Windows' configured Desktop, including redirected/OneDrive desktops."""
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
            value, _ = winreg.QueryValueEx(key, "Desktop")
        return Path(os.path.expandvars(value))
    return Path.home() / "Desktop"


class ConsoleLog:
    """Copy output to a timestamped file without changing console presentation."""
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
        self.line_start = True
        self.last_sync = time.monotonic()

    def write(self, text):
        with CONSOLE_LOCK:
            return self._write(text)

    def _write(self, text):
        # Save locally before writing to the console or any network stage log.
        for part in text.splitlines(keepends=True):
            if self.line_start:
                self.log.write(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] ")
            self.log.write(part)
            self.line_start = part.endswith(("\n", "\r"))
        self.log.flush()
        if time.monotonic() - self.last_sync >= 1:
            os.fsync(self.log.fileno())
            self.last_sync = time.monotonic()
        self.stream.write(text)
        self.stream.flush()
        return len(text)

    def flush(self):
        with CONSOLE_LOCK:
            self.log.flush()
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def run_with_desktop_log(argv=None, *, desktop=None):
    """Keep one live transcript per launch, independently of the data drive."""
    configure_utf8_output()
    folder = Path(desktop) if desktop is not None else desktop_directory()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"Eiche_Fischer_pipeline_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    with path.open("x", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(ConsoleLog(sys.stdout, log)), redirect_stderr(ConsoleLog(sys.stderr, log)):
            try:
                print(f"Desktop log: {path}", flush=True)
                print(f"Python: {sys.executable} | PID: {os.getpid()}", flush=True)
                result = main(argv)
                print(f"Runner exited with code {result}", flush=True)
                return result
            except SystemExit as exc:
                print(f"Runner exited with code {exc.code}", file=sys.stderr, flush=True)
                raise
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                log.flush()
                os.fsync(log.fileno())


def full_main(argv=None):
    configure_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, help="Override DATA_DIRECTORY from shared/data_paths.py")
    parser.add_argument("--python", dest="root_python", default=sys.executable, help="Root analysis Python")
    installed_path = ROOT / 'environments' / '.local' / 'installed.json'
    if not installed_path.is_file():
        installed_path = ROOT / 'environments' / 'installed.json'
    installed = json.loads(installed_path.read_text(encoding='utf-8')) if installed_path.is_file() else {}
    analysis_python = ROOT / installed['analysis']['python'] if 'analysis' in installed else Path(sys.executable)
    parser.set_defaults(root_python=str(analysis_python))
    policy_python = ROOT / installed['policy']['python'] if 'policy' in installed else (
        ROOT / 'environments/.local/envs/policy' / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    parser.add_argument("--policy-python", default=str(policy_python), help="PhysiLearning Python 3.10 environment")
    cellpose_python = ROOT / installed['cellpose']['python'] if 'cellpose' in installed else (
        ROOT / 'environments/.local/envs/cellpose' / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    parser.add_argument('--cellpose-python', default=str(cellpose_python), help='Declared raw-image/Cellpose interpreter')
    parser.add_argument("--only", nargs="+", choices=("prepare", "fig1", "fig2", "fig3", "fig4", "suppl", "demo", "stats"),
                        help="Selected groups, automatically including their prerequisites")
    parser.add_argument("--from-step", help="Start at this step; assumes earlier products already exist")
    parser.add_argument("--skip", action="append", default=[], metavar="STEP", help="Omit an independent step (repeatable)")
    parser.add_argument("--movie-dir", type=Path, help="Folder containing the three existing Figure 1 movie TIFFs")
    parser.add_argument("--list", action="store_true", help="List steps without reading external data or launching children")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without reading external data or running scripts")
    parser.add_argument("--check", action="store_true", help="Check paths/required inputs only; do not run analyses")
    parser.add_argument("--log-dir", type=Path, help="Default: DATA_DIRECTORY/tmp/pipeline_runs")
    parser.add_argument("--plate-workers", type=int, default=PLATE_WORKERS,
                        help="Independent plates processed concurrently")
    parser.add_argument("--workers", type=int, default=ANALYSIS_WORKERS,
                        help="Workers for run/replicate preparation and HDF5 analyses")
    parser.add_argument('--stage-workers', type=int, default=STAGE_WORKERS,
                        help='Maximum simultaneous independent stages; 1 restores sequential execution')
    parser.add_argument('--worker-budget', type=int, default=TOTAL_WORKER_BUDGET,
                        help='Combined worker allowance for overlapping stages')
    parser.add_argument('--min-free-memory-gb', type=float, default=MIN_FREE_MEMORY_GB,
                        help='RAM reserve before launching an additional stage')
    args = parser.parse_args(argv)
    if args.from_step is None and not args.only:
        args.from_step = START_FROM_STEP
    if args.plate_workers < 1 or args.workers < 1:
        parser.error("--plate-workers and --workers must be positive")
    if args.stage_workers < 1 or args.worker_budget < 1:
        parser.error('--stage-workers and --worker-budget must be positive')
    if not 0 <= args.min_free_memory_gb < float('inf'):
        parser.error('--min-free-memory-gb must be finite and nonnegative')
    try:
        steps = build_steps()
        validate_steps(steps)
        plan = select_steps(steps, args.only, args.from_step, args.skip)
        if not plan:
            raise ValueError("No steps selected")
        data_root = Path(args.data_root or DATA_DIRECTORY).expanduser().absolute()
        text = os.fspath(data_root)
        if os.name == 'nt' and len(text) >= 100 and not text.startswith('\\\\?\\'):
            data_root = Path('\\\\?\\UNC\\' + text[2:] if text.startswith('\\\\') else '\\\\?\\' + text)
        print(f"Data directory for all stages: {data_root}")
        interpreters = {
            name: str(Path(shutil.which(value) or value).expanduser().absolute())
            for name, value in (("root", args.root_python), ("policy", args.policy_python), ('cellpose', args.cellpose_python))
        }
        commands = [(step, command_for(step, interpreters, data_root, args.workers, args.plate_workers)) for step in plan]
        print(f"Stage concurrency: up to {args.stage_workers}; worker budget: {args.worker_budget}")
        for number, (step, command) in enumerate(commands, 1):
            print(f"{number:02d}. {step.name} [{step.group}; {step.environment}]")
            print("    " + shlex.join(command))
        if args.from_step:
            print("Earlier prerequisites are assumed to exist; they are not regenerated.")
        if args.list or args.dry_run:
            return 0
        preflight(plan, interpreters, data_root, args.movie_dir)
        if args.check:
            print("Input path checks passed. File contents and scientific compatibility are checked by each stage.")
            return 0
        log_dir = (args.log_dir or data_root / "tmp/pipeline_runs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_dir.mkdir(parents=True, exist_ok=False)
        env = child_environment(data_root, args.movie_dir, workers=args.workers)
        report = {"data_root": str(data_root), "status": "running", "steps": [],
                  'stage_workers': args.stage_workers, 'worker_budget': args.worker_budget}

        def save_report():
            (log_dir / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        save_report()
        print(f"Logs: {log_dir}", flush=True)
        def announce(message):
            with CONSOLE_LOCK:
                print(message, flush=True)

        return execute_plan(
            commands, env, log_dir, report, save_report, run_step, announce,
            max_stages=args.stage_workers, worker_budget=args.worker_budget,
            plate_workers=args.plate_workers, analysis_workers=args.workers,
            min_free_gb=args.min_free_memory_gb,
        )
    except ValueError as exc:
        parser.error(str(exc))


def main(argv=None):
    configure_utf8_output()
    route = argparse.ArgumentParser(add_help=False)
    route.add_argument('--mode', choices=('plots', 'full'), default='plots')
    options, remaining = route.parse_known_args(sys.argv[1:] if argv is None else argv)
    if options.mode == 'full':
        return full_main(remaining)
    from reproduction.plot_runner import main as plot_main
    return plot_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
