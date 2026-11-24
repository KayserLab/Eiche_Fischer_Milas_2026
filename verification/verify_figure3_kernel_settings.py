"""Check radius comparisons and stale-cache rejection using synthetic data only."""

from __future__ import annotations

import atexit
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
_scratch = tempfile.TemporaryDirectory(prefix="figure3_kernel_settings_", dir=ROOT / "tmp")
atexit.register(_scratch.cleanup)
SCRATCH = Path(_scratch.name)
# Some analysis modules create output directories on import. Keep all those
# paths here, so these checks never read or write the laboratory data drive.
for name, value in {
    "EICHE_FISCHER_DATA_ROOT": SCRATCH,
    "HDF5_KERNEL_RADIUS_UM": "160",
    "HDF5_ANALYSIS_DATA_DIR": SCRATCH / "analysis",
    "HDF5_SIMULATION_INPUT_DIR": SCRATCH / "simulation_inputs",
    "HDF5_EXPERIMENT_INPUT_DIR": SCRATCH / "experiment_inputs",
    "FLUX_OUTPUT_DIR": SCRATCH / "simulation_outputs",
    "EXP_FLUX_OUTPUT_DIR": SCRATCH / "experiment_outputs",
    "FLUX_FIGURES_DIR": SCRATCH / "plots",
    "MPLCONFIGDIR": SCRATCH / "matplotlib",
}.items():
    os.environ[name] = str(value)
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(ROOT))

from analysis.figure_3.HDF5_analysis import compute_master_curves as diagnostics
from analysis.figure_3.HDF5_analysis.constants import kernel_settings
from analysis.figure_3.HDF5_analysis import constants
from figures.Figure_3 import plot_at50_density_maps as maps
from figures.Figure_3 import plot_at50_hdf5_absolute_density_trajectories as trajectories
from figures.Figure_3 import plot_hdf5_master_curves as master


class KernelSettingsChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def diagnostic_file(self, radius):
        path = self.root / f"diagnostic_{radius}.h5"
        with h5py.File(path, "w") as handle:
            handle.attrs.update(kernel_settings(radius))
            for dataset in ("experiment", "simulation"):
                for strategy in diagnostics.STRATEGIES:
                    group = handle.create_group(f"{dataset}/{strategy}")
                    group.create_dataset("run_names", data=np.array([b"run_18"]))
                    for period in diagnostics.PERIODS:
                        for variant in diagnostics.VARIANTS:
                            curves = group.create_group(f"{period}/{variant}")
                            for name in ("median", "q25", "q75", "count",
                                         "median_rel", "q25_rel", "q75_rel", "count_rel"):
                                curves.create_dataset(name, data=np.ones((1, 3)))
        return path

    def test_neighbour_gate_matches_for_both_radii(self):
        for radius, minimum in (
            (maps.HDF5_BASE_KERNEL_RADIUS_UM, maps.HDF5_BASE_MIN_NEIGHBORS),
            (maps.HDF5_VARIANT_KERNEL_RADIUS_UM, maps.HDF5_VARIANT_MIN_NEIGHBORS),
        ):
            for count in (13, 14):
                with self.subTest(radius=radius, count=count):
                    total, _, valid = maps._hdf5_kernel_density_field(
                        np.zeros((count, 2)), np.zeros(count, dtype=int),
                        np.zeros((1, 2)), radius, (1, 1), min_neighbors=minimum,
                    )
                    self.assertEqual(bool(valid[0, 0]), count == 14)
                    np.testing.assert_allclose(total * radius ** 2, 2 * count / np.pi)

    def test_resistant_gate_matches_for_both_radii(self):
        for radius, numerator in (
            (maps.HDF5_BASE_KERNEL_RADIUS_UM, maps.HDF5_BASE_MIN_RESISTANT_NUMERATOR),
            (maps.HDF5_VARIANT_KERNEL_RADIUS_UM, maps.HDF5_VARIANT_MIN_RESISTANT_NUMERATOR),
        ):
            with self.subTest(radius=radius):
                native_radius = radius / maps.hdf5_constants.px_to_micron
                threshold = 2.25 / (np.pi * native_radius ** 2)
                resistant = threshold * np.array([[0.95, 1.0, 1.05]])
                field = (2 * resistant, resistant, np.ones((1, 3), dtype=bool))
                source = maps._hdf5_interval_source(
                    field, field, 1.0, kernel_radius_um=radius,
                    min_resistant_numerator=numerator,
                )
                np.testing.assert_array_equal(np.isfinite(source[2]), [[False, True, True]])

    def test_maps_and_trajectories_match_master_pixel_definitions(self):
        for radius in (120., 160.):
            threshold = 2.25/(np.pi*(radius/constants.px_to_micron)**2)
            r0 = threshold*np.array([[1., 2., 3., 4., 5., 6.]])
            r1 = r0*np.array([[1., 3., 10., 0., 2., 4.]])
            t0 = r0*4
            t1 = r1*5
            t1[0, -1] = np.nan
            valid0 = np.ones(r0.shape, bool)
            valid1 = valid0.copy()
            valid1[0, -2] = False
            support = constants.resistant_support(t0,r0,t1,r1,valid0,valid1,threshold=threshold)
            source = maps._hdf5_interval_source((t0,r0,valid0),(t1,r1,valid1),2.,
                kernel_radius_um=radius,min_resistant_numerator=2.25)
            panel = maps.panel_data((*source[:4], None, source[4]), relative=False)
            expected = constants.resistant_change_rate(r0,r1,2.,mode="absolute")/constants.px_to_micron**2
            np.testing.assert_allclose(panel[1][support],expected[support])
            np.testing.assert_array_equal(np.isfinite(panel[1]),support)
            rim = maps.panel_support_mask({"data":panel,"support_mask":np.isfinite(source[2])},
                threshold=2.25/(np.pi*radius**2))
            np.testing.assert_array_equal(rim,support)
            # Growth above the colour scale must remain data, not be discarded.
            self.assertGreater(panel[1][0,2],2e-4)
            self.assertEqual(panel[1][0,0],0.)
            row = trajectories._summary_row(start_day=1.,end_day=3.,total_t0=t0,
                resistant_t0=r0,total_t1=t1,resistant_t1=r1,valid_t0=valid0,
                valid_t1=valid1,support_threshold=threshold,density_scale=1000/constants.px_to_micron**2,
                source="experiment")
            self.assertEqual(row["n_pixels"],int(support.sum()))
            self.assertAlmostEqual(row[trajectories.ABSOLUTE_CHANGE_COLUMN],1000*np.mean(expected[support]))
            self.assertAlmostEqual(row["Mean_Total_Density"],np.mean(t0[support])*1000/constants.px_to_micron**2)
            self.assertAlmostEqual(row[trajectories.ABSOLUTE_CHANGE_STD_COLUMN],1000*np.std(expected[support],ddof=1))

    def test_plot_defaults_follow_shared_master_profile(self):
        with patch.object(sys,"argv",["plot"]):
            args = master.parse_args()
        # Omitting the radius now requests both kernels (the established CLI
        # behaviour); an explicit radius still selects one figure.
        self.assertIsNone(args.kernel_radius_um)
        with patch.object(master, 'plot_radius') as plot, patch.object(master, 'parse_args', return_value=args):
            master.main()
        self.assertEqual([call.args[0].kernel_radius_um for call in plot.call_args_list],
                         list(constants.MASTER_KERNEL_RADII_UM))
        self.assertEqual(args.density_coordinate,"first_endpoint")
        self.assertEqual(args.growth_modes,("relative",))
        self.assertEqual(args.periods,("all_timepoints",))
        self.assertEqual(trajectories.PLOT_END_DAY,40.0)
        self.assertEqual(trajectories.RUNNING_AVERAGE_WINDOW,4)
        self.assertEqual(trajectories.TRAJECTORY_GROWTH_MODE,"absolute")
        self.assertEqual(maps.MAP_GROWTH_MODE,"absolute")

    def test_original_centered_four_point_running_average(self):
        average, error = trajectories._running_mean_and_propagated_std(
            pd.Series([1.,2.,3.,4.,5.]), pd.Series([2.,2.,2.,2.,2.]))
        np.testing.assert_allclose(average,[1.5,2.,2.5,3.5,4.])
        np.testing.assert_allclose(error,[2/np.sqrt(2),2/np.sqrt(3),1.,1.,2/np.sqrt(3)])
        days = [1.,20.,39.,42.,45.]
        self.assertEqual(trajectories._through_end_day(days,lambda day:day),[1.,20.,39.,42.])

    def test_median_trajectory_cache_is_rejected(self):
        path = self.root/"median.tsv"
        frame = pd.DataFrame({"day":[1.]})
        trajectories._load_or_build(path,lambda:frame.copy(),force=False,kernel_radius_um=160.)
        cached = pd.read_csv(path,sep="\t")
        cached["summary_statistic"] = "median"
        cached.to_csv(path,sep="\t",index=False)
        builder = Mock(return_value=frame.copy())
        result = trajectories._load_or_build(path,builder,force=False,kernel_radius_um=160.)
        builder.assert_called_once()
        self.assertEqual(result["summary_statistic"].iloc[0],"mean")
        self.assertEqual(result["temporal_smoothing_window"].iloc[0],4)

    def test_map_main_only_exports_shared_hdf5_panels(self):
        dataset = (np.ones((2,2)),np.ones((2,2)),None,None,None)
        with patch.object(maps,"OUT",self.root), patch.object(maps,"hdf5_interval_sources",return_value=[]) as fields, patch.object(
            maps,"hdf5_absolute_comparison_panels",return_value=([{}],[dataset])) as panels, patch.object(
            maps,"draw_at50_two_timepoint_comparison") as draw, patch.object(
            maps,"load_comparison_cache",side_effect=AssertionError("Legacy TIFF path was used")):
            maps.main()
        self.assertEqual(fields.call_count,2)
        self.assertEqual(draw.call_count,4)
        self.assertEqual({c.kwargs["kernel_radius_um"] for c in fields.call_args_list},{120.,160.})
        self.assertTrue(all(not c.kwargs["relative"] for c in panels.call_args_list))
        self.assertTrue(all(not c.kwargs["relative_change"] for c in draw.call_args_list))
        self.assertTrue(all("absolute_growth" in c.args[1].name for c in draw.call_args_list))

    def test_relative_trajectory_cache_is_rejected(self):
        path = self.root/"relative.tsv"
        frame = pd.DataFrame({"day":[1.]})
        trajectories._load_or_build(path,lambda:frame.copy(),force=False,kernel_radius_um=160.)
        cached = pd.read_csv(path,sep="\t")
        cached["growth_mode"] = "relative"
        cached.to_csv(path,sep="\t",index=False)
        builder = Mock(return_value=frame.copy())
        result = trajectories._load_or_build(path,builder,force=False,kernel_radius_um=160.)
        builder.assert_called_once()
        self.assertEqual(result["growth_mode"].iloc[0],"absolute")

    def test_matching_kernel_absolute_trajectory_cache_is_rejected(self):
        path = self.root/"absolute.tsv"
        old = pd.DataFrame({name:[value] for name,value in kernel_settings(160.).items()})
        old["Mean_Resistant_Density_Absolute_Change"] = .01
        old.to_csv(path,sep="\t",index=False)
        builder = Mock(return_value=pd.DataFrame({"day":[1.]}))
        trajectories._load_or_build(path,builder,force=False,kernel_radius_um=160.)
        builder.assert_called_once()

    def test_raw_hdf5_fields_are_identical_between_maps_and_trajectories(self):
        simulation = maps.hdf5_sim
        experiment = maps.hdf5_exp
        exp_path = self.root/"experiment.h5"
        positions = np.column_stack((np.linspace(0.,10.,20),np.linspace(1.,11.,20)))
        with h5py.File(exp_path,"w") as handle:
            for day in (1,3):
                for kind in ("sus","res"):
                    handle.create_dataset(f"{kind}/day_{day}/block0_values",data=positions)
        run_dir = self.root/"run_18"
        run_dir.mkdir()
        for ts in (0,100000):
            with h5py.File(run_dir/f"simulation_{ts:09d}.h5","w") as handle:
                handle.create_dataset("data/cell_nucleus",data=positions)
                handle.create_dataset("data/cell_type",data=np.arange(len(positions))%2)
                handle.create_dataset("data/cell_flags",data=np.ones(len(positions),dtype=int))
        axis = np.linspace(-10.,10.,9)
        xx,yy = np.meshgrid(axis,axis)
        with patch.object(maps,"_hdf5_experimental_hdf5_path",return_value=exp_path), patch.object(
            experiment,"_sample_and_rule",return_value=("D5",...)), patch.object(
            simulation,"protocol_outputs_dir",return_value=str(run_dir)), patch.object(
            simulation,"GRID_PTS",np.column_stack((xx.ravel(),yy.ravel()))), patch.object(
            simulation,"N_GRID",9):
            for radius in (120.,160.):
                exp_fields,_ = maps._hdf5_experimental_fields([1,3],kernel_radius_um=radius,min_neighbors=14)
                with h5py.File(exp_path,"r") as handle:
                    points,shape = trajectories._experimental_fixed_grid(handle,kernel_radius_um=radius)
                    for day in (1,3):
                        total,_,resistant,valid = trajectories._experimental_local_densities(
                            handle,f"day_{day}",points,shape,kernel_radius_um=radius,min_neighbors=14)
                        for actual,expected in zip(exp_fields[day],(total,resistant,valid)):
                            np.testing.assert_array_equal(actual,expected)
                offset = simulation.escape_core.sim.DEFAULT_DAY_OFFSET
                sim_fields,_ = maps._hdf5_simulation_fields([offset,offset+1],kernel_radius_um=radius,min_neighbors=14)
                for ts in (0,100000):
                    total,_,resistant,valid = trajectories._simulation_local_densities(
                        str(run_dir),ts,kernel_radius_um=radius,min_neighbors=14)
                    for actual,expected in zip(sim_fields[offset+ts/100000],(total,resistant,valid)):
                        np.testing.assert_array_equal(actual,expected)

    def test_cell_size_refuses_unverified_pressure_tables(self):
        from figures.Figure_3 import b_plot_cell_size_vs_density as cells
        for protocol in cells.PROTOCOLS:
            with h5py.File(self.root/f"{protocol}.h5","w") as handle:
                handle.create_group("run_18/cells")
        with self.assertRaisesRegex(ValueError,"missing parameter metadata"):
            cells.load_pooled_data(self.root)
        for protocol in cells.PROTOCOLS:
            with h5py.File(self.root/f"{protocol}.h5","r+") as handle:
                handle.attrs.update(kernel_settings(constants.MASTER_KERNEL_RADIUS_UM))
                handle.attrs["density_method"] = "epanechnikov"
                handle.move("run_18","run_0")
                group = handle["run_0/cells"].create_group("window")
                group.create_dataset("cell_type",data=[0,1])
                group.create_dataset("rho_tot",data=[.1,.1])
                group.create_dataset("delta_d",data=[.1,.2])
        data = cells.load_pooled_data(self.root)
        self.assertEqual(len(data["rho_res"]),3)
        self.assertEqual(cells.MIN_BIN_COUNT,100)

    def test_old_map_cache_is_rejected(self):
        path = self.root / "old_maps.npz"
        np.savez(path, cache_version=maps.HDF5_CACHE_VERSION,
                 analysis_signature=[160.0, 6, 2.0])
        self.assertIsNone(maps._load_hdf5_cache(
            cache_file=path, cache_version=maps.HDF5_CACHE_VERSION,
            kernel_radius_um=160.0,
            min_neighbors=maps.HDF5_BASE_MIN_NEIGHBORS,
            min_resistant_numerator=maps.HDF5_BASE_MIN_RESISTANT_NUMERATOR,
        ))

    def test_trajectory_cache_rebuild_then_reuse(self):
        path = self.root / "trajectory.tsv"
        legacy = pd.DataFrame({"Mean_Total_Density": [1.0],
                               "kernel_radius_um": [160.0],
                               "kernel_min_neighbors": [6],
                               "minimum_resistant_numerator": [2.0]})
        legacy.to_csv(path, sep="\t", index=False)
        builder = Mock(return_value=pd.DataFrame({"Mean_Total_Density": [2.0]}))
        result = trajectories._load_or_build(
            path, builder, force=False, kernel_radius_um=160.0,
        )
        builder.assert_called_once()
        self.assertEqual(result["Mean_Total_Density"].iloc[0], 2.0)
        unused_builder = Mock(side_effect=AssertionError("Valid cache was not reused"))
        trajectories._load_or_build(path, unused_builder, force=False, kernel_radius_um=160.0)
        unused_builder.assert_not_called()

    def test_diagnostic_cache_requires_matching_parameters(self):
        path = self.diagnostic_file(diagnostics.prolif_radius_um)
        self.assertTrue(diagnostics._is_complete_output(path))
        for name, wrong in (("kernel_min_neighbors", 6), ("minimum_resistant_numerator", 2.0)):
            with h5py.File(path, "r+") as handle:
                handle.attrs.update(kernel_settings(diagnostics.prolif_radius_um))
                handle.attrs[name] = wrong
            self.assertFalse(diagnostics._is_complete_output(path))
        with h5py.File(path, "r+") as handle:
            del handle.attrs["kernel_min_neighbors"]
        self.assertFalse(diagnostics._is_complete_output(path))

    def test_master_plot_accepts_both_radii_and_rejects_old_settings(self):
        for radius in (120.0, 160.0):
            with self.subTest(radius=radius):
                path = self.diagnostic_file(radius)
                curves = master.load_first_endpoint_run_curves(
                    path, "simulation", "all_timepoints", "relative", radius,
                )
                self.assertEqual(set(curves), set(master.STRATEGIES))
                with h5py.File(path, "r+") as handle:
                    handle.attrs["kernel_min_neighbors"] = 6
                with self.assertRaisesRegex(ValueError, "Recompute"):
                    master.load_first_endpoint_run_curves(
                        path, "simulation", "all_timepoints", "relative", radius,
                    )

    def test_legacy_exception_requires_exact_audited_bytes(self):
        path = self.diagnostic_file(160)
        with h5py.File(path, "r+") as handle:
            del handle.attrs["kernel_min_neighbors"]
            del handle.attrs["minimum_resistant_numerator"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(constants, "VERIFIED_LEGACY_INPUTS", {160.0: {digest}}):
            with h5py.File(path, "r") as handle:
                constants.validate_kernel_settings(handle.attrs, 160.0, source=path)
            with h5py.File(path, "r+") as handle:
                handle["simulation/CT/all_timepoints/t0_density/median_rel"][0, 0] = 99
            with h5py.File(path, "r") as handle:
                with self.assertRaisesRegex(ValueError, "not an exact audited"):
                    constants.validate_kernel_settings(handle.attrs, 160.0, source=path)

    def test_master_computation_excludes_artificial_runs(self):
        for strategy in diagnostics.STRATEGIES:
            for name in ("run_0", "run_9", "run_10", "run_18"):
                (self.root / strategy / name).mkdir(parents=True)
        simulation = diagnostics.simulation
        with patch.object(simulation, "discover_simulation_protocols", return_value=list(diagnostics.STRATEGIES)), patch.object(
            simulation, "protocol_outputs_dir", side_effect=lambda strategy, run: str(self.root / strategy / run)
        ), patch.object(simulation, "_selected_timesteps", return_value=[0, 100000]), patch.object(
            simulation.escape_core, "simulation_escape_days", return_value={}
        ), patch.object(diagnostics, "N_JOBS", 1), patch.object(
            diagnostics, "_simulation_run", return_value={}
        ) as compute, patch.object(diagnostics, "_write_strategy") as write:
            diagnostics._compute_simulations(object())
        self.assertEqual(compute.call_count, 6)
        self.assertEqual({call.args[1] for call in compute.call_args_list}, {"run_10", "run_18"})
        self.assertEqual(write.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
