"""
TL;DR
-----
This script is the lightweight entrypoint for the microscopy-to-analysis
pipeline defined in `analysis_pipeline_2D_corr.py`. It runs the configured CZI
analysis workflow and optionally iterates over all `.czi` files in a plate.

Detailed
--------
Inputs:
- Raw `.czi` microscopy files and the path/configuration variables imported
  from `analysis_pipeline_2D_corr.py`.

Outputs:
- Whatever the imported pipeline is configured to write: segmented TIFFs,
  HDF5 point clouds, result text files, and optional timestacks.

How it works:
- Imports the full configuration and helper stack from
  `analysis_pipeline_2D_corr.py`.
- Defines a `main()` wrapper that opens a CZI file, extracts metadata, and
  dispatches into timepoint or timelapse analysis.
- When run as a script, it can loop over every `.czi` file in the configured
  plate directory.

Pipeline role:
- Thin executable wrapper around the main raw-image analysis pipeline.
"""

from analysis.basic.microscopy.analysis_pipeline_2D_corr import *
import time
#
experiment_directory = experiment_path

loading_tifs = os.path.join(experiment_directory, 'Tifs')
saving_tif_timestack = os.path.join(experiment_directory, 'timestacks')

def main():
    '''This is the main function of the program. It is used to run the program. It can be used to analyze a czi file or to load hdf5 files and analyze them.'''

    # Open czi file = Initiating the class
    if open_czi_from_path == True:
        opened_czi = open_czi_list(czi_file=czi_file)
    else:
        opened_czi = open_czi()

    '''
    #get a list of the czi file paths in the directory
    czi_list = []
    for file in os.listdir(experiment_path):
        if file.endswith(".czi"):
            czi_list.append(file)
    print(czi_list)

    for czi_file in czi_list:
        # Open czi file = Initiating the class
        opened_czi = open_czi.open_file_from_path(file_path=czi_file)
    '''
    # Extract metadata from czi file
    metadata = extract_metadata(opened_czi)
    czi_name = os.path.basename(metadata['file_path'])
    timepoint = re.findall(r'_[dt](\d\d)_', czi_name)[0]

    # Analyze czi file and save the results to hdf5 file
    if analyze_czi == True:

        print('The czi file contains the following scenes:', metadata['scene_names_list'])

        # Iterate over all scenes
        for scene in metadata['scene_names_list']:
            if not 'A1' in scene:
                print('    Scene', scene, 'skipped because it does not contain A1')
                continue
            start = time.perf_counter()
            print('Scene', scene, 'started')

            opened_czi.set_scene(scene)  # set scene
            dask = opened_czi.initiate_dask()  # initiate dask array from this scene

            #Iterate over all timepoints if there is a timelapse
            if dask.shape[0] != 1:
                print('    Scene', scene, 'is a timelapse')

                timelapse_analysis(dask, metadata['file_path'], time_list, time_unit, metadata['three_d'], metadata['x_scale'], metadata['y_scale'], metadata['z_scale'], npy_done,
                                   hdf5_savepath, scene, metadata['scene_names_list'], diameter=diameter, minsize=minsize, flow_Threshold=flow_Threshold, cellprob_threshold=cellprob_threshold, channels=channels)

            else:
                print('    Scene', scene, 'only has one timepoint')
                timepoint_analysis(dask, metadata['file_path'], metadata['three_d'], metadata['x_scale'], metadata['y_scale'], metadata['z_scale'], npy_done, hdf5_savepath,
                                   scene, metadata['scene_names_list'], diameter=diameter, minsize=minsize, flow_Threshold=flow_Threshold, cellprob_threshold=cellprob_threshold, timepoint=timepoint, channels=channels)

            print('    Scene', scene, 'finished in', time.perf_counter()-start, 'seconds')
    if load_hdffive == True:

        is_timelapse = True if dask.shape[0] != 1 else False

        #Loading the given hdf5 files and plotting the results
        results = analyze_files_and_plot(experiment_path, experiment_name, time_list, time_unit, metadata['three_d'], is_timelapse)

    if save_txt == True:

        #Saving the results in a txt file
        save_results_today_in_txt(timepoint=timepoint)
        save_results_in_txt()

    if save_tif == True:
        # Save the czi file as tif
        opened_czi.save_czi_as_tif(experiment_directory)

if __name__ == '__main__':
    if open_czi_from_path == True:

        # filelist = [r'\\syn03.frontend.storage.int.mpl.mpg.de\guck_division2\Members\Elias\MastersThesis\2024_G418_a2a4_Pulse\Plate_2\czi_NPY\20240423_d00_G418_a2a4_Plate2_MIPs.czi',
        #             r'\\syn03.frontend.storage.int.mpl.mpg.de\guck_division2\Members\Elias\MastersThesis\2024_G418_a2a4_Pulse\Plate_2\czi_NPY\20240424_d01_G418_a2a4_Plate2_MIPs.czi']
        # for file in filelist:
        #     czi_file = os.path.join(experiment_path, 'czi_NPY', file)
        #     main()


        for file in os.listdir(os.path.join(experiment_directory, 'czi_NPY')):
            if file.endswith(".czi"):
                czi_file = os.path.join(experiment_path, 'czi_NPY', file)
                main()



    else:
        main()

    if save_timestacks == True:
        # merging stacks to time stacks
        if not os.path.isdir(saving_tif_timestack):
            os.mkdir(saving_tif_timestack)
        stack_tifs_to_timestack(saving_path=saving_tif_timestack, loading_path=loading_tifs)
