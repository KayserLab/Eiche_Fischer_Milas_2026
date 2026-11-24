"""
TL;DR
-----
This is the main raw microscopy analysis pipeline. It reads Zeiss `.czi`
imaging data, segments susceptible and resistant cells with CellPose-derived
models, converts masks into cell-center point clouds, and writes the HDF5/TIFF
and result artifacts that later scripts use.

Detailed
--------
Inputs:
- Raw `.czi` microscopy files.
- User-configured segmentation and correction settings in the top-level
  parameter block (channels, thresholds, model names, output toggles, etc.).

Outputs:
- HDF5 point clouds for susceptible/resistant cells by day/timepoint.
- Overexposed-cell count HDF5 files.
- Exported TIFF imagery and optional timestacks.
- Text summaries in `Results` / `Results_today`.

How it works:
- Opens raw CZI scenes, extracts imaging metadata, and iterates over scenes
  and timepoints.
- Runs segmentation or loads existing CellPose outputs, then converts masks
  into 2D/3D cell-center coordinates.
- Applies optional background subtraction, bleed-through correction, and
- overexposure correction.
- Saves point clouds and summary outputs in a form consumed by downstream
  density, colony-mask, clone-tracking, and figure scripts.

Pipeline role:
- Earliest computational stage of the paper pipeline, converting raw imaging
  into the basic quantitative cell-position data used everywhere downstream.
"""

import numpy as np
import scipy
import pandas as pd
from cellpose import models, io
import os
import plotly.graph_objects as go
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from aicsimageio import AICSImage as aio
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import h5py
from PIL import Image
import tifffile as tiff
from scipy.ndimage import center_of_mass
import skimage
import time
######################################################### User input ####################################################################

'''Here you have to give input to the program. You can choose between the following options:'''

analyze_czi    = True                        # Do you want to open and analyze a czi file?
npy_done       = False                       # Do you want to load already existing NPY (Segmented CellPose) files and analyze them?
plot_scatter   = False                       # Do you want to plot the scatterplots of segmented cells after analyzing?
load_hdffive   = False                       # Do you want to load hdf5 files of segmented cells and analyze them?
plot_linegraph = False                       # Do you want to plot the graphs of cell numbers over time and scenes?
plot_interactiveplot_from_hdf5 = False       # Do you want to plot the interactive scatter plot from hdf5 files?
save_txt = True                              # Do you want to save the results of the analysis in txt files?
save_tif = True                             # Do you want to save the czi file as tif files?
save_timestacks = False
open_czi_from_path = False                   # Do you want to open the czi file from a given path? With that you can alanyze all czi files in a given directory

background_subtraction = [False, False]   # Do you want to subtract the background from the images in a specific channel?
background_value = [500, 800]                  # Value to subtract from the images
high_low_imaging = False                    # Is imaging with high AND low intensities?
bleed_through = False            # Do you want to correct the bleed through of the images?
channel_which_bleeds_through = 2
change_masks = False           # Do you want to change the masks of the images to the other channel?


overexposure_correction = True
# overexposure_threshold = 16382            #those values were used to get the cellcount of overexposed images in the 1st analysis for the AT experiment 2025_03_AT
# max_density_overexposure_res = 0.0016     #those values were used to get the cellcount of overexposed images in the 1st analysis for the AT experiment 2025_03_AT
# max_density_overexposure_sus = 0.0016     #those values were used to get the cellcount of overexposed images in the 1st analysis for the AT experiment 2025_03_AT

overexposure_threshold = 12000              #those values were used to get the cellcount of overexposed images in the 2nd analysis for the AT experiment 2025_03_AT_corrected and 2025_02_Pulse
max_density_overexposure_res = 0.00286      #those values were used to get the cellcount of overexposed images in the 2nd analysis for the AT experiment 2025_03_AT_corrected and 2025_02_Pulse
max_density_overexposure_sus = 0.00286      #those values were used to get the cellcount of overexposed images in the 2nd analysis for the AT experiment 2025_03_AT_corrected and 2025_02_Pulse


'''Here you have to give the program the following input:'''

time_list = list(range(63))                                            # List of timepoints to analyze (only needed if analyze_czi = True)
time_unit = 'd'                                                       # Time unit for the time_list above

experiment_path = r'D:\2026_01_Density\Plate_2'                    # Folder to load hdf5 or NPY file(s) from (only needed if load_hdffive = True or npy_done = True) and also where the czi files are stored
#experiment_path = r'D:\Test\Plate_4'                    # Folder to load hdf5 or NPY file(s) from (only needed if load_hdffive = True or npy_done = True) and also where the czi files are stored
experiment_name = "G418_a2a4"                                    # Name of the experiment to load hdf5 files from

hdf5_savepath = os.path.join(experiment_path, 'hdf5', experiment_name + ".h5")      # Path to save hdf5 file(s) in, also name of hdf5 file, timeunit/timepoint excluded
hdf5_loading_path = os.path.dirname(hdf5_savepath)                                  # Path to load hdf5 file(s) from
results_txt_today_path = os.path.join(experiment_path, 'Results_today')             # Path to save txt file of today's results in
results_txt_path = os.path.join(experiment_path, 'Results')


#create hdf5 saving directory if not existing
if not os.path.exists(os.path.join(experiment_path, 'hdf5')):
    os.makedirs(os.path.join(experiment_path, 'hdf5'))

'''Here you specify the cellpose segmentation parameters'''
# Model type for each channel.                  Example: ['3Dslice_red', '3Dslice_green']
# Can also be the full path to the model file   Example: 'C:/Users/labadmin-kayser/Desktop/Max_3D/3Dslice_red'
cellpose_models = ['a4_cyto2_PhD_1000_new', 'a2_cyto3_Epi_1000_PhD']  #also change in timepoint_analysis!!!
#channels = [[0, 0],[1, 0]]                      # Channels to segment for each model. Example: [[1, 0], [2, 0]]
channels = [[3, 0], [2, 0]]                      # Channels to segment for each model. Example: [[1, 0], [2, 0]]
diameter = [None, None]
minsize = [250, 250]                            #also change in timepoint_analysis!!!
flow_Threshold = [0, 0]                         #also change in timepoint_analysis!!!
cellprob_threshold = [-1, -1]                   #also change in timepoint_analysis!!!


class open_czi:
    """
        Opens a czi file and returns the aicsimageio object.

        :param file_path: Path to the czi file
        :type file_path: str | None

        :param img: aicsimageio object
        :type img: aicsimageio.AICSImage | None

        :param dims: Dimensions of the image
        :type dims: dict

        :param scenes: Scenes of the image
        :type scenes: list

        :param time_points: Time points of the image
        :type time_points: int

        :param z_stacks: Z stacks of the image
        :type z_stacks: int

        :param channels: Channels of the image
        :type channels: int

        :param x_pixels: X pixels of the image
        :type x_pixels: int

        :param y_pixels: Y pixels of the image
        :type y_pixels: int

        :param metadata: Metadata of the image
        :type metadata: dict

        :return: aicsimageio object
        :rtype: aicsimageio.AICSImage

        """
    def __init__(self):

        self.file_path = None
        self.img = None
        self.open_file()
        self.dims = self.img.dims
        self.scenes = self.img.scenes
        self.time_points = self.dims['T'][0]
        self.z_stacks = self.dims['Z'][0]
        self.channels = self.dims['C'][0]
        self.x_pixels = self.dims['X'][0]
        self.y_pixels = self.dims['Y'][0]
        self.metadata = self.img.metadata

    def open_file(self):

        """

        Opens a file dialog and sets the path to the selected file.
        With this, the file is accessible via self.xyz as an aicsimageio object.
        Here the file is not yet fully loaded, but in a lazy-read style

        """

        root = Tk()
        root.withdraw()  # Hide the root window
        root.attributes('-topmost', True)

        # Open file dialog and return the selected file path
        file_path = askopenfilename()

        # Stop when file_path is empty or does not end with .czi
        if file_path == '' or file_path[-4:] != '.czi':
            raise ValueError("No file or wrong file format selected.")
        else:
            self.file_path = file_path

        self.img = aio(file_path)

    def open_file_from_path(self, file_path):

        """
        Sets the path to the selected file path and uses this file for further analysis.
        With this, the file is accessible via self.xyz as an aicsimageio object.
        Here the file is not yet fully loaded, but in a lazy-read style

        :param file_path: Path to the czi file
        :return:
        """

        root = Tk()
        root.withdraw()  # Hide the root window
        root.attributes('-topmost', True)

        # Stop when file_path is empty or does not end with .czi
        if file_path == '' or file_path[-4:] != '.czi':
            raise ValueError("No file or wrong file format selected.")
        else:
            self.file_path = file_path

        self.img = aio(file_path)

    def save_czi_as_tif(self, experiment_path=experiment_path):

        """
        Saves the scenes of the czi file as seperate tif files.

        :param file_path: Path to the tif file
        :type file_path: str
        """
        #generate tif file directory in experiment directory and create it if not existing
        tif_directory = os.path.join(experiment_path, 'Tifs')
        if not os.path.exists(tif_directory):
            os.makedirs(tif_directory)
        print(self.scenes)
        for scene in self.scenes:
            scene_saving = scene
            pattern = r"^([A-Z]\d+)(_.+?)(?:-([A-Z]\d+))?$"
            m = re.match(pattern, scene)
            if m:
                first_code, middle, second_code = m.groups()
                if second_code:  # second exists
                    if first_code != second_code:
                        scene_saving = f"{first_code}{middle}-{first_code}"
                else:  # missing second code
                    scene_saving = f"{first_code}{middle}-{first_code}"
            print(scene)
            print(scene_saving)
            saving_path = os.path.join(tif_directory, os.path.basename(os.path.splitext(self.file_path)[0]) + '_' + scene_saving + '.tif')
            self.img.set_scene(scene)
            dask = self.initiate_dask()
            #rearange dask to fitting shape for saving (from TCZYX to TZCYX)
            dask = dask.transpose(0, 2, 1, 3, 4)

            #change channel order (dimension of dask with three dimenstions) from [0 - bf, 1 - green, 2 - red] to [2, 1, 0]
            if self.channels == 6:
                dask_transposed = dask[:, :, [1, 3, 2, 4, 0, 5], :, :]
            elif self.channels == 5:
                dask_transposed = dask[:, :, [1, 3, 2, 4, 0], :, :]
            elif self.channels == 4:
                dask_transposed = dask[:, :, [1, 2, 0, 3], :, :]
            elif self.channels == 3:
                dask_transposed = dask[:, :, [1, 2, 0], :, :]
            print(dask_transposed.shape)

            #save dask array as tif file
            tiff.imwrite(saving_path, dask_transposed, imagej=True, metadata={'axes': 'TZCYX'}, compression='lzw')

    def set_scene(self, scene):

        '''

        This sets the scene for the aicsimageio object. The scene is the well of the plate. This is always done before
        extracting information out of the (not fully loaded) aicsimageio object.

        '''

        if scene in self.scenes:
            self.img.set_scene(scene)
        else:
            raise ValueError(f"No valid scene selected.Valid options are: {self.scenes}")

    def initiate_xarray(self):

        '''

        This initiates the xarray_dask_data of the aicsimageio object. This is the actual data of the image in an xarray.
        The xarray is not so easy to work with, initiating a das array is easier, as the dask array is a standard numpy array.

        '''

        return self.img.xarray_dask_data

    def initiate_dask(self):

        '''

        Here the dask array is initiated. This is the actual data of the image in a dask array, which is almost the same as a numpy array.
        First you always have to set the scene, then you can initiate the dask array. It returns a 5D TCZYX (time, channel, z-stack, y, x) dask array.

        '''

        return self.img.dask_data # returns 5D TCZYX dask array

    def show_image(self, scene, time_point, channel, z_stack):

        '''

        This shows the image of the aicsimageio object of a given timepoint, channel and z stack.

        :param scene: Scene of the image
        :type scene: str

        :param time_point: Time point of the image
        :type time_point: int

        :param channel: Channel of the image
        :type channel: int

        :param z_stack: Z stack of the image
        :type z_stack: int

        :return: Image of the aicsimageio object
        :rtype: matplotlib.pyplot.imshow

        '''

        self.set_scene(scene)
        dask = self.initiate_dask()
        data = dask[time_point, channel, z_stack, :, :]
        plt.imshow(data)
        plt.show()

    def get_scale(self):

        '''

        This returns the pixel size of the image in µm.

        '''

        return self.img.physical_pixel_sizes[2]

class open_czi_list:
    """
        Opens a czi file and returns the aicsimageio object.

        :param file_path: Path to the czi file
        :type file_path: str | None

        :param img: aicsimageio object
        :type img: aicsimageio.AICSImage | None

        :param dims: Dimensions of the image
        :type dims: dict

        :param scenes: Scenes of the image
        :type scenes: list

        :param time_points: Time points of the image
        :type time_points: int

        :param z_stacks: Z stacks of the image
        :type z_stacks: int

        :param channels: Channels of the image
        :type channels: int

        :param x_pixels: X pixels of the image
        :type x_pixels: int

        :param y_pixels: Y pixels of the image
        :type y_pixels: int

        :param metadata: Metadata of the image
        :type metadata: dict

        :return: aicsimageio object
        :rtype: aicsimageio.AICSImage

        """
    def __init__(self, czi_file):

        self.file_path = None
        self.img = None
        self.open_file_from_path(czi_file=czi_file)
        self.dims = self.img.dims
        self.scenes = self.img.scenes
        self.time_points = self.dims['T'][0]
        self.z_stacks = self.dims['Z'][0]
        self.channels = self.dims['C'][0]
        self.x_pixels = self.dims['X'][0]
        self.y_pixels = self.dims['Y'][0]
        self.metadata = self.img.metadata

    def open_file(self):

        """

        Opens a file dialog and sets the path to the selected file.
        With this, the file is accessible via self.xyz as an aicsimageio object.
        Here the file is not yet fully loaded, but in a lazy-read style

        """

        root = Tk()
        root.withdraw()  # Hide the root window
        root.attributes('-topmost', True)

        # Open file dialog and return the selected file path
        file_path = askopenfilename()

        # Stop when file_path is empty or does not end with .czi
        if file_path == '' or file_path[-4:] != '.czi':
            raise ValueError("No file or wrong file format selected.")
        else:
            self.file_path = file_path

        self.img = aio(file_path)

    def open_file_from_path(self, czi_file):

        """
        Sets the path to the selected file path and uses this file for further analysis.
        With this, the file is accessible via self.xyz as an aicsimageio object.
        Here the file is not yet fully loaded, but in a lazy-read style

        :param czi_file: Path to the czi file
        :return:
        """

        root = Tk()
        root.withdraw()  # Hide the root window
        root.attributes('-topmost', True)

        # Stop when file_path is empty or does not end with .czi
        if czi_file == '' or czi_file[-4:] != '.czi':
            raise ValueError("No file or wrong file format selected.")
        else:
            self.file_path = czi_file

        self.img = aio(czi_file)

    def save_czi_as_tif(self, experiment_path=experiment_path):

        """
        Saves the scenes of the czi file as seperate tif files.

        :param file_path: Path to the tif file
        :type file_path: str
        """
        #generate tif file directory in experiment directory and create it if not existing
        tif_directory = os.path.join(experiment_path, 'Tifs')
        if not os.path.exists(tif_directory):
            os.makedirs(tif_directory)
        print(self.scenes)
        for scene in self.scenes:
            scene_saving = scene
            pattern = r"^([A-Z]\d+)(_.+?)(?:-([A-Z]\d+))?$"
            m = re.match(pattern, scene)
            if m:
                first_code, middle, second_code = m.groups()
                if second_code:  # second exists
                    if first_code != second_code:
                        scene_saving = f"{first_code}{middle}-{first_code}"
                else:  # missing second code
                    scene_saving = f"{first_code}{middle}-{first_code}"
            print(scene)
            print(scene_saving)
            saving_path = os.path.join(tif_directory, os.path.basename(os.path.splitext(self.file_path)[0]) + '_' + scene_saving + '.tif')
            self.img.set_scene(scene)
            dask = self.initiate_dask()
            #rearange dask to fitting shape for saving (from TCZYX to TZCYX)
            dask = dask.transpose(0, 2, 1, 3, 4)

            #change channel order (dimension of dask with three dimenstions) from [0 - bf, 1 - green, 2 - red] to [2, 1, 0]
            if self.channels == 6:
                dask_transposed = dask[:, :, [1, 3, 2, 4, 0, 5], :, :]
            elif self.channels == 5:
                dask_transposed = dask[:, :, [1, 3, 2, 4, 0], :, :]
            elif self.channels == 4:
                dask_transposed = dask[:, :, [1, 2, 0, 3], :, :]
            elif self.channels == 3:
                dask_transposed = dask[:, :, [1, 2, 0], :, :]
            print(dask_transposed.shape)

            #save dask array as tif file
            tiff.imwrite(saving_path, dask_transposed, imagej=True, metadata={'axes': 'TZCYX'}, compression='lzw')

    def set_scene(self, scene):

        '''

        This sets the scene for the aicsimageio object. The scene is the well of the plate. This is always done before
        extracting information out of the (not fully loaded) aicsimageio object.

        '''

        if scene in self.scenes:
            self.img.set_scene(scene)
        else:
            raise ValueError(f"No valid scene selected.Valid options are: {self.scenes}")

    def initiate_xarray(self):

        '''

        This initiates the xarray_dask_data of the aicsimageio object. This is the actual data of the image in an xarray.
        The xarray is not so easy to work with, initiating a das array is easier, as the dask array is a standard numpy array.

        '''

        return self.img.xarray_dask_data

    def initiate_dask(self):

        '''

        Here the dask array is initiated. This is the actual data of the image in a dask array, which is almost the same as a numpy array.
        First you always have to set the scene, then you can initiate the dask array. It returns a 5D TCZYX (time, channel, z-stack, y, x) dask array.

        '''

        return self.img.dask_data # returns 5D TCZYX dask array

    def show_image(self, scene, time_point, channel, z_stack):

        '''

        This shows the image of the aicsimageio object of a given timepoint, channel and z stack.

        :param scene: Scene of the image
        :type scene: str

        :param time_point: Time point of the image
        :type time_point: int

        :param channel: Channel of the image
        :type channel: int

        :param z_stack: Z stack of the image
        :type z_stack: int

        :return: Image of the aicsimageio object
        :rtype: matplotlib.pyplot.imshow

        '''

        self.set_scene(scene)
        dask = self.initiate_dask()
        data = dask[time_point, channel, z_stack, :, :]
        plt.imshow(data)
        plt.show()

    def get_scale(self):

        '''

        This returns the pixel size of the image in µm.

        '''

        return self.img.physical_pixel_sizes[2]

def extract_metadata(opened_czi):
    '''

    Description: This function extracts the metadata from the opened czi file and returns it as a dictionary.

    :param opened_czi: Opened czi file
    :type opened_czi: czifile.CziFile

    :return: Dictionary of the extracted metadata ("file_path": file_path, "three_d": three_d, "x_scale": x_scale, "y_scale": y_scale, "z_scale": z_scale, "scene_names_list": scene_names_list)
    :rtype: dict

    '''

    # Extract file path
    file_path = opened_czi.file_path

    # Check if the image is 3D or 2D
    three_d = opened_czi.dims.Z != 1
    print('This is a 3D image' if three_d else 'This is a 2D image')

    # Get physical sizes of pixels
    x_scale = opened_czi.img.physical_pixel_sizes.X
    y_scale = opened_czi.img.physical_pixel_sizes.Y
    z_scale = opened_czi.img.physical_pixel_sizes.Z if opened_czi.img.physical_pixel_sizes.Z is not None else 1

    # Get scene names
    scene_names_list = opened_czi.img.scenes

    # Return the extracted metadata
    return {
        "file_path": file_path,
        "three_d": three_d,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "z_scale": z_scale,
        "scene_names_list": scene_names_list
    }

def cellpose_to_NPY(img, file_name: str, channels: list = [[1,0],[2,0]], cellprob_threshold: list = [0.0, 0.0], MinimumSize: list = minsize, stitch_threshold: list = [0.0, 0.0], flow_Threshold: list = [0.0, 0.0], Diameter: list = [None, None], do3D: bool = True, gpu: bool = True, background_subtraction = background_subtraction, background_value = background_value, cellpose_model_input=cellpose_models):

    '''

    Segments images to NPY files using CellPose.

    :param img: The input image to be segmented. Shape should be (channels, z-stack, x, y).
    :type img: 4D numpy array

    :param file_name: Name of the file to be processed. Example: 'C:/Users/labadmin-kayser/Desktop/Max_3D/3D_timelapse_multiplescenes_sceneC2_d1.czi'.
    :type file_name: str

    :param channels: Channels to segment for each model. Example: [[1, 0], [2, 0]].
    :type channels: list of list of int

    :param cellprob_threshold: Cell probability threshold (CellPose Parameter). Example: [0.0, 0.0].
    :type cellprob_threshold: list of float

    :param MinimumSize: Minimum size of cells in pixels (CellPose Parameter). Example: [100, 100].
    :type MinimumSize: list of int

    :param stitch_threshold: Stitch threshold (CellPose Parameter). Example: [0.0, 0.0].
    :type stitch_threshold: list of float

    :param flow_Threshold: Flow threshold (CellPose Parameter). Example: [0.6, 0.6].
    :type flow_Threshold: list of float

    :param Diameter: Rough diameter of cells in pixels (CellPose Parameter). None indicates auto-detection. Example: [None, None].
    :type Diameter: list of int or None

    :param do3D: If True, 3D segmentation is performed, else 2D.
    :type do3D: bool

    :param gpu: If True, GPU is used for processing, else CPU is used.
    :type gpu: bool

    :return: The function returns masks of the segmented cells for each channel. These masks can be used to calculate the centers of the cells later.
    :rtype: list of numpy arrays

    Notes:
    -----
    Outputs are NPY files for each channel. The function assumes that all necessary CellPose parameters are provided correctly for accurate segmentation.

    '''

    # If img is 2D, eliminate the first dimension (z_stacks) for compatibility with CellPose (Correct input shape)
    if do3D != True:
        img = img[:,0,:,:]

    model_input = cellpose_model_input

    #Execute CellPose
    for index, channel in enumerate(channels):
        # if background_subtraction == true: set every pixel with values lower than 1000 to 0
        if background_subtraction[index] == True:
            img[img < background_value[index]] = 0

        # Define model
        model = models.CellposeModel(gpu=gpu, model_type=model_input[index])

        # Define file name for each channel
        file_name_per_channel = os.path.splitext(file_name)[0] + '_' + str(channel) + os.path.splitext(file_name)[1]

        # Run segmentation
        masks, flows, styles = model.eval(img, diameter=Diameter[index], channels=channel, flow_threshold=flow_Threshold[index], cellprob_threshold=cellprob_threshold[index], min_size=minsize[index], stitch_threshold=stitch_threshold[index], do_3D=do3D)

        # Save masks and flows
        io.masks_flows_to_seg(images=img, masks=masks, flows=flows, file_names=file_name_per_channel, channels=channel, diams=Diameter[index])

        if index == 0:
            mask_res = masks
            flows_res = flows
        elif index == 1:
            mask_sus = masks
            flows_sus = flows

    return mask_sus, mask_res, flows_sus, flows_res

def load_mask(path_sus, path_res):

    '''

    Loads NPY Files from CellPose as numpy array from paths and returns the masks.

    '''

    temporary_load_sus = tiff.imread(path_sus)
    temporary_load_res = tiff.imread(path_res)

    #cellpose_output_sus = temporary_load_sus[()]
    #cellpose_output_res = temporary_load_res[()]

    return temporary_load_sus, temporary_load_res

def calculate_centers_3d(mask_sus, mask_res, x_scale, y_scale, z_scale):

    '''

    Calculates centers from 3D NPY Files from CellPose and converts them to dataframes for susceptible and resistant cells.

    :param mask_sus: The mask CellPose found for the susceptible cells.
    :type mask_sus: CellPose format

    :param mask_res: The mask CellPose found for the resistant cells.
    :type mask_res: CellPose format

    :param x_scale: The pixel size in x direction in micrometers (µm). Example: 0.65
    :type x_scale: float

    :param y_scale: The pixel size in y direction in micrometers (µm). Example: 0.65
    :type y_scale: float

    :param z_scale: The pixel size in z direction in micrometers (µm). Example: 10
    :type z_scale: float

    :return: Dataframes for the centers of susceptible and resistant cells.
    :rtype: tuple of pandas.DataFrame

    '''

    slices_sus = scipy.ndimage.find_objects(mask_sus)
    centers_sus = []
    for i, si in enumerate(slices_sus):
        if si is not None:
            sr, sc, sl = si
            mask = mask_sus[sr, sc, sl]
            center = scipy.ndimage.center_of_mass(mask)
            center_rounded = list([round((center[0] + sr.start) * z_scale), round((center[1] + sc.start) * x_scale), round((center[2] + sl.start) * y_scale)])
            centers_sus.append(center_rounded)

    slices_res = scipy.ndimage.find_objects(mask_res)
    centers_res = []
    for i, si in enumerate(slices_res):
        if si is not None:
            sr, sc, sl = si
            mask = mask_res[sr, sc, sl]
            center = scipy.ndimage.center_of_mass(mask)
            center_rounded = list([round((center[0] + sr.start) * z_scale), round((center[1] + sc.start) * x_scale), round((center[2] + sl.start) * y_scale)])
            centers_res.append(center_rounded)

    #Converts list of points to pandas dataframe
    df_sus = pd.DataFrame(centers_sus, columns=['z', 'x', 'y'])
    df_res = pd.DataFrame(centers_res, columns=['z', 'x', 'y'])

    return df_sus, df_res

def calculate_centers_2d(mask_sus, mask_res, x_scale=1, y_scale=1):

    '''

    Calculates centers from 2D NPY Files from CellPose and converts them to dataframes for susceptible and resistant cells.

    :param mask_sus: The mask CellPose found for the susceptible cells.
    :type mask_sus: CellPose format

    :param mask_res: The mask CellPose found for the resistant cells.
    :type mask_res: CellPose format

    :param x_scale: The pixel size in x direction in micrometers (µm). Example: 0.65
    :type x_scale: float

    :param y_scale: The pixel size in y direction in micrometers (µm). Example: 0.65
    :type y_scale: float

    :return: Dataframes for the centers of susceptible and resistant cells.
    :rtype: tuple of pandas.DataFrame

    '''

    slices_sus = scipy.ndimage.find_objects(mask_sus)
    centers_sus = []
    for i, si in enumerate(slices_sus):
        if si is not None:
            sr, sc = si
            mask = mask_sus[sr, sc]
            center = scipy.ndimage.center_of_mass(mask)
            center_rounded = list([round((center[0] + sr.start) * x_scale), round((center[1] + sc.start) * y_scale)])
            centers_sus.append(center_rounded)

    slices_res = scipy.ndimage.find_objects(mask_res)
    centers_res = []
    for i, si in enumerate(slices_res):
        if si is not None:
            sr, sc = si
            mask = mask_res[sr, sc]
            center = scipy.ndimage.center_of_mass(mask)
            center_rounded = list([round((center[0] + sr.start) * x_scale), round((center[1] + sc.start) * y_scale)])
            centers_res.append(center_rounded)

    #Converts list of points to pandas dataframe
    df_sus = pd.DataFrame(centers_sus, columns=['y', 'x'])
    df_res = pd.DataFrame(centers_res, columns=['y', 'x'])

    return df_sus, df_res

def rescale_centers_2d(df_sus, df_res, x_scale, y_scale):
    '''

    Rescales the centers of the cells to the original image size in Pixels

    :param df_sus: pandas df with the centers of the susceptible cells

    :param df_res: pandas df with the centers of the susceptible cells

    :param x_scale: The pixel size in x direction in micrometers (µm). Example: 0.65
    :type x_scale: float

    :param y_scale: The pixel size in y direction in micrometers (µm). Example: 0.65
    :type y_scale: float

    :return: Dataframes for the centers of susceptible and resistant cells.
    :rtype: tuple of pandas.DataFrame

    '''
    #Rescale the centers to the original image size in Pixels
    df_sus_re = df_sus.copy()
    df_res_re = df_res.copy()
    df_sus_re['x'] = df_sus['x'] / x_scale
    df_sus_re['y'] = df_sus['y'] / y_scale
    df_res_re['x'] = df_res['x'] / x_scale
    df_res_re['y'] = df_res['y'] / y_scale

    return df_sus_re, df_res_re

def pointcloud_save(df_sus: pd.DataFrame, df_res: pd.DataFrame, filename: str, scene = 0, scene_names_list = ('Well'), time: str = None, unit: str = None, intensity: str = None):

    '''

    The function saves the dataframes to hdf5 files. The dataframes are saved in the hdf5 file under the keys 'sus' and 'res'.

    :param df_sus: Dataframe of the x, y and z positions of susceptible cells.
    :type df_sus: pandas.DataFrame

    :param df_res: Dataframe of the x, y and z positions resistant cells.
    :type df_res: pandas.DataFrame

    :param filename: Name to use to save the hdf5 file.
    :type filename: str

    :param time: Time point of the image.
    :type time: int

    :param unit: Unit of the time point. Example: 'day'
    :type unit: str

    :param scene: Name of the scene of the image. Example: 'TR12-B2'
    :type scene: str

    :param scene_names_list: List of names of scenes of the image
    :type scene_names_list: list of str

    :return: None
    :rtype: None

    '''
    if intensity is None:
        filename_hdf5 = os.path.splitext(filename)[0] + '_' + str(scene) + os.path.splitext(filename)[1]  # Define file name for each channel
    else:
        filename_hdf5 = os.path.splitext(filename)[0] + '_' + str(scene) + '_' + str(intensity) + os.path.splitext(filename)[1]  # Define file name for each channel

    df_sus.to_hdf(filename_hdf5, key=f'sus/{unit}_{time}')
    df_res.to_hdf(filename_hdf5, key=f'res/{unit}_{time}')

def overexposed_cellcount_save(df_sus: pd.DataFrame, df_res: pd.DataFrame, filename: str, scene = 0, scene_names_list = ('Well'), time: str = None, unit: str = None, intensity: str = None):

    '''

    The function saves the dataframes to hdf5 files. The dataframes are saved in the hdf5 file under the keys 'sus' and 'res'.

    :param df_sus: Dataframe of the x, y and z positions of susceptible cells.
    :type df_sus: pandas.DataFrame

    :param df_res: Dataframe of the x, y and z positions resistant cells.
    :type df_res: pandas.DataFrame

    :param filename: Name to use to save the hdf5 file.
    :type filename: str

    :param time: Time point of the image.
    :type time: int

    :param unit: Unit of the time point. Example: 'day'
    :type unit: str

    :param scene: Name of the scene of the image. Example: 'TR12-B2'
    :type scene: str

    :param scene_names_list: List of names of scenes of the image
    :type scene_names_list: list of str

    :return: None
    :rtype: None

    '''
    if intensity is None:
        filename_hdf5 = os.path.splitext(filename)[0] + '_' + str(scene) + '_overexposed' + os.path.splitext(filename)[1]  # Define file name for each channel
    else:
        filename_hdf5 = os.path.splitext(filename)[0] + '_' + str(scene) + '_overexposed' + str(intensity) + os.path.splitext(filename)[1]  # Define file name for each channel

    df_sus.to_hdf(filename_hdf5, key=f'sus/{unit}_{time}')
    df_res.to_hdf(filename_hdf5, key=f'res/{unit}_{time}')

def pointcloud_read_old(filename: str, time: str, unit: str = 'day'):
    ''' Reads pandas dataframes with x, y and z position of susceptible and resistant cells from hdf5 file

    :param filename: Name of the hdf5 file.
    :type filename: str

    :param time: Time point of the image.
    :type time: int

    :param unit: Unit of the time point. Example: 'day'
    :type unit: str

    :return: Dataframes of the x, y and z positions of susceptible and resistant cells.
    :rtype: tuple of pandas.DataFrame

    '''
    #load dataframes from hdf5 file if key exist, if not return None
    try:
        df_sus = pd.read_hdf(filename, key=f'sus/{unit}_{time}')
    except:
        df_sus = None
    try:
        df_res = pd.read_hdf(filename, key=f'res/{unit}_{time}')
    except:
        df_res = None
    return df_sus, df_res


def pointcloud_read(filename: str, time: str, unit: str = 'd'):
    df_sus = pd.DataFrame(columns=['x','y'])
    df_res = pd.DataFrame(columns=['x','y'])

    with h5py.File(filename, 'r') as f:
        for kind in ('sus', 'res'):
            ds = f.get(f'{kind}/{unit}_{time}/block0_values')
            if ds is None:
                continue

            raw = ds[...]
            arr = np.atleast_2d(raw)

            # coordinate data?
            if arr.shape[1] == 2:
                df_kind = pd.DataFrame(arr[:, ::-1], columns=['x','y'])

            # single-value “overexposed” count?
            elif arr.shape[1] == 1:
                # make a 1×1 DataFrame so .values[0][0] grabs the count
                df_kind = pd.DataFrame(arr)

            else:
                # unexpected shape, skip
                continue

            if kind == 'sus':
                df_sus = df_kind
            else:
                df_res = df_kind

    return df_sus, df_res


def plot_3d(df_sus, df_res):

    '''

    Plots interactive 3D Scatterplot with Plotly from Dataframes

    :param df_sus: Dataframe of the x, y and z positions of susceptible cells.
    :type df_sus: pandas.DataFrame

    :param df_res: Dataframe of the x, y and z positions resistant cells.
    :type df_res: pandas.DataFrame

    :return: None
    :rtype: None

    '''

    fig = go.Figure(
        data=[go.Scatter3d(
            x=df_res['x'].values,
            y=df_res['y'].values,
            z=df_res['z'].values,
            mode='markers',
            marker=dict(size=15, opacity=0.7, color='red'),
            name='res'),  # Set the legend label for trace 1
            go.Scatter3d(
                x=df_sus['x'].values,
                y=df_sus['y'].values,
                z=df_sus['z'].values,
                mode='markers',
                marker=dict(size=15, opacity=0.7, color='green'),
                name='sus'  # Set the legend label for trace 2
            )])

    # Update layout for a better view
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title='X Axis [µm]', range=[0, 1200]),
            yaxis=dict(title='Y Axis [µm]', range=[0, 1200]),
            zaxis=dict(title='Z Axis [µm]', range=[0, 800]),
        aspectmode='cube'  # This ensures equal aspect ratio
        )
    )

    fig.show()

def plot_2d(df_sus, df_res):

    '''

    Plots 2D Scatterplot with Plotly from Dataframes

    :param df_sus: Dataframe of the x and y positions of susceptible cells.
    :type df_sus: pandas.DataFrame

    :param df_res: Dataframe of the x and y positions resistant cells.
    :type df_res: pandas.DataFrame

    :return: None
    :rtype: None

    '''

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df_res['x'].values,
                y=df_res['y'].values,
                mode='markers',
                marker=dict(size=20, opacity=0.5, color='red'),
                name='res'),  # Set the legend label for trace 1
            go.Scatter(
                x=df_sus['x'].values,
                y=df_sus['y'].values,
                mode='markers',
                marker=dict(size=20, opacity=0.5, color='green'),
                name='sus'  # Set the legend label for trace 2
            )
        ])

    # Update layout for a better view
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(title='X Axis [µm]', scaleanchor='y'),
        yaxis=dict(title='Y Axis [µm]', scaleratio = 1)
    )

    fig.show()

def high_low_correction(df_to_correct, masks_low, x_scale, y_scale, masks_to_correct = None):
    '''

    Corrects for double events in colonies which were imaged at low and high intensities. The function takes the centers of the cells from the two channels and deletes the low imaged cells if it is in a mask of a high imaged cell.

    :param df_sus: pandas df with the centers of the susceptible cells
    :param df_res: pandas df with the centers of the susceptible cells
    :param masks_res: mask of the resistant cells
    :param masks_sus: mask of the susceptible cells
    :return: df_sus, df_res

    '''

    # transform the masks with x- and y-scale
    df_corr1 = df_to_correct.copy()


    # Generate indices_to_drop using list comprehension
    indices_to_drop = [index for index, row in df_corr1.iterrows() if masks_low[int(row['y'] / y_scale), int(row['x'] / x_scale)] != 0]
    df_corr1.drop(indices_to_drop, inplace=True)

    # drop the indices to swap from the susceptible mask
    #masks_to_correct[np.isin(masks_to_correct, np.array(indices_to_drop) + 1)] = 0

    return df_corr1

def bleed_through_correction(df_sus, df_res, masks_sus, masks_res, x_scale, y_scale, img, channel_which_bleeds_through = channel_which_bleeds_through, change_masks = change_masks, channels = channels):
    '''

    Corrects for bleed through between channels. The function takes the centers of the cells from the two channels and deletes the susceptible cells if it is in a mask of a resistant cell.

    :param df_sus: pandas df with the centers of the susceptible cells
    :param df_res: pandas df with the centers of the susceptible cells
    :param masks_res: mask of the resistant cells
    :param masks_sus: mask of the susceptible cells
    :return: df_sus, df_res

    '''
    #transform the masks with x- and y-scale
    df_sus_corr = df_sus.copy()
    df_res_corr = df_res.copy()
    masks_sus_corrected = masks_sus.copy()
    masks_res_corrected = masks_res.copy()
    indices_to_swap = []

    # Generate indices_to_drop using list comprehension
    indices_to_drop = [index for index, row in df_sus_corr.iterrows() if masks_res[int(row['y'] / y_scale), int(row['x'] / x_scale)] != 0]
    df_sus_corr.drop(indices_to_drop, inplace=True)

    # Modify masks_sus_corrected if change_masks is True
    if change_masks:
        masks_sus_corrected[np.isin(masks_sus_corrected, np.array(indices_to_drop) + 1)] = 0
        for index, row in df_sus_corr.iterrows():
            #get the average pixel intensity of the red image channel for the susceptible mask
            mask_values_res = img[channel_which_bleeds_through][masks_sus == index + 1]
            mask_values_sus = img[channels[1][0] - 1][masks_sus == index + 1]
            intensity_res = np.mean(mask_values_res)
            intensity_sus = np.mean(mask_values_sus)
            if intensity_res > intensity_sus:
                indices_to_swap.append(index)
                df_res_corr = pd.concat([df_res_corr, row.to_frame().T], ignore_index=True)
                index_res = df_res_corr.index[-1]
                masks_res_corrected[masks_sus == index + 1] = index_res + 1

        #drop the indices to swap from the susceptible mask
        masks_sus_corrected[np.isin(masks_sus_corrected, np.array(indices_to_swap) + 1)] = 0
        df_sus_corr.drop(indices_to_swap, inplace=True)



    return df_sus_corr, df_res_corr, masks_sus_corrected, masks_res_corrected

def timepoint_analysis(dask, file_path, three_d, x_scale, y_scale, z_scale, npy_done, hdf5_savepath, scene = 0, scene_names_list = ('Well'), diameter = [None, None], minsize = minsize, flow_Threshold=[0.0, 0.0], cellprob_threshold=[0.0, 0.0], timepoint='00', channels = channels, bleed_through = bleed_through, change_masks = change_masks, cellpose_model_input = cellpose_models):

    '''

    This function runs the analysis for a single timepoint of a czi file. It can be used for 2D and 3D images, which it detects from the three_d parameter.
    It can also be used for 2D and 3D NPY files from CellPose. This is detected from the npy_done parameter.
    In the end it saves the centers of the cells to a hdf5 file and plots the scatterplot of the cells.

    :param dask: Dask array of the image of the dimension (z-stack, channels, x, y)
    :type dask: dask.array.core.Array

    :param file_path: Path to the image
    :type file_path: str

    :param three_d: If True, 3D segmentation is performed, else 2D.
    :type three_d: bool

    :param x_scale: The pixel size in x direction in micrometers (µm). Example: 0.65
    :type x_scale: float

    :param y_scale: The pixel size in y direction in micrometers (µm). Example: 0.65
    :type y_scale: float

    :param z_scale: The pixel size in z direction in micrometers (µm). Example: 10
    :type z_scale: float

    :param npy_done: If True, the NPY files were already previously created and the centers are calculated from the NPY files.
                     If False, the NPY files are created and the centers are calculated from the newly created NPY files.
    :type npy_done: bool

    :param hdf5_savepath: Path to save the hdf5 file created
    :type hdf5_savepath: str

    :param scene: Name of the scene of the image. Example: 'TR12-B2'
    :type scene: str

    :param scene_names_list: List of names of scenes of the image
    :type scene_names_list: list of str

    :return: None
    :rtype: None

    '''
    mask_sus, mask_res, flows_sus, flows_res = [None, None], [None, None], [None, None], [None, None]


    #change scene name if Tile Region is not in the well in ZEN
    pattern = r"^([A-Z]\d+)(_.+?)(?:-([A-Z]\d+))?$"
    scene_saving = scene
    m = re.match(pattern, scene)
    if m:
        first_code, middle, second_code = m.groups()

        if second_code:  # second exists
            if first_code != second_code:
                scene_saving = f"{first_code}{middle}-{first_code}"
        else:  # missing second code
            scene_saving = f"{first_code}{middle}-{first_code}"

    #dask_high = dask[:, [0, 1, 3], :, :, :]
    #dask_low = dask[:, [0, 2, 4], :, :, :]
    #dask_high = dask[:, [0, 1], :, :, :]
    #dask_low = dask[:, [0, 1], :, :, :]
    dask_a = dask[:, [0, 1, 2], :, :, :]
    list_dask = [dask_a]

    for i, dask in enumerate(list_dask):
        if high_low_imaging:
            if i == 0:
                intensity = 'high'
                cellpose_model_input = [cellpose_models[0], cellpose_models[1]]
                minsize = [250, 250]
                cellprob_threshold = [-1, -1]
            elif i == 1:
                intensity = 'low'
                minsize = [250, 150]
                cellprob_threshold = [-1, -2]
                cellpose_model_input = [cellpose_models[0], cellpose_models[1]]
        elif high_low_imaging == False:
            intensity = None
            cellpose_model_input = [cellpose_models[0], cellpose_models[1]]
            minsize = [250, 250]
            cellprob_threshold = [-1, -1]

        # Convert dask array to numpy array eliminating the first dimension (time)
        timepoint_numpy = dask[0].compute()

        # Define file name for each scene
        if intensity is None:
            NPY_path = os.path.splitext(file_path)[0] + '_' + str(scene_saving) + os.path.splitext(file_path)[1]
        else:
            NPY_path = os.path.splitext(file_path)[0] + '_' + str(scene_saving) + '_' + intensity + os.path.splitext(file_path)[1]

        # If npy_done is True, calculate centers from already existing NPY files
        if npy_done == True:

            # Get paths to NPY files
            NPY_path_sus, NPY_path_res = get_npy_file_paths(os.path.join(experiment_path, 'czi_NPY'), os.path.splitext(os.path.basename(file_path))[0], False)
            # Find the correct path for the current scene
            NPY_path_sus_scene = next(path for path in NPY_path_sus if scene_saving in path)
            NPY_path_res_scene = next(path for path in NPY_path_res if scene_saving in path)

            print('NPY path susceptible:', NPY_path_sus_scene)
            print('NPY path resistant:', NPY_path_res_scene)
            # Load NPY files
            mask_sus[i], mask_res[i] = load_mask(NPY_path_sus_scene, NPY_path_res_scene)
        else:
            # CellPose to NPY
            mask_sus[i], mask_res[i], flows_sus[i], flows_res[i] = cellpose_to_NPY(timepoint_numpy, NPY_path, channels=channels, flow_Threshold=flow_Threshold, cellprob_threshold=cellprob_threshold, MinimumSize=minsize, stitch_threshold=[0.0, 0.0], Diameter=diameter, do3D=three_d, cellpose_model_input=cellpose_model_input)

    if high_low_imaging:
        # Calculate centers from cellpose masks
        df_sus_high, df_res_high = calculate_centers_2d(mask_sus[0], mask_res[0], 1, 1)
        df_sus_low, df_res_low = calculate_centers_2d(mask_sus[1], mask_res[1], 1, 1)

        pointcloud_save(df_sus_high, df_res_high, hdf5_savepath, scene_saving, scene_names_list, time=timepoint, unit=time_unit, intensity='high')
        pointcloud_save(df_sus_low, df_res_low, hdf5_savepath, scene_saving, scene_names_list, time=timepoint, unit=time_unit, intensity='low')

        # combine df low and high and delete doubled entries
        df_sus_corr1 = high_low_correction(df_sus_high, mask_sus[1], x_scale, y_scale)
        df_sus_corr2 = pd.concat([df_sus_corr1, df_sus_low], ignore_index=True)
        if bleed_through == True:
            df_sus_corr3 = high_low_correction(df_sus_corr2, mask_res[0], x_scale, y_scale)
            df_sus_corr = high_low_correction(df_sus_corr3, mask_res[1], x_scale, y_scale)
        else:
            df_sus_corr = df_sus_corr2
        df_res_corr1 = high_low_correction(df_res_high, mask_res[1], x_scale, y_scale)
        df_res_corr = pd.concat([df_res_corr1, df_res_low], ignore_index=True)

        # Save corrected Pointclouds to hdf5 file
        pointcloud_save(df_sus_corr, df_res_corr, hdf5_savepath, scene_saving, scene_names_list, time=timepoint, unit=time_unit)
        '''
        # do bleed through correction
        if three_d != True:
            img = timepoint_numpy[:, 0, :, :]
        df_sus, df_res, masks_sus, masks_res = bleed_through_correction(df_sus_low, df_sus_high, mask_sus[1], mask_sus[0], x_scale, y_scale, img=img, channel_which_bleeds_through=channels[0][1]-1)
        if change_masks == True:
            file_name_npy_sus = os.path.splitext(NPY_path)[0] + '_' + str(channels[1]) + '_corrected' + os.path.splitext(NPY_path)[1]
            file_name_npy_res = os.path.splitext(NPY_path)[0] + '_' + str(channels[0]) + '_corrected' + os.path.splitext(NPY_path)[1]
            io.masks_flows_to_seg(images=img, masks=masks_sus, flows=flows_sus, file_names=file_name_npy_sus, channels=channels[1], diams=diameter[1])
            io.masks_flows_to_seg(images=img, masks=masks_res, flows=flows_res, file_names=file_name_npy_res, channels=channels[0], diams=diameter[0])
        '''
    elif high_low_imaging == False:
        df_sus, df_res = calculate_centers_2d(mask_sus[0], mask_res[0], 1, 1)
        pointcloud_save(df_sus, df_res, hdf5_savepath, scene_saving, scene_names_list, time=timepoint, unit=time_unit)

    if overexposure_correction == True:
        # correct for overexposure
        #convert dask[:, [2], :, :, :] to numpy array
        res_over_array = dask[0, [2], 0, :, :]
        add_res_count = pd.DataFrame([[overexposure_correction_function(image=res_over_array, masks=mask_res[0], threshold=overexposure_threshold, max_density=max_density_overexposure_res, scene=str(scene_saving), timepoint=timepoint, channel='res')]], columns=['value'])
        sus_over_array = dask[0, [1], 0, :, :]
        add_sus_count = pd.DataFrame([[overexposure_correction_function(image=sus_over_array, masks=mask_sus[0], threshold=overexposure_threshold, max_density=max_density_overexposure_sus, scene=str(scene_saving), timepoint=timepoint, channel='sus')]], columns=['value'])
        #save the add count data in a hdf5 file
        overexposed_cellcount_save(add_sus_count, add_res_count, hdf5_savepath, scene_saving, scene_names_list, time=timepoint, unit=time_unit)

    # Plot scatterplot
    if plot_scatter == True:
        plot_2d(df_sus_corr, df_res_corr)

def timelapse_analysis(dask, file_path, time_list, time_unit, three_d, x_scale, y_scale, z_scale, npy_done, hdf5_savepath, scene = 0, scene_names_list = ('Well'), diameter = [None, None], minsize = minsize, flow_Threshold=[0.6, 0.6], cellprob_threshold=[0.0, 0.0], channels = [[1, 0],[2, 0]]):

    '''

    This function runs the analysis for a timelapse of a czi file. It can be used for 2D and 3D images, which it detects from the three_d parameter.
    It can also be used for 2D and 3D NPY files from CellPose. This is detected from the npy_done parameter.
    In the end it saves the centers of the cells to a hdf5 file per timepoint and plots the scatterplot of the cells per timepoint.

    :param dask: Dask array of the image of the dimension (time, z-stack, channels, x, y)
    :type dask: dask.array.core.Array

    :param file_path: Path to the image
    :type file_path: str

    :param time_list: List of timepoints of the image
    :type time_list: list of int

    :param time_unit: Unit of the time point. Example: 'day'
    :type time_unit: str

    :param three_d: If True, 3D segmentation is performed, else 2D.
    :type three_d: bool

    :param x_scale: The pixel size in x direction in micrometers (µm). Example: 0.65
    :type x_scale: float

    :param y_scale: The pixel size in y direction in micrometers (µm). Example: 0.65
    :type y_scale: float

    :param z_scale: The pixel size in z direction in micrometers (µm). Example: 10
    :type z_scale: float

    :param npy_done: If True, the NPY files were already previously created and the centers are calculated from the NPY files.
                     If False, the NPY files are created and the centers are calculated from the newly created NPY files.
    :type npy_done: bool

    :param hdf5_savepath: Path to save the hdf5 file created
    :type hdf5_savepath: str

    :param scene: Name of the scene of the image. Example: 'TR12-B2'
    :type scene: str

    :param scene_names_list: List of names of scenes of the image
    :type scene_names_list: list of str

    :return: None
    :rtype: None

    '''

    for i in range(dask.shape[0]):

        print('        Timepoint', i + 1, 'of', dask.shape[0], 'started')
        if len(time_list) != dask.shape[0]:
            raise ValueError('The time_list does not match the amount of timepoints in the czi file')

        # Load dask array from timepoint i and convert to numpy array
        timepoint = dask[i]
        timepoint_numpy = timepoint.compute()

        # Define file name for each scene and timepoint
        NPY_path = os.path.splitext(file_path)[0] + '_' + str(scene) + '_' + str(time_unit[0]) + str(time_list[i]) + os.path.splitext(file_path)[1]

        # If npy_done is True, calculate centers from already existing NPY files
        if npy_done == True:

            # Get paths to NPY files
            NPY_path_list_sus, NPY_path_list_res = get_npy_file_paths(experiment_path, os.path.splitext(os.path.basename(file_path))[0], True)

            # Load NPY files
            cellpose_output_sus, cellpose_output_res = load_mask(NPY_path_list_sus[scene_names_list.index(scene)][i], NPY_path_list_res[scene_names_list.index(scene)][i])

            # Calculate centers from cellpose masks
            if not three_d:
                df_sus, df_res = calculate_centers_2d(cellpose_output_sus['masks'], cellpose_output_res['masks'], 1, 1)
            else:
                df_sus, df_res = calculate_centers_3d(cellpose_output_sus['masks'], cellpose_output_res['masks'], x_scale, y_scale, z_scale)

        else:

            # CellPose to NPY
            mask_sus, mask_res, flows_sus, flows_res = cellpose_to_NPY(timepoint_numpy, NPY_path, channels=channels,
                                                 flow_Threshold=flow_Threshold, cellprob_threshold=cellprob_threshold, MinimumSize=minsize,
                                                 stitch_threshold=[0.0, 0.0], Diameter=diameter, do3D=three_d)

            # Calculate centers from cellpose masks
            if not three_d:
                df_sus, df_res = calculate_centers_2d(mask_sus, mask_res, 1, 1)
            else:
                df_sus, df_res = calculate_centers_3d(mask_sus, mask_res, x_scale, y_scale, z_scale)

        # Save Pointclouds to hdf5 file
        pointcloud_save(df_sus, df_res, hdf5_savepath, time_list[i], time_unit, scene, scene_names_list)

        # Plot scatterplot
        if plot_scatter == True:

            if not three_d:
                plot_2d(df_sus, df_res)
            else:
                plot_3d(df_sus, df_res)

def get_npy_file_paths(folder_path, czi_file_name, is_timelapse):
    '''

    For a timelapse:
        Returns a 2D array of file paths for all scenes and days in the experiment for all saved NPY files to analyze
    For a timepoint:
        Returns a 1D array of file paths for all scenes in the experiment for all saved NPY files to analyze

    :param folder_path: Path to the folder where the NPY files are saved
    :type folder_path: str

    :param czi_file_name: Name of the experiment (and the corresponding NPY files)
    :type czi_file_name: str

    :param is_timelapse: If True, the experiment is a timelapse, else it is a single timepoint
    :type is_timelapse: bool

    :return: 2D array of file paths for all scenes and days in the experiment for all saved NPY files to analyze
    :rtype: numpy.ndarray

    '''

    if is_timelapse == True:

        channel_1_dict = defaultdict(dict)
        channel_2_dict = defaultdict(dict)

        # Adjust the regex pattern
        pattern = re.compile(rf"{re.escape(czi_file_name)}_([a-zA-Z0-9_-]+)_[dt](\d+)\[(\d+), (\d+)\]_masks\.tif")

        for file in os.listdir(folder_path):
            if file.endswith(".tif") and czi_file_name in file:
                match = pattern.match(file)
                if match:
                    scene, day, channel, _ = match.groups()
                    day = int(day)

                    channel = int(channel)
                    file_path = os.path.join(folder_path, file)

                    if channel == 1 or channel == 3:
                        channel_1_dict[scene][day] = file_path
                    elif channel == 2:
                        channel_2_dict[scene][day] = file_path
                else:
                    print("No match for file:", file)

        def dict_to_np_array(d):
            max_days = max(max(days.keys()) for days in d.values()) if d else 0
            arr_list = []

            for scene in sorted(d.keys()):
                scene_list = [d[scene].get(day) for day in range(1, max_days + 1) if day in d[scene]]
                arr_list.append(scene_list)

            return np.array(arr_list, dtype=object)

        channel_1_files = dict_to_np_array(channel_1_dict)
        channel_2_files = dict_to_np_array(channel_2_dict)

        return channel_2_files, channel_1_files

    else:

        channel_1_files = []
        channel_2_files = []

        # Adjust the regex pattern to match new file format
        pattern = re.compile(rf"{re.escape(czi_file_name)}_([a-zA-Z0-9_-]+)\[(\d+), (\d+)\]_masks\.tif")
        for file in os.listdir(folder_path):
            if file.endswith(".tif") and czi_file_name in file:
                match = pattern.match(file)
                if match:
                    scene, channel, _ = match.groups()

                    channel = int(channel)
                    file_path = os.path.join(folder_path, file)

                    if channel == 1 or channel == 3:
                        channel_1_files.append(file_path)
                    elif channel == 2:
                        channel_2_files.append(file_path)
                else:
                    print("No match for file:", file)

        return np.array(channel_2_files, dtype=object), np.array(channel_1_files, dtype=object)

def get_hdf5_file_paths(experiment_path, experiment_name):

    '''

    Returns a 2D array of file paths for all scenes and days in the experiment for all saved hdf5 files to analyze

    :param experiment_path: Path to the folder where the hdf5 files are saved
    :type experiment_path: str

    :param experiment_name: Name of the experiment (and the corresponding hdf5 files)
    :type experiment_name: str

    :return: 2D array of file paths for all scenes and days in the experiment for all saved hdf5 files to analyze
    :rtype: numpy.ndarray

    '''

    file_dict = {}

    # Regular expression to match the file naming pattern
    pattern = re.compile(rf"{re.escape(experiment_name)}_scene(.*?)_{time_unit}(\d+)\.h5")

    for file in os.listdir(experiment_path):
        if file.endswith(".h5") and experiment_name in file:
            match = pattern.search(file)
            if match:
                scene, day = match.groups()
                day = int(day)
                if scene not in file_dict:
                    file_dict[scene] = {}
                file_dict[scene][day] = os.path.join(experiment_path, file)
            else:
                print("No match for file:", file)  # Print if no match found

    # Find unique scenes and days to create a consistent indexing
    all_scenes = sorted(file_dict.keys())
    all_days = sorted({day for scene in file_dict for day in file_dict[scene]})
    scene_to_idx = {scene: idx for idx, scene in enumerate(all_scenes)}
    day_to_idx = {day: idx for idx, day in enumerate(all_days)}

    # Create a 2D array of file paths
    file_paths_array = np.full((len(all_scenes), len(all_days)), None)

    # Fill the array with file paths
    for scene in file_dict:
        for day in file_dict[scene]:
            scene_idx = scene_to_idx[scene]
            day_idx = day_to_idx[day]
            file_paths_array[scene_idx, day_idx] = file_dict[scene][day]
    #file_paths_array = np.array([path for path in scene if  'total' not in path])
    return file_paths_array

def analyze_files_and_plot(experiment_path, experiment_name, time_list, time_unit, three_d, is_timelapse):

    '''

    Description: This function analyzes all saved hdf5 files for a given experiment (timelapse) in a given folder and plots the line graph for each scene.

    :param experiment_path: Path to the folder where the hdf5 files are saved
    :type experiment_path: str

    :param experiment_name: Name of the experiment (and the corresponding hdf5 files)
    :type experiment_name: str

    :param time_list: List of timepoints of the image
    :type time_list: list of int

    :param time_unit: Unit of the time point. Example: 'day'
    :type time_unit: str

    :param three_d: If True, 3D segmentation is performed, else 2D.
    :type three_d: bool

    :param is_timelapse: If True, the experiment is a timelapse, else it is a single timepoint
    :type is_timelapse: bool

    :return: 3D array of the number of susceptible and resistant cells for each scene and day
    :rtype: numpy.ndarray

    '''

    # Find all saved hdf5 files for a given experiment in a given folder and append it to a 2D array (First dimension: Scenes, Second dimension: Days)
    hdf_file_paths = get_hdf5_file_paths(experiment_path, experiment_name)

    # Loop over all hdf5 paths and loaf them into the two dataframes (sus and res). Extract the length of these dataframes and append them to a list for each scene
    results_array = np.zeros((hdf_file_paths.shape[0], hdf_file_paths.shape[1], 2), dtype=int)

    for scene_idx, scene_files in enumerate(hdf_file_paths):
        scene_name = None

        for day_idx, file_path in enumerate(scene_files):
            if file_path is not None:
                # Extract scene name from the first file path
                if scene_name is None:
                    regex_pattern = rf'scene([^_]+)_{time_unit}'
                    match = re.search(regex_pattern, file_path)
                    if match:
                        scene_name = match.group(1)

                df_sus, df_res = pointcloud_read(file_path, time_list[day_idx], time_unit)
                results_array[scene_idx, day_idx, 0] = len(df_sus)
                results_array[scene_idx, day_idx, 1] = len(df_res)

                if plot_interactiveplot_from_hdf5 == True:

                    # Plotting the interactive graph for each scene

                    if not three_d:
                        plot_2d(df_sus, df_res)
                    else:
                        plot_3d(df_sus, df_res)


        if plot_linegraph == True and is_timelapse == True:

            # Plotting the graph for each scene
            plt.figure()
            plt.plot(time_list, results_array[scene_idx, :, 0], color='green', label='sus_cells')
            plt.plot(time_list, results_array[scene_idx, :, 1], color='red', label='res_cells')
            plt.xlabel('Time (' + time_unit + ')')
            plt.ylabel('Number of Cells')
            plt.title(scene_name if scene_name else f'Scene {scene_idx + 1}')
            plt.legend()
            plt.show()

    return results_array

def save_results_today_in_txt(timepoint, loading_path=hdf5_savepath, saving_path=results_txt_today_path, unit=time_unit):
    '''

    saves all results of the current day in one txt file

    :param loading_path: path of the directory the hdf5 files should be loaded from
    :type loading_path: str

    :param saving_path: path of the directory the txt file should be saved to
    :type saving_path: str

    :return: None
    :rtype: None

    '''

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    #deletes text file if already existing
    if os.path.exists(saving_path + '\\Results_' + str(unit) + str(timepoint) + '.txt'):
        os.remove(saving_path + '\\Results_' + str(unit) + str(timepoint) + '.txt')

    for file in os.listdir(os.path.dirname(loading_path)):
        #only use files where there is no high and low in filename
        if 'high' in file or 'low' in file or 'overexposed' in file:
            continue
        else:
            file_path = os.path.join(os.path.dirname(loading_path), file)

        #load the results from the hdf5 file
        df_sus, df_res = pointcloud_read(file_path, time=timepoint, unit=time_unit)
        if df_sus is None or df_res is None:
            continue


        #get number of lines in the dataframe
        cellnumber_sus = len(df_sus)
        if df_sus.shape[1] == 1:
            cellnumber_sus = 0

        # if only one column please return the value 0
        cellnumber_res = len(df_res)
        if df_res.shape[1] == 1:
            cellnumber_res = 0

        #save the results in a txt file in the following order: hdf5-filename, susceptible cellnumber, resistant cellnumber; with space as seperator
        with open(saving_path + '\\Results_' + str(unit) + str(timepoint) + '.txt', 'a') as f:
            f.write(file[:-3] + '_' + time_unit + timepoint + ' ' + str(cellnumber_sus) + ' ' + str(cellnumber_res) + '\n')

def save_results_in_txt(loading_path=hdf5_loading_path, saving_path=results_txt_path, unit=time_unit):
    '''

    saves all results of one hdf5 file in txt file

    :param loading_path: path of the directory the hdf5 files should be loaded from
    :type loading_path: str

    :param saving_path: path of the directory the txt file should be saved to
    :type saving_path: str

    :return: None
    :rtype: None

    '''

    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    if overexposure_correction == True:
        for file in os.listdir(loading_path):
            if 'overexposed' in file:
                continue
            if 'total' in file:
                continue
            file_path = os.path.join(loading_path, file)
            #find corresponding overexposed file
            overexposed_file = file[:-3] + '_overexposed.h5'
            overexposed_file_path = os.path.join(loading_path, overexposed_file)
            saving_path_file_total = os.path.join(saving_path, 'Results_' + str(os.path.splitext(file)[0]) + '.txt')
            saving_path_file = os.path.join(saving_path, 'Results_' + str(os.path.splitext(file)[0]) + '_seg.txt')
            saving_path_overexposed_file = os.path.join(saving_path, 'Results_' + str(os.path.splitext(overexposed_file)[0]) + '.txt')
            # deletes text file if already existing
            if os.path.exists(saving_path_file):
                os.remove(saving_path_file)
            if os.path.exists(saving_path_file_total):
                os.remove(saving_path_file_total)
            if os.path.exists(saving_path_overexposed_file):
                os.remove(saving_path_overexposed_file)

            #extracting timepoints from hdf5 file keys
            keys = list(h5py.File(file_path, 'r').get('sus').keys())
            #keys_overexposed = list(h5py.File(overexposed_file_path, 'r').get('sus').keys())
            timepoints_hdf5 = [ke.split('_')[1] for ke in keys]

            #load the results from the hdf5 file
            for timepoint_hdf5 in timepoints_hdf5:
                df_sus, df_res = pointcloud_read(file_path, time=timepoint_hdf5, unit=time_unit)
                df_sus_over, df_res_over = pointcloud_read(overexposed_file_path, time=timepoint_hdf5, unit=time_unit)

                # get number of lines in the dataframe
                cellnumber_sus = len(df_sus)
                if df_sus.shape[1] == 1:
                    cellnumber_sus = 0

                # if only one column please return the value 0
                cellnumber_res = len(df_res)
                if df_res.shape[1] == 1:
                    cellnumber_res = 0

                if df_sus_over is not None and not df_sus_over.empty:
                    cellnumber_sus_over = df_sus_over.iloc[0][0]
                else:
                    print("Warning: df_sus_over is None or empty for timepoint", timepoint_hdf5)
                    cellnumber_sus_over = 0

                if df_res_over is not None and not df_res_over.empty:
                    cellnumber_res_over = df_res_over.iloc[0][0]
                else:
                    print("Warning: df_res_over is None or empty for timepoint", timepoint_hdf5)
                    cellnumber_res_over = 0

                total_cellnumber_sus = cellnumber_sus + cellnumber_sus_over
                total_cellnumber_res = cellnumber_res + cellnumber_res_over

                #save the results in a txt file in the following order: hdf5-filename, susceptible cellnumber, resistant cellnumber; with space as seperator
                with open(saving_path_file, 'a') as f:
                    f.write(file[:-3] + '_' + time_unit + timepoint_hdf5 + ' ' + str(cellnumber_sus) + ' ' + str(cellnumber_res) + '\n')
                with open(saving_path_overexposed_file, 'a') as f:
                    f.write(overexposed_file[:-3] + '_' + time_unit + timepoint_hdf5 + ' ' + str(cellnumber_sus_over) + ' ' + str(cellnumber_res_over) + '\n')
                with open(saving_path_file_total, 'a') as f:
                    f.write(file[:-3] + '_' + time_unit + timepoint_hdf5 + ' ' + str(total_cellnumber_sus) + ' ' + str(total_cellnumber_res) + '\n')

    elif overexposure_correction == False:
        for file in os.listdir(loading_path):
            file_path = os.path.join(loading_path, file)
            #find corresponding overexposed file
            overexposed_file_path = os.path.join(loading_path, file_path)
            saving_path_file = os.path.join(saving_path, 'Results_' + str(os.path.splitext(file)[0]) + '.txt')
            # deletes text file if already existing
            if os.path.exists(saving_path_file):
                os.remove(saving_path_file)

            #extracting timepoints from hdf5 file keys
            keys = list(h5py.File(file_path, 'r').get('sus').keys())
            timepoints_hdf5 = [ke.split('_')[1] for ke in keys]

            #load the results from the hdf5 file
            for timepoint_hdf5 in timepoints_hdf5:
                df_sus, df_res = pointcloud_read(file_path, time=timepoint_hdf5, unit=time_unit)

                #get number of lines in the dataframe
                cellnumber_sus = len(df_sus)
                cellnumber_res = len(df_res)

                #save the results in a txt file in the following order: hdf5-filename, susceptible cellnumber, resistant cellnumber; with space as seperator
                with open(saving_path_file, 'a') as f:
                    f.write(file[:-3] + '_' + time_unit + timepoint_hdf5 + ' ' + str(cellnumber_sus) + ' ' + str(cellnumber_res) + '\n')

def create_circular_array(shape):
    center = round(shape // 2)          # Center of the array
    radius = shape // 2
    y, x = np.ogrid[:shape, :shape]
    mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
    circ_array = np.zeros((shape, shape))
    circ_array[mask] = 1
    return circ_array

def plot_NPY_density(saving_path, npy_directory, xscale = 1, yscale = 1, radius = 99):
    '''

    reads all NPY files in a given directory and plots the density of the cells in the image

    :param saving_path: path of the directory the density plots should be saved to
    :type saving_path: str

    :param npy_directory: path of the directory the npy files should be loaded from
    :type npy_directory: str

    :param radius: radius of the circular mean filter
    :type radius: int

    :param loading_path: path of the directory the hdf5 files should be loaded from
    :type loading_path: str
    '''

    # get list of NPY file names
    npy_files_list_total = []
    for file in os.listdir(npy_directory):
        if file.endswith(".npy"):
            npy_files_list_total.append(file)

    npy_files_list = np.unique([filename.split('[')[0] for filename in npy_files_list_total])

    df_list_sus = []
    df_list_res = []

    circ_weights = create_circular_array(radius)  # create the circular mean filter



    for file in npy_files_list:
        basepath = os.path.join(npy_directory, file)

        # get name of scene out of NPY file name
        #scene = re.findall(r'_MIPs_([a-zA-Z0-9-]+)_\[', file)[0]

        # extracting timepoints from NPY filenames
        #timepoint = re.findall(r'_[dt](\d\d)_', file)[0]

        specific_files = [item for item in npy_files_list_total if file in item]
        print(specific_files)

        res_file = [item for item in specific_files if '[1' in item]
        res_file_path = os.path.join(npy_directory, res_file[0])
        sus_file = [item for item in specific_files if '[2' in item]
        sus_file_path = os.path.join(npy_directory, sus_file[0])

        #load masks from NPY file
        cellpose_sus, cellpose_res = load_mask(sus_file_path, res_file_path)
        df_centers_sus, df_centers_res = calculate_centers_2d(cellpose_sus['masks'], cellpose_res['masks'], 1, 1)

        complete_name_sus = os.path.join(saving_path, file + 'density_sus.tif')
        complete_name_res = os.path.join(saving_path, file + 'density_res.tif')

        dat_res = np.load(res_file_path, allow_pickle=True).item()
        image_res = dat_res['img']
        if image_res.shape[0] == 3:
            image_res = image_res.transpose(1, 2, 0)

        # get image dimensions from NPY file
        ydim, xdim = image_res.shape[0:2]
        mean_filter_data_sus = np.zeros((ydim, xdim))
        mean_filter_data_res = np.zeros((ydim, xdim))
        for i in range(len(df_centers_sus)):
            x, y = df_centers_sus.iloc[i]['x'], df_centers_sus.iloc[i]['y']
            mean_filter_data_sus[y, x] = 1
        for i in range(len(df_centers_res)):
            x, y = df_centers_res.iloc[i]['x'], df_centers_res.iloc[i]['y']
            mean_filter_data_res[y, x] = 1

        # apply a mean filter to the images
        filtered_data_res = scipy.ndimage.convolve(mean_filter_data_res, mode='constant', weights=circ_weights)
        filtered_data_sus = scipy.ndimage.convolve(mean_filter_data_sus, mode='constant', weights=circ_weights)
        im_res = Image.fromarray(filtered_data_res)
        im_sus = Image.fromarray(filtered_data_sus)
        im_res.save(complete_name_res)
        im_sus.save(complete_name_sus)
    return

def plot_hdf5_density(saving_path, npy_directory, xscale, yscale, radius = 99, loading_path=hdf5_loading_path):
    '''

    reads all hdf5 files in a given directory and plots the density of the cells in the image. Be careful, the centers are saved in the hdf5 files in microns, so by backtransformation in pixels there is an additional rounding error.

    :param saving_path: path of the directory the density plots should be saved to
    :type saving_path: str

    :param npy_directory: path of the directory the npy files should be loaded from
    :type npy_directory: str

    :param radius: radius of the circular mean filter
    :type radius: int

    :param loading_path: path of the directory the hdf5 files should be loaded from
    :type loading_path: str

    '''
    # get list of NPY file names
    npy_files_list = []
    for file in os.listdir(npy_directory):
        if file.endswith(".npy"):
            npy_files_list.append(file)

    df_list_sus = []
    df_list_res = []

    circ_weights = create_circular_array(radius)    # create the circular mean filter

    for file in os.listdir(loading_path):
        file_path = os.path.join(loading_path, file)

        # get name of scene out of hdf5 file name
        scene = re.findall(str(experiment_name)+r'_([a-zA-Z0-9-_]+).', file)[0]

        # extracting timepoints from hdf5 file keys
        keys = list(h5py.File(file_path, 'r').get('sus').keys())
        timepoints_hdf5 = [ke.split('_')[1] for ke in keys]
        time_unit = keys[0].split('_')[0]

        # load the results from the hdf5 file
        for timepoint_hdf5 in timepoints_hdf5:
            completeNameG = os.path.join(saving_path, os.path.splitext(os.path.basename(file))[0] + '_' + time_unit + '_' + timepoint_hdf5 + '_density_sus.tif')
            completeNameR = os.path.join(saving_path, os.path.splitext(os.path.basename(file))[0] + '_' + time_unit + '_' + timepoint_hdf5 + '_density_res.tif')

            df_sus, df_res = pointcloud_read(file_path, time=timepoint_hdf5, unit=time_unit)

            df_list_sus.append(df_sus)
            df_list_res.append(df_res)

            # find corresponding NPY file to hdf5 file
            matchingR = [s for s in npy_files_list if timepoint_hdf5 in s and '[1' in s and scene in s]
            npy_filepath_R = os.path.join(npy_directory, matchingR[0])
            print(npy_filepath_R)
            datR = np.load(npy_filepath_R, allow_pickle=True).item()
            imageR = datR['img']
            if imageR.shape[0] == 3:
                imageR = imageR.transpose(1, 2, 0)

            # # get image dimensions from NPY file
            ydim, xdim = imageR.shape[0:2]
            mean_filter_data_G = np.zeros((ydim, xdim))
            mean_filter_data_R = np.zeros((ydim, xdim))
            for i in range(len(df_sus)):
                xG, yG = round(df_sus.iloc[i]['x']/yscale), round(df_sus.iloc[i]['y']/xscale)
                mean_filter_data_G[yG, xG] = 1
            for i in range(len(df_res)):
                xR, yR = round(df_res.iloc[i]['x']/yscale), round(df_res.iloc[i]['y']/xscale)
                mean_filter_data_R[yR, xR] = 1

            # apply a mean filter to the image
            filtered_data_R = scipy.ndimage.convolve(mean_filter_data_R, mode='constant', weights=circ_weights)
            filtered_data_G = scipy.ndimage.convolve(mean_filter_data_G, mode='constant', weights=circ_weights)
            imR = Image.fromarray(filtered_data_R)
            imG = Image.fromarray(filtered_data_G)
            imR.save(completeNameR)
            imG.save(completeNameG)
    return

def stack_density_to_timestack(saving_path, loading_path):
    '''

    reads all density tif files in a given directory and merges the images to a timestack

    :param saving_path: path of the directory the density plots should be saved to
    :type saving_path: str

    :param loading_path: path of the directory the density plots should be loaded from
    :type loading_path: str

    '''

    image_timesteps = {}

    # Iterate over the files in the folder
    for filename in os.listdir(loading_path):
        if filename.endswith('res.tif'):
            #get the name of the tile region
            scene = re.findall(r'MIPs_([a-zA-Z0-9-_]+)_density', filename)[0]
            # Get the timestep identifier (e.g., 01, 02, 03, etc.)
            timestep = re.findall(r'_[dt](\d\d)_', filename)[0]

            # Check if the scene is already in the dictionary
            if scene + '_res' in image_timesteps:
                # If it is, add the current filename to the corresponding timestep's list
                image_timesteps[scene + '_res'][timestep] = filename
            else:
                # If it's not, create a new entry in the dictionary
                image_timesteps[scene + '_res'] = {timestep: filename}

        if filename.endswith('sus.tif'):
            # get the name of the tile region
            scene = re.findall(r'MIPs_([a-zA-Z0-9-_]+)_density', filename)[0]
            # Get the timestep identifier (e.g., 01, 02, 03, etc.)
            timestep = re.findall(r'_[dt](\d\d)_', filename)[0]

            # Check if the scene is already in the dictionary
            if scene + '_sus' in image_timesteps:
                # If it is, add the current filename to the corresponding timestep's list
                image_timesteps[scene + '_sus'][timestep] = filename
            else:
                # If it's not, create a new entry in the dictionary
                image_timesteps[scene + '_sus'] = {timestep: filename}

    # Iterate over the image timesteps and merge them
    for scene, timesteps in image_timesteps.items():
        # Sort the timesteps to ensure correct ordering
        sorted_timesteps = sorted(timesteps.keys())

        # Create an image stack from the timestep images
        max_width = max((tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)[0] for timestep in sorted_timesteps)
        print(max_width)
        total_height = max((tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)[1] for timestep in sorted_timesteps)
        print(total_height)

        concatenated_images = []
        # Paste each image into the concatenated image, center-aligned
        for timestep in sorted_timesteps:
            img = Image.open(os.path.join(loading_path, timesteps[timestep]))
            x_offset = round((max_width - img.size[0]) / 2)
            y_offset = round((total_height - img.size[1]) / 2)
            concatenated_image = Image.new("L", (max_width, total_height))
            concatenated_image.paste(img, (x_offset, y_offset))
            concatenated_images.append(concatenated_image)

        # Save the image stack
        merged_filename = scene + ".tif"
        merged_filepath = os.path.join(saving_path, merged_filename)
        tiff.imwrite(merged_filepath, concatenated_images, bigtiff=True)

    return

def stack_tifs_to_timestack(saving_path, loading_path, scaling_factor=0.25, channel_number=3, channels=[1, 0, 4]):
    '''

    reads all density tif files in a given directory and merges the images to a timestack

    :param saving_path: path of the directory the density plots should be saved to
    :type saving_path: str

    :param loading_path: path of the directory the density plots should be loaded from
    :type loading_path: str

    '''

    image_timesteps = {}
    tiffiles = [f for f in os.listdir(loading_path) if f.endswith('.tif')]
    # Iterate over the files in the folder
    for filename in tiffiles:


        # get the name of the tile region
        scene = re.findall(r'MIPs_([a-zA-Z0-9-_]+)\.tif', filename)[0]
        # Get the timestep identifier (e.g., 01, 02, 03, etc.)
        timestep = re.findall(r'_[dt](\d\d)_', filename)[0]

        # Check if the scene is already in the dictionary
        if scene in image_timesteps:
            # If it is, add the current filename to the corresponding timestep's list
            image_timesteps[scene][timestep] = filename
        else:
            # If it's not, create a new entry in the dictionary
            image_timesteps[scene] = {timestep: filename}
    # Iterate over the image timesteps and merge them
    for scene, timesteps in image_timesteps.items():
        print(scene)
        print(timesteps)
        # Sort the timesteps to ensure correct ordering
        sorted_timesteps = sorted(timesteps.keys())
        print(sorted_timesteps)
        #print(tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)
        max_width = max((tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)[2] for timestep in sorted_timesteps)
        print(max_width)
        total_height = max((tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)[1] for timestep in sorted_timesteps)
        print(total_height)

        concatenated_images = []
        # Paste each image into the concatenated image, center-aligned
        for timestep in sorted_timesteps:
            print(timestep)
            print(timesteps[timestep])
            img = tiff.imread(os.path.join(loading_path, timesteps[timestep]))
            print(img.shape)
            #only get three channels 0, 1, 4
            img = img[channels,:, :]
            x_offset = round(((max_width - img.shape[2]) / 2)*scaling_factor)
            y_offset = round(((total_height - img.shape[1]) / 2)*scaling_factor)
            #creating a new 3 channel image with the dimensions of the maximum image
            concatenated_image = np.zeros((channel_number, round(total_height*scaling_factor), round(max_width*scaling_factor)))
            #resize image
            resized_img = skimage.transform.rescale(img, scaling_factor, channel_axis=0, anti_aliasing=True)

            #paste the img into the concatenated image
            concatenated_image[:, y_offset:y_offset + resized_img.shape[1], x_offset:x_offset + resized_img.shape[2]] = resized_img
            #append this image to the concatenated images list
            concatenated_images.append(concatenated_image)

        # Save the image stack
        merged_filename = scene + ".tif"
        merged_filepath = os.path.join(saving_path, merged_filename)
        tiff.imwrite(merged_filepath, np.array(concatenated_images), bigtiff=True, compression='deflate',
                     metadata={'axes': 'TCYX'})
        print(f"Saved merged image stack for scene {scene} to {merged_filepath}")

    return

def stack_tifs_to_timestack_density(saving_path, loading_path, scaling_factor=0.25, channel_number=3, channels=[1, 0, 2]):
    '''

    reads all density tif files in a given directory and merges the images to a timestack

    :param saving_path: path of the directory the density plots should be saved to
    :type saving_path: str

    :param loading_path: path of the directory the density plots should be loaded from
    :type loading_path: str

    '''

    image_timesteps = {}
    tiffiles = [f for f in os.listdir(loading_path) if f.endswith('.tif')]
    # Iterate over the files in the folder
    for filename in tiffiles:

        # NEW: get the "scene" from everything after "MIPs_" until ".tif"
        # Example: 20260128_d01_Plate1_MIPs_A1_1.00_6-A1.tif  -> scene = "A1_1.00_6-A1"
        scene = re.findall(r'MIPs_(.+)\.tif$', filename)[0]

        # Get the timestep identifier (e.g., 01, 02, 03, etc.)
        # Works for d01 (day) and also t01 if you ever have that again
        timestep = re.findall(r'_[dt](\d\d)_', filename)[0]

        # Check if the scene is already in the dictionary
        if scene in image_timesteps:
            # If it is, add the current filename to the corresponding timestep's list
            image_timesteps[scene][timestep] = filename
        else:
            # If it's not, create a new entry in the dictionary
            image_timesteps[scene] = {timestep: filename}

    # Iterate over the image timesteps and merge them
    for scene, timesteps in image_timesteps.items():
        print(scene)
        print(timesteps)
        # Sort the timesteps to ensure correct ordering
        sorted_timesteps = sorted(timesteps.keys())
        print(sorted_timesteps)
        #print(tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)
        max_width = max((tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)[2] for timestep in sorted_timesteps)
        print(max_width)
        total_height = max((tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)[1] for timestep in sorted_timesteps)
        print(total_height)

        concatenated_images = []
        # Paste each image into the concatenated image, center-aligned
        for timestep in sorted_timesteps:
            print(timestep)
            print(timesteps[timestep])
            img = tiff.imread(os.path.join(loading_path, timesteps[timestep]))
            print(img.shape)
            #only get three channels 0, 1, 4
            img = img[channels,:, :]
            x_offset = round(((max_width - img.shape[2]) / 2)*scaling_factor)
            y_offset = round(((total_height - img.shape[1]) / 2)*scaling_factor)
            #creating a new 3 channel image with the dimensions of the maximum image
            concatenated_image = np.zeros((channel_number, round(total_height*scaling_factor), round(max_width*scaling_factor)))
            #resize image
            resized_img = skimage.transform.rescale(img, scaling_factor, channel_axis=0, anti_aliasing=True)

            #paste the img into the concatenated image
            concatenated_image[:, y_offset:y_offset + resized_img.shape[1], x_offset:x_offset + resized_img.shape[2]] = resized_img
            #append this image to the concatenated images list
            concatenated_images.append(concatenated_image)

        # Save the image stack
        merged_filename = scene + ".tif"
        merged_filepath = os.path.join(saving_path, merged_filename)
        tiff.imwrite(merged_filepath, np.array(concatenated_images), bigtiff=True, compression='deflate',
                     metadata={'axes': 'TCYX'})
        print(f"Saved merged image stack for scene {scene} to {merged_filepath}")

    return

def stack_tifs_to_timestack_cp(saving_path, loading_path, scaling_factor=0.25, channel_number=3):
    '''

    reads all density tif files in a given directory and merges the images to a timestack

    :param saving_path: path of the directory the density plots should be saved to
    :type saving_path: str

    :param loading_path: path of the directory the density plots should be loaded from
    :type loading_path: str

    '''

    image_timesteps = {}
    tiffiles = [f for f in os.listdir(loading_path) if f.endswith('.tif')]
    # Iterate over the files in the folder
    for filename in tiffiles:
        # get the name of the tile region
        scene = re.findall(r'MIPs_([a-zA-Z0-9-_]+)\.tif', filename)[0]
        # Get the timestep identifier (e.g., 01, 02, 03, etc.)
        timestep = re.findall(r'_[dt](\d\d)_', filename)[0]

        # Check if the scene is already in the dictionary
        if scene in image_timesteps:
            # If it is, add the current filename to the corresponding timestep's list
            image_timesteps[scene][timestep] = filename
        else:
            # If it's not, create a new entry in the dictionary
            image_timesteps[scene] = {timestep: filename}
    # Iterate over the image timesteps and merge them
    for scene, timesteps in image_timesteps.items():
        print(scene)
        print(timesteps)
        # Sort the timesteps to ensure correct ordering
        sorted_timesteps = sorted(timesteps.keys())
        print(sorted_timesteps)
        #print(tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)
        max_width = max((tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)[2] for timestep in sorted_timesteps)
        print(max_width)
        total_height = max((tiff.imread(os.path.join(loading_path, timesteps[timestep])).shape)[1] for timestep in sorted_timesteps)
        print(total_height)

        concatenated_images = []
        # Paste each image into the concatenated image, center-aligned
        for timestep in sorted_timesteps:
            print(timestep)
            print(timesteps[timestep])
            img = tiff.imread(os.path.join(loading_path, timesteps[timestep]))
            print(img.shape)
            x_offset = round(((max_width - img.shape[2]) / 2)*scaling_factor)
            y_offset = round(((total_height - img.shape[1]) / 2)*scaling_factor)
            #creating a new 3 channel image with the dimensions of the maximum image
            concatenated_image = cp.zeros((channel_number, round(total_height*scaling_factor), round(max_width*scaling_factor)))
            #resize image
            resized_img = skimage.transform.rescale(img, scaling_factor, channel_axis=0, anti_aliasing=True)

            #paste the img into the concatenated image
            concatenated_image[:, y_offset:y_offset + resized_img.shape[1], x_offset:x_offset + resized_img.shape[2]] = cp.array(resized_img)
            #append this image to the concatenated images list
            concatenated_images.append(concatenated_image)

        # Save the image stack
        merged_filename = scene + ".tif"
        merged_filepath = os.path.join(saving_path, merged_filename)
        tiff.imwrite(merged_filepath, cp.asnumpy(cp.stack(concatenated_images)), bigtiff=True, compression='lzw')
        print(f"Saved merged image stack for scene {scene} to {merged_filepath}")

    return

def load_images(folder_path, scene, channel='sus'):
    """ Load all TIFF images from the specified folder. Returns a list of images and a list of timepoints"""
    images = []
    timepoints = []
    files = os.listdir(folder_path)
    files.sort()
    for file in files:
        if scene in file and channel in file:
            img_path = os.path.join(folder_path, file)
            timepoint = re.findall(r'[dt](\d\d)_', file)[0]
            with Image.open(img_path) as img:
                images.append(np.array(img))
                timepoints.append(timepoint)
    return images, timepoints

def calculate_center_of_mass(image):
    """ Calculate the center of mass of the given image. """
    center = center_of_mass(image)
    #plt.imshow(image)
    #plt.scatter(center[1], center[0], c='r', s=100)
    #plt.show()
    return center

def calculate_radial_density(image, center):
    """ Calculate the radial density profile of the image. """
    y, x = np.indices((image.shape))
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int64)

    tbin = np.bincount(r.ravel(), image.ravel())    # sum of intensity in each radius bin
    nr = np.bincount(r.ravel())                     # number of pixels in each radius bin
    radial_profile = tbin / nr
    return radial_profile

def plot_radial_density_profiles(images, channel, timepoints, scene, yscale='linear'):
    """ Plot the radial density profiles of all images. Labels are the timepoints."""
    all_profiles = []

    for image in images:
        center = calculate_center_of_mass(image)
        radial_profile = calculate_radial_density(image, center)
        all_profiles.append(radial_profile)

    if channel == 'sus':
        #use green color map
        colors = plt.cm.Greens(np.linspace(0, 1, len(all_profiles)))
    else:
        colors = plt.cm.Reds(np.linspace(0, 1, len(all_profiles)))

    fig = plt.figure(figsize=(10, 6))
    for i, (profile, color) in enumerate(zip(all_profiles, colors)):
        plt.plot(profile, color=color, label= 'day ' + str(timepoints[i]))
        plt.yscale(yscale)
        plt.legend()
        plt.xlabel('Radial Distance')
        plt.ylabel('Average Density')
        plt.title('Radial Density Profiles (' + scene + ' ' + channel + ')')
    return fig

def get_scenes_from_folder(folder_path):
    """ Get all unique scenes from the folder. """
    scenes = []
    for file in os.listdir(folder_path):
        if file.endswith(".tif"):
            scene = re.findall(r'MIPs_([a-zA-Z0-9-_]+)_density', file)[0]
            if scene not in scenes:
                scenes.append(scene)
    return scenes

def get_scenes_from_hdf5(folder_path):
    """ Get all unique scenes from the folder. """
    scenes = []
    for file in os.listdir(folder_path):
        if file.endswith(".h5"):
            scene = re.findall(r'a2a4_([a-zA-Z0-9-_]+)-', file)[0]
            if scene not in scenes:
                scenes.append(scene)
    return scenes

def get_failure_days(results_directory, failure_threshold=3, normalization_day=1, no_failure='None'):
    """ Get the day when the number of total cells exceeds the threshold. """
    #get filenames of all txt files in the directory
    filenames = [os.path.join(results_directory, file) for file in os.listdir(results_directory) if file.endswith(".txt")]
    failure_results = []


    highest_day = 0

    #find highest day value of all files
    if no_failure == 'latest':
        for file in filenames:
            results_array = np.loadtxt(file, delimiter=' ', dtype=str)
            day = max(int((re.findall(r'[dt](\d\d)', file)[0])) for file in results_array[:, 0])
            if int(day) > highest_day:
                highest_day = int(day)

    for file in filenames:
        scene = re.findall(r'a4_([a-zA-Z0-9_]+)-', file)[0]
        #load text file in array
        results_array = np.loadtxt(file, delimiter=' ', dtype=str)
        total_cells = results_array[:, 1].astype(int) + results_array[:, 2].astype(int)
        #get the lowest index, where the total number of cells exceeds the threshold for the first time
        failure_day = np.argmax(total_cells/total_cells[normalization_day] > failure_threshold)
        #get day out of index via first column in results_array
        failure_name = results_array[failure_day, 0]
        #extract the day from the failure_name
        failure_day = re.findall(r'[dt](\d\d)', failure_name)[0]
        if failure_day == '00':
            if no_failure == 'None':
                failure_day = 'NoFailure'
            elif no_failure == 'latest':
                failure_day = highest_day+1
        failure_results.append([scene, failure_day])
    return failure_results

def overexposure_correction_function(image, threshold, masks, max_density, scene, timepoint, channel):
    """ Correct cell count for not segmented overexposed regions in the image"""
    #Create a boolean array for overexposed pixels (True where image > threshold)
    overexposed_bool = image > threshold
    #Create a boolean array for the mask (True where masks > 0)
    mask_bool = masks > 0

    #We only want overexposed pixels that do NOT belong to a mask
    overexposed_outside_mask_bool = overexposed_bool & (~mask_bool)

    #Count the number of overexposed pixels outside of the mask
    overexposed_count = overexposed_outside_mask_bool.sum().compute()

    #Calculate the number of cells using the max density
    overexposed_cells = int(overexposed_count * max_density)
    print(f"    {channel}: cells {overexposed_cells}")

    #Save a TIFF visualization of the overexposed pixels
    #Convert boolean array to 8-bit (0 or 255)
    overexposed_img = overexposed_outside_mask_bool.astype(np.uint8) * 255

    # Make sure the output directory exists
    overexposed_img_saving_path = os.path.join(experiment_path, 'overexposed_images')
    os.makedirs(overexposed_img_saving_path, exist_ok=True)

    # 8. Save image as TIFF (compression optional)
    output_path = os.path.join(
        overexposed_img_saving_path, f"{scene}_{timepoint}_{channel}_overexposed_img.tif")
    tiff.imwrite(output_path, overexposed_img, compression='lzw')

    return overexposed_cells


