# pysehi
 Data processing and analysis tools for Secondary Electron Hyperspectral Imaging (SEHI) and Scanning Electron Microscope (SEM) derived Secondary Electron (SE) spectroscopy.
 
![Fig.A. SEHI data volume with regions of interest. Fig.B. SE spectra from regions of interest](https://ars.els-cdn.com/content/image/1-s2.0-S0968432822000300-gr2_lrg.jpg?raw=true "Title")
*(a) SEHI data volume with outlined regions of interest. (b) SE spectra derived from regions of interest in (a) with labelled spectral features. Reproduced from https://doi.org/10.1016/j.micron.2022.103234 under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*.

## Get started
### Branch or download the code
Unzip it in a new pysehi directory.
### Create a pysehi environment
Create a pysehi environment and install the dependancies using [anaconda prompt](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) and the [pysehi_env.yml](https://github.com/operandos/pysehi/blob/main/pysehi_env.yml):  
In Anaconda Prompt: 
 1) cd ...\path-to\pysehi-dir
 2) conda env create --name pysehi --file pysehi_env.yml 
### Activate pysehi
In Anaconda Prompt run:
  1) `activate sehi`
  2) `cd *...\some_path\pysehi*`  
  **note** - if pysehi is in a different drive (eg. *G:*) to the default (eg. *C:*), switch drive by running `g:` and then do (2)
  4) `jupyter notebook` to run scripts, **or** `spyder` for the development environment

## Compatible data file structure
**pysehi functions will only work for data saved with a compatible file tree.**  
The file tree required by `process_data` is *...\Raw\material\YYMMDD\data_folder*. The processsed data is then saved to a *Processed* location that mirrors the *Raw* data file tree.  
Functions `list_files` and `load` and the `data` class also require the folder (raw or processed) to be in the format *...\{Raw or Processed}\material\\...\YYMMDD\\...\data_folder*.  
Any number of sub-classes can be provided after 'material' and after 'YYMMDD' in the data folder path eg. *...\Raw\material_class\material_subclass\material_condition\YYMMDD\experiment_1\data_folder*.  

## Functions
### `process_files`
#### Outputs
**stack.tif** - Registered and cropped image stack. The SEHI data volume. Scale and slice label metadata compatible with FIJI (ImageJ).  
**avg_img.tif** - Average image of SEHI data volume along energy axis. Scale metadata compatible with FIJI (ImageJ).  
**avg_img_scaled.png** - Normalised average image with scalebar saved at 400 dpi.  
**stack_meta.json** - Dictionary structure of FEI/ThermoScientific image metadata for each slice in the SEHI data volume.  
### `load`
Provide a path to the data files to load as a pysehi object.
### `load().plot_spec`
Plot a spectrum from the whole field of view or from a region of interest if an imagJ roi file or mask array is provided.

## Citation
Please cite use of the pysehi functions and processing scripts by refering to the most recent pysehi release hosted at 10.15131/shef.data.21647084
