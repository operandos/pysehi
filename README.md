# pysehi
 Data processing and analysis tools for Secondary Electron Hyperspectral Imaging (SEHI) and Scanning Electron Microscope (SEM) derived Secondary Electron (SE) spectroscopy.
 
![Fig.A. SEHI data volume with regions of interest. Fig.B. SE spectra from regions of interest](https://ars.els-cdn.com/content/image/1-s2.0-S0968432822000300-gr2_lrg.jpg?raw=true "Title")
*(a) SEHI data volume with outlined regions of interest. (b) SE spectra derived from regions of interest in (a) with labelled spectral features. Reproduced from https://doi.org/10.1016/j.micron.2022.103234 under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*.

## Get started
Create a pysehi environment and install the dependancies using [anaconda prompt](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) and either:  
 - **pysehi_dependencies.yml** (with Jupyter notebook to run processing and analysis scripts)  
 - **pysehi_dependencies_spyder.yml** (with spyder IDE, for development) 
 In Anaconda Prompt run
  1) `activate sehi`
  2) `cd *path_to_pysehi_folder*`
  3) `jupyter notebook` to run scripts, or `spyder` for the development environment

## Compatible data file structure
**pysehi functions will only work for data saved with a compatible file tree.**  
The file tree required by `process_data` is *...\Raw\material\YYMMDD\data_folder*. The processsed data is then saved to a *Processed* location that mirrors the *Raw* data file tree.  
Functions `list_files` and `load` and the `data` class also require the folder (raw or processed) to be in the format *...\{Raw or Processed}\material\\...\YYMMDD\\...\data_folder*.  
Any number of sub-classes can be provided after 'material' and after 'YYMMDD' in the data folder path eg. *...\Raw\material_class\material_subclass\material_condition\YYMMDD\experiment_1\data_folder*.  

## Outputs of `process_files`
**stack.tif** - Registered and cropped image stack. The SEHI data volume. Scale and slice label metadata compatible with FIJI (ImageJ).  
**avg_img.tif** - Average image of SEHI data volume along energy axis. Scale metadata compatible with FIJI (ImageJ).  
**avg_img_scaled.png** - Normalised average image with scalebar saved at 400 dpi.  
**stack_meta.json** - Dictionary structure of FEI/ThermoScientific image metadata for each slice in the SEHI data volume.  
