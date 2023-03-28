# pysehi
 Data processing and analysis tools for Secondary Electron Hyperspectral Imaging (SEHI) and Scanning Electron Microscope (SEM) derived Secondary Electron (SE) spectroscopy.
 
![Fig.A. SEHI data volume with regions of interest. Fig.B. SE spectra from regions of interest](https://ars.els-cdn.com/content/image/1-s2.0-S0968432822000300-gr2_lrg.jpg?raw=true "Title")
*(a) SEHI data volume with outlined regions of interest. (b) SE spectra derived from regions of interest in (a) with labelled spectral features. Reproduced from https://doi.org/10.1016/j.micron.2022.103234 under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*.

## Get started
### Branch or download the code
Unpack it in a new pysehi-project folder.
### Create a pysehi environment
Once [miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) is installed, open the terminal and use the [pysehi_manual_build.yml](https://github.com/operandos/pysehi/blob/main/pysehi_manual_build.yml) file to make a pysehi environment.  

In the terminal run: 
 1) ```cd ...\path-to\pysehi-project```  
  **note** - if pysehi is in a different drive (eg. *G:*) to the default (eg. *C:*), switch drive by running `g:` and then do (1)
 2) ```conda env create --name pysehi --file pysehi_manual_build.yml```
### Activate pysehi
In the terminal run:
  1) `cd ...\path-to\pysehi-project`
  2) `conda activate pysehi`
  3) `jupyter notebook` to run scripts, **or** `spyder` for the development environment

## Citation
Please cite **https://doi.org/10.15131/shef.data.22310068.v1** and the most recent release tag when you have used pysehi. For example:  
> Nohl, James F. (2023), pysehi releases version 1.1.1, The University of Sheffield, Software, https://doi.org/10.15131/shef.data.22310068.v1

The packaged releases are hosted on FigShare at *link to published figshare when published*

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

## Compatible data file structure
**pysehi functions will only work for data saved with a compatible file tree.**  
The file tree required by `process_data` is *...\Raw\material\YYMMDD\data_folder*. The processsed data is then saved to a *Processed* location that mirrors the *Raw* data file tree.  
Functions `list_files` and `load` and the `data` class also require the folder (raw or processed) to be in the format *...\{Raw or Processed}\material\\...\YYMMDD\\...\data_folder*.  
Any number of sub-classes can be provided after 'material' and after 'YYMMDD' in the data folder path eg. *...\Raw\material_class\material_subclass\material_condition\YYMMDD\experiment_1\data_folder*.  
