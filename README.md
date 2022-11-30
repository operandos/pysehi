# pysehi
 Data processing and analysis tools for Secondary Electron Hyperspectral Imaging (SEHI) and Scanning Electron Microscope (SEM) derived Secondary Electron (SE) spectroscopy.
 
![Fig.A. SEHI data volume with regions of interest. Fig.B. SE spectra from regions of interest](https://ars.els-cdn.com/content/image/1-s2.0-S0968432822000300-gr2_lrg.jpg?raw=true "Title")
*(a) SEHI data volume with outlined regions of interest. (b) SE spectra derived from regions of interest in (a) with labelled spectral features. Reproduced from https://doi.org/10.1016/j.micron.2022.103234 under CC BY 4.0 https://creativecommons.org/licenses/by/4.0/*

## Get started
Install the pysehi environment using the anaconda prompt and pysehi.yml by following the [instructions](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)

## Compatible data file structure
pysehi function 'process_files' will only work for data saved with a compatible file tree. The format required for raw data to be processed is *...\Raw\material\YYMMDD\data_folder*. The processsed data is then saved to a *Processed* location that mirrors the *Raw* data file tree. Any number of material sub-classes can be provided in the *Raw* path eg. *...\Raw\material_class\materials_subclass\material_condition\YYMMDD\data_folder*.
