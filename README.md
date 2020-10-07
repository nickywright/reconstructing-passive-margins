# Reconstructing passive margin paleobathymetry

Code for reconstructing passive margin paleobathymetry


## Requirements:
- numpy
- xarray
- scipy
- pandas
- nlopt
- matplotlib
- scikit-image
- netCDF4
- healpy
- GMT
- PlateTectonicTools (https://github.com/EarthByte/PlateTectonicTools)
- GPlatesReconstructionModel (https://github.com/siwill22/GPlatesReconstructionModel)
- pybacktrack (https://pybacktrack.readthedocs.io/en/latest/index.html)
- ETOPO grid
- present-day sediment thickness grid


## Workflow procedure
- Specify the following files in `01_run_subsidence.py`:
    + Set the *`data_dir`* variable to the location of all your files (plate model files, ETOPO, etc.)
    + Set *`COB_terrane_file`*, `isocob_features`, and `rotation_file` to match their filenames
    + Set `etopo_file` to match your ETOPO filename (note: this should be in metres)
    + Set `sedimentthickness_file` to match your sediment thickness grid filename (note: this should be in metres)
    + Set `no_iterations`:
        * When initial testing workflow, set `no_iterations` to *1*. This will help you identify any errors quickly.
        * When running to determine the optimal stretching factor, set `no_iterations` to *20* or higher.
- Run the Python script:
      `python 01_run_subsidence.py`
    + This script will output a text file (`subsidenceinfo_[date].txt`)
- Specify the following files in `02_create_paleobathymetry.py`:
    + Set the *`data_dir`* variable to the location of all your files (plate model files, ETOPO, etc.)
    + Set *`rotation_file`* and `static_polygon_file` to match their filenames
    + Set `passivemarginfile` to match its filename. This is a polygon gpml that outlines all passive margins to be reconstructed, in case you don't want to reconstruct all regions.
    + Set `grid_spacing` to be your desired output grid resolution (i.e. 0.1 for 0.1d)
    + Set `min_time`, `max_time`, and `time_step` to be your desired output times
    + Set `source_data` to be the name of the subsidence info file from part 1 (i.e. `subsidenceinfo_[date].txt`)
    + Set `sedimentation_mode` to 'Constant' for constant sedimentation model, or anything else for the 'Keep pace' sedimentation model
    + Set `output_directory` to output directory folder name
- Run the Python script:
      `python 02_create_paleobathymetry.py`
    + This script will output reconstructed basement depth, sediment thickness, and paleobathymetry grids

## Reference
Wright, N.M., Seton, M., Williams, S.E., Whittaker, J.M. and Müller, R.D., 2020. Sea level fluctuations driven by changes in global ocean basin volume following supercontinent break-up. Earth-Science Reviews, https://doi.org/10.1016/j.earscirev.2020.103293
