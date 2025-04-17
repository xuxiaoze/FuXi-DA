# FuXi-DA
This repository provides an example codebase for running the FuXi-DA model.

Published in npj Climate and Atmospheric Science: FuXi-DA: a generalized deep learning data assimilation framework for assimilating satellite observations
by Xiaoze Xu, Xiuyu Sun, Wei Han, Xiaohui Zhong, Lei Chen, Zhiqiu Gao & Hao Li.

For ERA5 data preprocessing and FuXi model forecasting, please refer to the original FuXi repository at https://github.com/tpys/FuXi.

Note: The background and analysis fields used in this example have been standardized and interpolated to 720 latitude points. The reverse standardization and restoration to the original 721-point latitude grid can be found in the ``` result_plot.py ``` script.

# Installation
The downloaded files shall be organized as the following hierarchy:
```text 
|-- code_inference
|   |-- inference.py
|   |-- read_data.py
|   |-- result_plot.py
|   `-- run_inference.sh
|-- model
|   |-- assimilation_v6.py
|   `-- final_cast_10_assim_model.pth
|-- output
|   `-- 2023081012_refer.nc
`-- test_data
    |-- bg
    |   |-- 2023081012.nc
    |   |-- latitude_era5.npy
    |   |-- longitude_era5.npy
    |   |-- mean_era5.npy
    |   `-- std_era5.npy
    `-- obs
        `-- data_zarr
            |-- SatelliteZenith_gridmean.npy
            |-- channel
            |-- lat
            |-- lon
            |-- mean_agri_gridmean.npy
            |-- std_agri_gridmean.npy
            |-- time
            `-- x
```
The assimilation result is saved as:``` output/2023081012.nc ```. 
For reference, the corresponding baseline assimilation result is provided in:``` output/2023081012_refer.nc ```. 

This code requires the following Python libraries:
```text 
torch, xarray, dask, netCDF4, zarr
```

# Demo
```text 
bash run_inference.sh
```
