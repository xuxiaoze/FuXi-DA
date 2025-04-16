# FuXi-DA
This is the official repository for the FuXi-DA paper.

FuXi-DA: a generalized deep learning data assimilation framework for assimilating satellite observations

Published on npj Climate and Atmospheric Science: FuXi-DA: a generalized deep learning data assimilation framework for assimilating satellite observations

by Xiaoze Xu, Xiuyu Sun, Wei Han, Xiaohui Zhong, Lei Chen, Zhiqiu Gao & Hao Li

# Installation
The downloaded files shall be organized as the following hierarchy:
|-- code_inference
|   |-- inference.py
|   |-- read_data.py
|   |-- result_plot.py
|   `-- run_inference.sh
|-- model
|   |-- assimilation_v6.py
|   `-- final_cast_10_assim_model.pth
|-- output
|   `-- 2023081012.nc
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
