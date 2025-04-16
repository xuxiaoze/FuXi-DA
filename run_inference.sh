#!/bin/bash 
export PATH="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/xuxiaoze/miniconda3/envs/pytorch/bin:$PATH"

python inference.py \
--model_dir ../model \
--fcst_dir ../bg \
--obs_dir ../obs/data_zarr \
--output_dir ../output \
--assimilation 1 \
--eval_model final_cast_10_assim_model.pth \
--eval_step 6 \
--eval_years 2023081012 2023081012 \
