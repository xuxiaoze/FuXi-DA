import argparse
from collections import OrderedDict
import torch
import pandas as pd
import xarray as xr
import numpy as np
import sys
from tqdm import tqdm
from read_data import read_obs, read_fcst


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--fcst_dir', type=str, required=True)
parser.add_argument('--obs_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--assimilation', type=int, default=0)
parser.add_argument('--eval_model', type=str, required=True)
parser.add_argument('--eval_step', type=int, default=6)
parser.add_argument('--eval_years', type=str, nargs="+", default=['2023100100', '2023100100'])
args = parser.parse_args()

device = 'cpu'
dtype = torch.float32


def load_checkpoint(checkpoint_path, model):
    if hasattr(model, 'module'):
        model = model.module
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_dict = model.state_dict()
    check_ = OrderedDict()
    for num, (k, v) in enumerate(model_dict.items()):
        check_[k] = checkpoint['model'][f"assimilation.{k}"]
    model.load_state_dict(check_)
    return model


def chunk_time(ds):
    dims = {k: v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds


def get_date(start_str, end_str, step):
    start_time = pd.to_datetime(start_str, format="%Y%m%d%H")
    end_time = pd.to_datetime(end_str, format="%Y%m%d%H")
    dates = pd.date_range(start_time, end_time, freq=f"{int(step)}h")
    return dates


def inference(model, bg, obs):
    bg = bg.unsqueeze(0).unsqueeze(0)
    obs = obs.unsqueeze(0)
    with torch.no_grad():
        data_out = model(bg, obs)
    return data_out.cpu().detach().numpy()[0]


def save_nc(ds, save_name, dtype=np.float32):
    from dask.diagnostics import ProgressBar
    ds = chunk_time(ds)
    ds = ds.astype(dtype)
    delayed_ds = ds.to_netcdf(save_name, compute=False)
    with ProgressBar():
        delayed_ds.compute()


def save_out(date, output, output_dir, latlon_dir):
    channel = ['z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500',
               'z600', 'z700', 'z850', 'z925', 'z1000', 't50', 't100', 't150',
               't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850',
               't925', 't1000', 'u50', 'u100', 'u150', 'u200', 'u250', 'u300',
               'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000', 'v50',
               'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600',
               'v700', 'v850', 'v925', 'v1000', 'r50', 'r100', 'r150', 'r200',
               'r250', 'r300', 'r400', 'r500', 'r600', 'r700', 'r850', 'r925',
               'r1000', 't2m', 'u10', 'v10', 'msl', 'tp']
    lat = np.load(f'{latlon_dir}/latitude_era5.npy')[:, 0]
    lon = np.load(f'{latlon_dir}/longitude_era5.npy')[0, :]
    ds_ana = xr.Dataset({'z': (['time', 'channel', 'lat', 'lon'], output.astype(np.float32))},
                        coords={'time': [date], 'channel': channel, 'lat': lat, 'lon': lon})
    ds_ana['time'].encoding['dtype'] = 'float64'
    date_str = date.strftime("%Y%m%d%H")
    ds_ana = chunk_time(ds_ana)
    save_nc(ds_ana, f"{output_dir}/{date_str}.nc", dtype=np.float32)
    return None


def main():
    sys.path.append(args.model_dir)
    from assimilation_v6 import AssimilationNetv6
    model = AssimilationNetv6(bg_chans=70, 
                              embed_dim=256, 
                              obs_chans=15, 
                              obs_frames=8, 
                              depth=(1, 1, 1), 
                              obs_rect=(40, 680, 210, 850))

    model = model.to(dtype=dtype, device=device)
    check_model = f"{args.model_dir}/{args.eval_model}"
    model = load_checkpoint(check_model, model)
    dates = get_date(start_str=args.eval_years[0], end_str=args.eval_years[1], step=args.eval_step)

    dataset_bg = read_fcst(data_name=args.fcst_dir)
    dataset_obs = read_obs(data_name=args.obs_dir)
    for date in tqdm(dates):
        data_fcst = dataset_bg.prepare_data(date)
        data_obs = dataset_obs.prepare_data(date)
        data_fcst = data_fcst.to(dtype=dtype, device=device)
        data_obs = data_obs.to(dtype=dtype, device=device)
        data_out = inference(model=model, bg=data_fcst, obs=data_obs)
        save_out(date=date, output=data_out, output_dir=args.output_dir, latlon_dir=args.fcst_dir)


if __name__ == "__main__":
    main()
