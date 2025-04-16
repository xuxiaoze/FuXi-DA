import numpy as np
import pandas as pd
import torch
import xarray as xr


def chunk_time(ds):
    dims = {k: v for k, v in ds.dims.items()}
    dims['time'] = 1
    ds = ds.chunk(dims)
    return ds


class read_obs:
    def __init__(
            self,
            data_name,
            time_range=(-1, 1),
            obs_step=15,
            rect=(4, 644, 6, 646),
            **kwargs
    ):
        # 观测数据
        self.data_name = data_name
        ds = xr.open_zarr(data_name, consolidated=True)
        ds = chunk_time(ds)
        self.dataset = ds

        satellite_zenith = np.load(f"{data_name}/SatelliteZenith_gridmean.npy")
        satellite_zenith = torch.from_numpy(satellite_zenith)
        satellite_zenith = torch.nan_to_num(satellite_zenith)

        satellite_zenith = satellite_zenith.unsqueeze(0)
        satellite_zenith = torch.cos(satellite_zenith / 180. * 3.1415)
        
        agri_lat = ds.lat.values
        agri_lat = torch.from_numpy(agri_lat)
        agri_lat = torch.nan_to_num(agri_lat)
        agri_lat = torch.sin(agri_lat / 180. * 3.1415)
        agri_lat = agri_lat.unsqueeze(-1).repeat(1, satellite_zenith.shape[-1]).unsqueeze(0)
        agri_lon = ds.lon.values
        agri_lon = torch.from_numpy(agri_lon)
        agri_lon = torch.nan_to_num(agri_lon)
        agri_lon = torch.cos(agri_lon / 180. * 3.1415).unsqueeze(0).repeat(satellite_zenith.shape[-2], 1).unsqueeze(0)

        self.time_range = time_range
        self.obs_step = obs_step
        self.rect = rect
        self.const_encode = torch.concat([satellite_zenith, agri_lat, agri_lon], dim=0).unsqueeze(0)
        if self.rect is not None:
            self.const_encode = self.const_encode[:, :, self.rect[0]:self.rect[1], self.rect[2]:self.rect[3]]

        self.obs_mean = np.load(f'{data_name}/mean_agri_gridmean.npy')
        self.obs_mean = torch.from_numpy(self.obs_mean)
        self.obs_mean = torch.nan_to_num(self.obs_mean)

        self.obs_std = np.load(f'{data_name}/std_agri_gridmean.npy')
        self.obs_std = torch.from_numpy(self.obs_std)
        self.obs_std = torch.nan_to_num(self.obs_std)

    def times_encoding(self, times):
        times = [pd.to_datetime(t) for t in times]
        encode = np.array([(np.cos(t.dayofyear / 366 * 2 * 3.1415),
                            np.sin(t.dayofyear / 366 * 2 * 3.1415),
                            np.cos((t.hour * 60 + t.minute) / 60 / 24 * 2 * 3.1415),
                            np.sin((t.hour * 60 + t.minute) / 60 / 24 * 2 * 3.1415)) for t in times])
        encode = torch.from_numpy(encode)
        encode = torch.nan_to_num(encode)
        encode = encode.unsqueeze(-1).unsqueeze(-1)
        return encode

    def prepare_data(self, assim_time):
        start_time_obs = assim_time + pd.Timedelta(hours=self.time_range[0])
        end_time_obs = assim_time + pd.Timedelta(hours=self.time_range[1])

        times = np.array([t for t in pd.to_datetime(self.dataset.time.values) if (t >= start_time_obs and t < end_time_obs)])

        _, _, obs_h, obs_w = self.const_encode.shape
        time_encode = self.times_encoding(times)
        time_encode = time_encode.repeat(1, 1, obs_h, obs_w)

        if len(times) == 0:  # no obs data in this time window
            print(f'assim_time {assim_time} has no obs')
            return None

        obs_data = torch.from_numpy(self.dataset.sel(time=times).x.values)
        obs_mask = 1 - torch.isnan(obs_data).int()
        obs_data = torch.nan_to_num(obs_data)
        obs_data = obs_data[:, :-1]
        obs_mean = self.obs_mean[:-1]
        obs_std = self.obs_std[:-1]
        obs_mask = obs_mask[:, :-1]
        obs_data = (obs_data - obs_mean) / obs_std * obs_mask

        if self.rect is not None:
            obs_data = obs_data[:, :, self.rect[0]:self.rect[1], self.rect[2]:self.rect[3]]

        obs_t = obs_data.shape[0]
        const_encode = self.const_encode.repeat(obs_t, 1, 1, 1)
        obs_data = torch.concat([obs_data, const_encode, time_encode], dim=1)
        idx = 0
        obs_idx = 0
        obs_num = (self.time_range[1] - self.time_range[0]) * 4
        obs = torch.zeros((obs_num, obs_data.shape[1], obs_data.shape[2], obs_data.shape[3]))
        while start_time_obs < end_time_obs:
            if start_time_obs in times:
                obs[idx, :] = obs_data[obs_idx, :]
                obs_idx = obs_idx + 1
            start_time_obs = start_time_obs + pd.Timedelta(minutes=self.obs_step)
            idx = idx + 1
        return obs


class read_fcst:
    def __init__(
            self,
            data_name,
            **kwargs
    ):
        self.data_name = data_name

    def prepare_data(self, assim_time):
        date_str = assim_time.strftime("%Y%m%d%H")
        ds = xr.open_dataset(f"{self.data_name}/{date_str}.nc")
        bg_data = torch.from_numpy(ds.sel(time=assim_time).z.values)
        bg_data = torch.nan_to_num(bg_data)
        return bg_data