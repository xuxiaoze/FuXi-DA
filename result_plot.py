import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker


def reverse_standardization(ds, standard_dir):
    mean = np.load(f"{standard_dir}/mean_era5.npy")
    std = np.load(f"{standard_dir}/std_era5.npy")
    ds = (ds * std + mean)
    ds_tp = ds.sel(channel='tp')
    ds_tp = np.exp(ds_tp) - 1
    ds.loc[{'channel': 'tp'}] = ds_tp
    return ds


def process_output(ds, hw=(721, 1440)):
    output = torch.from_numpy(ds.z.values)
    output = F.interpolate(
        output.float(),
        size=hw,
        mode="bilinear",
        align_corners=False)
    output = output.numpy()
    ds_output = xr.Dataset({'z': (['time', 'channel', 'lat', 'lon'], output.astype(np.float32))},
                           coords={'time': ds.time,
                                   'channel': ds.channel,
                                   'lat': np.linspace(90, -90, 721),
                                   'lon': np.linspace(0, 359.75, 1440)})
    return ds_output


def add_bottom_cax(ax1, ax2, pad, width):
    axpos1 = ax1.get_position()
    axpos2 = ax2.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(axpos1.x0,
                                              axpos1.y0 - pad,
                                              axpos2.x1,
                                              axpos2.y0 - pad + width
                                              )
    cax = ax1.figure.add_axes(caxpos)
    return cax


def add_right_cax(ax1, ax2, pad, width):
    axpos1 = ax1.get_position()
    axpos2 = ax2.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(axpos2.x1 + pad,
                                              axpos2.y0,
                                              axpos1.x1 + pad + width,
                                              axpos1.y1
                                              )
    cax = ax1.figure.add_axes(caxpos)
    return cax


def add_bar(ax1, ax2, cmap, norm, dir, ticks, extend, title, pad, width):
    if dir == 'horizontal':
        cax = add_bottom_cax(ax1, ax2, pad=pad, width=width)
    elif dir == 'vertical':
        cax = add_right_cax(ax1, ax2, pad=pad, width=width)
    if ticks is not None:
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                            cax=cax, orientation=dir, extend=extend,
                            ticks=ticks)
    else:
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                            cax=cax, orientation=dir, extend=extend)
    cbar.ax.tick_params(labelsize=8)
    font = {'family': 'Times New Roman', 'color': 'k', 'weight': 'bold', 'size': 8}
    cbar.set_label(title, fontdict=font)
    return None


def plot_pcolormesh(lat, lon, data, vmin, vmax, save_name, Extent=[0, 360, -90, 90]):
    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(16, 8), dpi=300)
    plt.rcParams.update({'font.family': 'Times New Roman',
                         'font.size': 10,
                         'font.weight': 'bold',
                         'xtick.direction': 'out',
                         'ytick.direction': 'out'})
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))
    WestLon, EastLon, SouthLat, NorthLat = Extent
    ax.set_extent(Extent, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(WestLon - 360, EastLon + 1, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(SouthLat, NorthLat + 1, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.tick_params(axis='y', direction='in', which='major',
                    length=5, width=2, colors='k',
                    grid_color='k', grid_alpha=0.5)
    ax.tick_params(axis='x', direction='in', which='major',
                    length=5, width=2, colors='k',
                    grid_color='k', grid_alpha=0.5)
    ax.pcolormesh(lon, lat, data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), zorder=2)
    add_bar(ax, ax, cmap, norm, 'horizontal', None, 'neither', ' ', 0.05, 0.015)
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    return None


if __name__ == '__main__':
    standard_dir = '../bg'
    data_dir = '../output'
    save_dir = '../output'
    plot_channel = 't2m'
    vmin = 280
    vmax = 320
    date_str = '2023081012'
    
    ds = xr.open_dataset(f'{data_dir}/{date_str}.nc')
    ds = reverse_standardization(ds=ds, standard_dir=standard_dir)
    ds = process_output(ds=ds, hw=(721, 1440))
    plot_data = ds.sel(channel=plot_channel).z.values[0]
    lat = ds.lat.values
    lon = ds.lon.values
    plot_pcolormesh(lat=lat, 
                    lon=lon, 
                    data=plot_data, 
                    vmin=vmin, 
                    vmax=vmax, 
                    save_name=f"{save_dir}/{date_str}_{plot_channel}.png", 
                    Extent=[0, 360, -90, 90])