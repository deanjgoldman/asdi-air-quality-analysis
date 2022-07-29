import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
from matplotlib.ticker import LogFormatter 
from matplotlib import cm

import numpy as np
import pandas as pd
import geopandas as gpd
import copy
import os
import time


# open pollution dataset
fn ='CO.nc'
data_dir_silam = f"./data/silam"
cat = fn.split(".")[0]
sub_dirs_silam = os.listdir(data_dir_silam)
if "agg" in sub_dirs_silam:
    sub_dirs_silam.remove("agg")
# Load fp of aggregate pollution data, otherwise
# build aggregate pollution dataset and save
dir_agg = os.path.join(data_dir_silam, "agg")
os.makedirs(dir_agg, exist_ok=True)
fp_agg = os.path.join(dir_agg, f"pol_agg_{cat}.nc")
if os.path.exists(fp_agg):
    pol = xr.open_dataset(fp_agg)
else:
    # open an initial xr dataset to use xarray's format
    pol = xr.open_dataset(os.path.join(data_dir_silam, sub_dirs_silam[0], fn))
    pols = []
    for i in range(len(sub_dirs_silam)):
        path_pol = os.path.join(data_dir_silam, sub_dirs_silam[i], fn)
        pol = xr.open_dataset(path_pol)
        pols.append(pol)
    days = list(range(len(pols)))
    pol_agg = None
    for i in days:
        pol = pols[i]
        if pol_agg is None:
            pol_agg = pol[cat].data.mean(0)
        else:
            pol_agg += pol[cat].data.mean(0)
    # average, convert to ppm if molecular
    pol_agg /= len(days)
    pol[cat] = (('lat', 'lon'), pol_agg)
    pol.to_netcdf(fp_agg)

pol = pol.astype(np.uint8)

# compute divergence
def divergence(f):
    return np.ufunc.reduce(np.add, np.gradient(f, 15))
div = divergence(pol[cat])
pol['divergence'] = (('lat', 'lon'), div)

# open population dataset
pop = xr.open_dataset(os.path.join("./data/fb/pop.nc"))
pop = pop.sortby(["lat", "lon"])

# plot population
fig, ax = plt.subplots()
cmap_ = copy.copy(cm.Greens)
cmap_.set_under((1, 1, 1, 0.0))
ax_pop = pop['population'].fillna(0.0).plot(
    ax=ax,
    alpha=1.0,
    cmap=cmap_,
    norm=colors.PowerNorm(0.25),
    cbar_kwargs={"shrink": 0.5})

# plot pollution divergence
cmap = copy.copy(cm.bwr)
cmap.set_under("white")
norm = colors.TwoSlopeNorm(
    vmin=pol['divergence'].min(),
    vmax=pol['divergence'].max(),
    vcenter=0.0,)
ax_pol = pol['divergence'].plot(
    ax=ax,
    alpha=0.5,
    cmap=cmap,
    norm=norm,
    cbar_kwargs={"shrink": 0.5})
# plot map
wmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
wmap.plot(color="lightgrey", ax=ax, alpha=0.25)

# plot title
fig.suptitle(f"Divergence Approx. of {cat} Pollution (SILAM) and Population (Meta)")
plt.show()


