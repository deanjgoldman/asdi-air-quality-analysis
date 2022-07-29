import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# from matplotlib.image import NonUniformImage
from matplotlib.ticker import LogFormatter 
from matplotlib import cm

import numpy as np
import pandas as pd
import geopandas as gpd

import iso3166

import os
import time

import sys; sys.path.append("..")
import settings

#################
# User settings #
#################
# population
country = "USA"
data_dir_pop = settings.data_dir_pop_reg
pop = xr.open_dataset(os.path.join(data_dir_pop, f"pop_{country}.nc"))
pop = pop['population'].data.flatten()
pop_min = 0
mask = [pop > pop_min]
# air quality
fn ='PM25.nc'
data_dir_pol = settings.data_dir_pol
cat = fn.split(".")[0]
i = 0
sub_dirs_pol = os.listdir(data_dir_pol)
if "agg" in sub_dirs_pol:
    sub_dirs_pol.remove("agg")
path_pol = os.path.join(data_dir_pol, sub_dirs_pol[i], fn)
pol = xr.open_dataset(path_pol)
pol = pol[cat].mean('time').data.flatten()
# remove where pop below threshold
pop = pop[mask]
pol = pol[mask]

#############
# Histogram #
#############
# create histogram bins
pop_bins = np.concatenate([
    np.arange(pop_min, 1e+3, 1e+3),
    np.arange(1e+3, 1e+4, 1e+3),
    np.arange(1e+5, 1e+6, 1e+5)])
pol_bins = np.concatenate([
    np.arange(5, 40, 5),
    np.arange(40, 100, 10),
    np.arange(100, 600, 100)])

# qcx = pd.cut(pop, bins=pop_bins, duplicates='drop')
# qcy = pd.cut(pol, bins=pol_bins, duplicates='drop')
qcx = pd.qcut(pop, q=25, duplicates='drop')
qcy = pd.qcut(pol, q=25, duplicates='drop')


edges_x = [el.left for el in qcx.categories.to_numpy()]
edges_y = [el.left for el in qcy.categories.to_numpy()] 

hist, ex, ey = np.histogram2d(
    x=pop,
    y=pol,
    bins=(edges_x, edges_y))

cmap = plt.get_cmap('jet')
cmap.set_under('white')

fig, ax = plt.subplots()
im = ax.imshow(
    hist,
    cmap=cmap,
    norm=colors.LogNorm(),
    aspect='auto',
    extent=[0, len(ex), 0, len(ey)])

ax.set_xlabel("Population", fontsize=10)
ax.set_xticks(np.arange(len(ex)))
ax.set_xticklabels([int(el) for el in ex], fontsize=8, rotation=45)
ax.set_ylabel(f"Pollution: {cat}", fontsize=10)
ax.set_yticks(np.arange(len(ey)))
ax.set_yticklabels(np.round(ey, 2), fontsize=8)

country_name = iso3166.countries_by_alpha3[country].name
title = f"{country_name} Population vs. Pol ({cat}) Histogram"
ax.set_title(title, fontsize=10)
formatter = LogFormatter(10, labelOnlyBase=False)
fig.colorbar(im, ax=ax, format=formatter)
plt.show()
fig.savefig(os.path.join(settings.dir_output, f"hist_{country}.png"))
plt.close("all")
import pdb; pdb.set_trace();



