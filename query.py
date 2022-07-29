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
import sys
import os
import time

mol_weight = { # molecular weights pulled from google search
    "CO": 28.01,
    "NO2": 46.0055,
    "NO": 30.01,
    "O3": 48,
    "SO2": 64.066

}
health_threshold = {
    "CO": 50.0, # https://www.epa.gov/indoor-air-quality-iaq/carbon-monoxides-impact-indoor-air-quality
    "NO2": 3.0, # https://www.airnow.gov/sites/default/files/2018-06/no2.pdf
    "NO": 25.0, # https://www.cdc.gov/niosh/docs/81-123/pdfs/0448.pdf?id=10.26616/NIOSHPUB81123
    "O3": 0.075, # https://www3.epa.gov/region1/airquality/avg8hr.html
    "PM10": 54.0, # https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html
    "PM25": 35.0, # https://www.epa.gov/sites/default/files/2016-04/documents/2012_aqi_factsheet.pdf
    "SO2": 5.0 # https://www.cdc.gov/niosh/npg/npgd0575.html
}

cat = sys.argv[1]
fn = f'{cat}.nc'
data_dir_silam = f"./data/silam"
sub_dirs_silam = os.listdir(data_dir_silam)
path_pol = os.path.join(data_dir_silam, sub_dirs_silam[0], fn)
pol = xr.open_dataset(path_pol)
prev = pol[cat][0].data[..., None]
prev = prev.astype(np.uint8)


pols = []
for i in range(len(sub_dirs_silam)):
    path_pol = os.path.join(data_dir_silam, sub_dirs_silam[i], fn)
    pol = xr.open_dataset(path_pol)
    pols.append(pol)


days = list(range(len(pols)))
cur = None
for i in days:
    pol = pols[i]
    if cur is None:
        cur = pol[cat].data.mean(0)
    else:
        cur += pol[cat].data.mean(0)

# average, convert to ppm if molecular
cur /= len(days)
if cat in mol_weight.keys():
    cur = ((cur * 24.45) / mol_weight[cat]) * 1e-3
    units = "ppm"
else:
    units = "ug/m3"

pop = xr.open_dataset(os.path.join("./data/fb/pop_reg.nc"))
pop = pop.sortby(["lat", "lon"])
pop = pop['population'].data

pop_sum = pop.sum()
nbins = 50
bins = np.linspace(cur.min(), cur.max(), nbins)
bins = np.round(bins, 2)
data = np.zeros(nbins)
for i, t in enumerate(bins):
    data[i] = pop[cur >= t].sum() / pop_sum

fig, ax = plt.subplots()
ax.bar(np.arange(len(data)), data)
# axis labels
ax.set_xlabel(f"{cat} concentration ({units})")
ax.set_ylabel(f"Proportion of total population")
# xticks
ax.set_xticks(np.arange(0, nbins, 5))
ax.set_xticklabels(bins[::5], rotation=45)
# yticks
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.set_yticklabels(np.round(np.arange(0, 1.1, 0.1), 1))
# healthy threshold lines
thresh = health_threshold[cat]
xpos = np.argmin(np.abs(bins - thresh))
ypos = data[xpos]
ax.axvline(xpos, color='gray', linestyle='--')
ax.axhline(ypos, color='gray', linestyle='--')

note_threshold = "Recommended\nthreshold: "+str(thresh)
note_estimate = "Estimated pop.\naffected: "+str(int(np.round(ypos * pop_sum, 0)))

ax.annotate(note_threshold, (xpos, 0.5), bbox={"boxstyle": "square", "fc": "cyan"})
ax.annotate(note_estimate, (xpos, ypos+0.1), bbox={"boxstyle": "square", "fc": "magenta"})
ax.grid()

title = f"Cumulative Frequency Distribution of Population with Pollution ({cat}) {units}"
title += "\n(y-axis represents proportion of pop. with {units} >= x-axis coordinate)"
fig.suptitle(title)
plt.show()
plt.close('all')

