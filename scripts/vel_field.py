import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
from matplotlib.ticker import LogFormatter 
from matplotlib import cm
from scipy.signal import correlate

import numpy as np
import pandas as pd
import geopandas as gpd

import os
import time

wmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# params for ShiTomasi corner detection
feature_params = dict(
    maxCorners=150,
    qualityLevel=0.2,
    minDistance=5,
    blockSize=50)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

minmax = lambda x: (x - x.min()) / (x.max() - x.min())

fn ='PM25.nc'
data_dir_silam = f"./data/silam"
cat = fn.split(".")[0]
sub_dirs_silam = os.listdir(data_dir_silam)
path_pol = os.path.join(data_dir_silam, sub_dirs_silam[0], fn)
pol = xr.open_dataset(path_pol)
prev = pol[cat][0].data#[..., None]

prev = minmax(prev) * 255
prev = prev.astype(np.uint8)

color = np.random.randint(0, 255, (100000, 3))
mask = np.zeros_like(prev)

pols = []
for i in range(len(sub_dirs_silam)):
    path_pol = os.path.join(data_dir_silam, sub_dirs_silam[i], fn)
    pol = xr.open_dataset(path_pol)
    pols.append(pol)
    

def vel_field(curr_frame, next_frame, win_size):
    ys = np.arange(0, curr_frame.shape[0], win_size)
    xs = np.arange(0, curr_frame.shape[1], win_size)
    dys = np.zeros((len(ys), len(xs)))
    dxs = np.zeros((len(ys), len(xs)))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            int_win = curr_frame[y : y + win_size, x : x + win_size]
            search_win = next_frame[y : y + win_size, x : x + win_size]
            cross_corr = correlate(
                search_win - search_win.mean(), int_win - int_win.mean(), method="fft"
            )
            dys[iy, ix], dxs[iy, ix] = (
                np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
                - np.array([win_size, win_size])
                + 1
            )
    # draw velocity vectors from the center of each window
    ys = ys + win_size / 2
    xs = xs + win_size / 2
    return xs, ys, dxs, dys

# traverse backward through time
days = list(range(len(pols)))[::-1]
hours = list(range(24))[::-1]
for i in days:
    pol = pols[i]
    
    for j in hours:
        cur = pol[cat][j].data#[..., None]

        cur = minmax(cur) * 255
        cur = cur.astype(np.uint8)

        xs, ys, dxs, dys = vel_field(cur, prev, 32)
        import pdb; pdb.set_trace();
        norm_drs = np.sqrt(dxs ** 2 + dys ** 2)

        fig, ax = plt.subplots(figsize=(6, 6))
        # we need these flips on y since quiver uses a bottom-left origin, while our
        # arrays use a top-right origin
        ax.quiver(
            xs,
            ys[::-1],
            dxs,
            -dys,
            norm_drs,
            cmap=plt.cm.jet,
            angles="xy",
            scale_units="xy",
            scale=1.0,
        )
        # wmap.plot(color="lightgrey", ax=ax, alpha=0.25)
        # ax.set_aspect("equal")
        plt.show()

        # Now update the previous frame and previous points
        prev = cur.copy()
        print(i, j)

