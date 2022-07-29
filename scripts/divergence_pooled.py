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

from numpy.lib.stride_tricks import as_strided


def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
    
    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))


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

fn ='CO.nc'
data_dir_silam = f"./data/silam"
cat = fn.split(".")[0]
sub_dirs_silam = os.listdir(data_dir_silam)
path_pol = os.path.join(data_dir_silam, sub_dirs_silam[0], fn)
pol = xr.open_dataset(path_pol)
prev = pol[cat][0].data[..., None]

prev = minmax(prev) * 255
prev = prev.astype(np.uint8)

color = np.random.randint(0, 255, (100000, 3))
mask = np.zeros_like(prev)

pols = []
for i in range(len(sub_dirs_silam)):
    path_pol = os.path.join(data_dir_silam, sub_dirs_silam[i], fn)
    pol = xr.open_dataset(path_pol)
    pols.append(pol)

def divergence(f):
    return np.ufunc.reduce(np.add, np.gradient(f, 15))


days = list(range(len(pols)))
cur = None
for i in days:
    pol = pols[i]
    if cur is None:
        cur = pol[cat].data.mean(0)
    else:
        cur += pol[cat].data.mean(0)

cur /= len(days)
cur = cur.astype(np.uint8)
cur = pool2d(cur, kernel_size=3, stride=1, padding=0, pool_mode='avg')
g = divergence(cur)

pop = xr.open_dataset(os.path.join("./data/fb/pop_reg.nc"))
pop = pop.sortby(["lat", "lon"])
p = pop['population'].fillna(0.0).data
p = pool2d(p, kernel_size=3, stride=1, padding=0, pool_mode='avg')

wmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

fig, ax = plt.subplots()
cmap_ = copy.copy(cm.Blues)
cmap_.set_under((1, 1, 1, 0.0))
ax.matshow(
    np.flipud(p),
    alpha=1.0,
    cmap=cmap_,
    norm=colors.PowerNorm(0.25))

cmap = cm.CMRmap.reversed()
cmap.set_under("white")
bounds = np.concatenate([
    np.arange(1, 20, 1),
    np.arange(20, g.max(), 10)]) 
norm = colors.BoundaryNorm(bounds, ncolors=len(bounds))
ax.matshow(
    np.flipud(g),
    alpha=0.5,
    cmap=cmap,
    norm=norm)

# wmap.plot(color="lightgrey", ax=ax, alpha=0.25)

fig.suptitle(f"Divergence Approx. of {cat} Pollution (SILAM) and Population (Meta)")
plt.show()


