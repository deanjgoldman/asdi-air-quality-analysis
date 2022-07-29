import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
from matplotlib.ticker import LogFormatter 
from matplotlib import cm

import numpy as np
import pandas as pd
import geopandas as gpd

import os
import time


# params for ShiTomasi corner detection
feature_params = dict(
    maxCorners=150,
    qualityLevel=0.2,
    minDistance=5,
    blockSize=50)

# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize=(25, 25),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.03))

minmax = lambda x: (x - x.min()) / (x.max() - x.min())

fn ='PM25.nc'
data_dir_silam = f"./data/silam"
cat = fn.split(".")[0]
sub_dirs_silam = os.listdir(data_dir_silam)
path_pol = os.path.join(data_dir_silam, sub_dirs_silam[0], fn)
pol = xr.open_dataset(path_pol)
prev = pol[cat][0].data[..., None]

prev = minmax(prev) * 255
prev = prev.astype(np.uint8)

color = np.random.randint(0, 255, (1000000, 3))
mask = np.zeros_like(prev)

p0 = np.argwhere(prev > 100)[:, None, :-1]
p0 = p0[:, :, ::-1].astype(np.float32)

# p01 = cv2.goodFeaturesToTrack(prev, **feature_params).astype(np.float32)
pols = []
for i in range(len(sub_dirs_silam)):
    path_pol = os.path.join(data_dir_silam, sub_dirs_silam[i], fn)
    pol = xr.open_dataset(path_pol)
    pols.append(pol)
    
# import pdb; pdb.set_trace();
for i in range(len(pols)):
    pol = pols[i]
    # cv2.imshow('frame', pol[cat][0].data[..., None])
    for j in range(pol[cat].data.shape[0]):
        cur = pol[cat][j].data[..., None]

        cur = minmax(cur) * 255
        cur = cur.astype(np.uint8)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev, cur, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1].astype(np.float32)
        good_old = p0[st==1].astype(np.float32)

        # draw the tracks
        for k,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel().astype(np.uint32)
            c,d = old.ravel().astype(np.uint32)
            mask = cv2.line(mask, (a,b), (c,d), color[k].tolist(), thickness=1)
            frame = cv2.circle(cur, (a,b), 1, color[k].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        prev = cur.copy()
        p0 = good_new.reshape(-1,1,2)

