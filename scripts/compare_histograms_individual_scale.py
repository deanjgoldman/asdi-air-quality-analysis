import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter 
from matplotlib import cm

import iso3166

import copy
import os
import time

#################
# User settings #
#################
data_dir = "./data"
pop_settings = {
    "pop_min": 1,              # filter out values less than pop_min 
    "countries": [             # countries to compare
        "USA",
        "SAU",]
}
pol_settings = {
    "categories": [            # pollution categories to compare
        "PM25",
        "CO",
        "SO3",
        ],              
    "date_start": "20220701",  # aggregate pollution date start
    "date_end": "20220705",    # aggregate pollution date start
}

lab_countries = "_".join(pop_settings["countries"])
lab_categories = "_".join(pol_settings["categories"])
plot_save_fp = f'./hist_{lab_countries}_{lab_categories}'


# These are the default dimensions that SILAM produces its
# estimates with. Just keeping them as global variables. 
dims = [897, 1800]

##########################
# Histogram bin settings #
########################## 
# These can be either numpy arrays specifying edges,
# or integers which will produce that many equally spaced bins.
pol_bins = 25 
pop_bins = 25

################# 
# plot settings #
#################
ncountries = len(pop_settings['countries'])
ncategories = len(pol_settings['categories'])
nrows = ncountries
ncols = ncategories 
fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(30, 30),
    constrained_layout=True)

##############################
# Load data, plot histograms #
##############################

data_dir_fb = os.path.join(data_dir, "fb", "registered")
data_dir_silam = os.path.join(data_dir, "silam")
sub_dirs_silam = os.listdir(data_dir_silam)
sub_dirs_sorted = sorted([int(sub_dir) for sub_dir in sub_dirs_silam])

def load_population_xr_array(country):
    """
    Loads netcdf of population data, returns a flattened numpy array.
    
    arguments 
        -- country: string, three character signifier of country, see
                    <data_dir_fb> for available countries. 
    """
    pop = xr.open_dataset(os.path.join(data_dir_fb, f"pop_{country}.nc"))
    pop = pop['population'].data.flatten()
    return pop

def load_pollution_xr_array(cats, date_start, date_end):
    """
    Loads netcdf of pollution data, returns a flattened numpy array.
    
    arguments 
        -- cat: string, category of pollution type. Can be one of: 
         CO, NO, NO2, O3, PM10, PM25, SO2
        -- date: string, date of data collection, format is YYYYMMDD.
    """
    fn = f'{cat}.nc'
    date_start = int(date_start)
    date_end = int(date_end)
    sub_dirs_select = [str(sub_dir) for sub_dir in sub_dirs_sorted \
                       if sub_dir >= date_start and sub_dir <= date_end]
    pol = np.empty((len(sub_dirs_select), np.product(dims)))
    for i, sub_dir in enumerate(sub_dirs_select):
        # get index of sub dir according to date
        idx = sub_dirs_silam.index(sub_dir)
        path_pol = os.path.join(data_dir_silam, sub_dirs_silam[idx], fn)
        pol_i = xr.open_dataset(path_pol)
        pol_i = pol_i[cat].mean('time').data.flatten()
        pol[i] = pol_i
    pol = np.mean(pol, 0)
    return pol


vmin = np.inf
vmax = 0.0
exs = []
eys = []
hists = []
for i, country in enumerate(pop_settings['countries']):
    exs.append([])
    eys.append([])
    hists.append([])
    # load population
    pop = load_population_xr_array(country)
    # filter na values
    mask = pop > pop_settings["pop_min"]
    pop = pop[mask]
    for j, cat in enumerate(pol_settings['categories']):
        # load pollution
        pol = load_pollution_xr_array(
            cat, pol_settings["date_start"], pol_settings["date_end"])
        pol = pol[mask]

        # compute histogram

        # use qcut if slicing into equally sized bins,
        # use cut if slicing into custom bins, this requires a numpy array
        # specifying bin edges.
        if isinstance(pop_bins, int):
            bins_x = pd.qcut(pop, q=pop_bins, duplicates='drop')
        else:
            bins_x = pd.cut(pop, bins=pop_bins, duplicates='drop')
        
        if isinstance(pol_bins, int):
            bins_y = pd.qcut(pol, q=pol_bins, duplicates='drop')
        else:
            bins_y = pd.cut(pol, bins=pol_bins, duplicates='drop')
        
        edges_x = [el.left for el in bins_x.categories.to_numpy()]
        edges_y = [el.left for el in bins_y.categories.to_numpy()] 

        hist, ex, ey = np.histogram2d(
            x=pop,
            y=pol,
            bins=(edges_x, edges_y))

        vmin = min(hist.min(), vmin)
        vmax = max(hist.max(), vmax)

        hists[i].append(hist)
        exs[i].append(ex)
        eys[i].append(ey)


vmin = max(vmin, 1)
cmap = copy.copy(plt.get_cmap('BrBG'))
cmap.set_under('white')
for i, country in enumerate(pop_settings['countries']):
    for j, cat in enumerate(pol_settings['categories']):
        gamma = hists[i][j].mean()/hists[i][j].max()
        gamma = min(1.0, gamma*2)
        # import pdb; pdb.set_trace();
        im = axes[i][j].imshow(
            hists[i][j],
            cmap=cmap,
            norm=colors.PowerNorm(gamma=gamma, vmin=max(hists[i][j].min(), 1), vmax=hists[i][j].max()),
            # norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            extent=[0, len(exs[i][j]), 0, len(eys[i][j])],
            aspect='auto')
        # import pdb; pdb.set_trace();
        axes[i][j].set_xlabel("Population", fontsize=10)
        axes[i][j].set_xticks(np.arange(len(exs[i][j])))
        axes[i][j].set_xticklabels(
            [int(el) for el in exs[i][j]],
            fontsize=8,
            rotation=45)
        axes[i][j].set_ylabel(f"Pollution: {cat}", fontsize=10)
        axes[i][j].set_yticks(np.arange(len(eys[i][j])))
        axes[i][j].set_yticklabels(
            np.round(eys[i][j], 2),#.astype(np.uint8),
            fontsize=8)
        title = f"Histogram Population Density ({country}) vs. Pollution ({cat})"
        axes[i][j].set_title(
            title, fontsize=10)
        if (hist.min() == 0.0) and (hist.max() == 0.0):
            continue
        else:
            fig.colorbar(
                im,
                ax=axes[i][j],
                location='right',
                fraction=0.10,
                aspect=80,
                pad=0.25,
                format='%1.0f')


# set the spacing between subplots
plt.subplots_adjust(left=0.025,
                    bottom=0.065, 
                    right=0.85, 
                    top=0.9, 
                    wspace=0.07, 
                    hspace=0.35)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.025, 0.7])
# fig.colorbar(
#     im,
#     cax=cbar_ax,
#     fraction=0.10,
#     aspect=50,
#     pad=0.15,
#     format='%1.0f')
countries_title = [iso3166.countries_by_alpha3[c].name for c in pop_settings["countries"]]
title = "Comparison of Pollution Histograms over Countries:\n"
title += ', '.join(countries_title)
fig.suptitle(title)
# plt.subplot_tool()
plt.show()
plt.savefig(plot_save_fp)
plt.close("all")



