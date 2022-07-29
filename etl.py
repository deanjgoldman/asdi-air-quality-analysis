import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import pandas as pd
import geopandas as gpd

import time
import subprocess
from tqdm import tqdm
import os

import s3fs
import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config
import gzip

import settings

# directory settings
dir_output = settings.data_dir
dir_output_pol = settings.dir_output_pol
dir_output_pop = settings.dir_output_pop
dir_output_pop_raw = settings.dir_output_pop_raw
dir_output_pop_reg = settings.dir_output_pop_reg
os.makedirs(dir_output, exist_ok=True)
os.makedirs(dir_output_pol, exist_ok=True)
os.makedirs(dir_output_pop, exist_ok=True)
os.makedirs(dir_output_pop_raw, exist_ok=True)
os.makedirs(dir_output_pop_reg, exist_ok=True)
###########################
# ETL pollution estimates #
###########################
print("Running etl: SILAM pollution estimate dataset...")

# code aws paths
bucket_name_pol = 'fmi-opendata-silam-surface-netcdf'
path_prefix_pol = 'global'
# query s3 bucket
cmd = f"aws s3 ls --no-sign-request s3://{bucket_name_pol}/{path_prefix_pol}/"
stdout = subprocess.check_output([cmd], shell=True)
# parse results for s3 paths to read
rows_pol = stdout.decode().split("\n")
sub_dirs_pol  = [each.strip().split("PRE ")[1][:-1] for each in rows_pol if "PRE" in each]
# instantiate s3fs client
fs = s3fs.S3FileSystem(anon=True)

if len(os.listdir(dir_output_pol)) != len(sub_dirs_pol):
	pol = {}
	for date in tqdm(sub_dirs_pol):

		dir_output_pol_date = os.path.join(dir_output_pol, date)
		os.makedirs(dir_output_pol_date, exist_ok=True)

		pol_date = {}

		cmd = f"aws s3 ls --no-sign-request s3://{bucket_name_pol}/{path_prefix_pol}/{date}/"
		stdout = subprocess.check_output([cmd], shell=True)
		rows_pol = stdout.decode().split("\n")
		rows_pol = [each.split(" ")[-1] for each in rows_pol]
		# organize paths by category
		per_cat = {}
		for fn in rows_pol:
			split = fn.split("_")

			# just grab the first day for now
			if not "d0" in split[-1]:
				continue

			cat = split[-2]
			path_pol_src = f'{bucket_name_pol}/{path_prefix_pol}/{date}/{fn}'
			path_pol_dst = os.path.join(dir_output_pol_date, f"{cat}.nc")

			# skip if data already exists
			if os.path.exists(path_pol_dst):
				continue

			with fs.open(path_pol_src, 'rb') as f:
				pol_ds = xr.open_dataset(f)
				
				pol_date[cat] = path_pol_dst
				pol_ds.to_netcdf(path_pol_dst)

		pol[date] = pol_date
else:
	print("SILAM data collected, loading dataset for registration...")
	# grab any complete dataset to use
	pol_ds = xr.open_dataset(os.path.join(dir_output_pol, sub_dirs_pol[0], "CO.nc"))

########################
# Lat lon registration #
########################
# requires at least one pollution dataset `pol[date]` in memory

def register_lat_lon_chunks(chunk, df):
	"""
	Align Meta population data with SILAM pollution data.
	For all points in Meta datasets coordinate space, find the
	nearest point in SILAM coordinates. Applies to a partition of
	the whole dataset being cumulatively aggregated: `df`, and
	some new `chunk` of the dataset being aggregated into `df`."""

	lat = []
	lon = []
	pol_lat = np.array(pol_ds['lat'].data)[:, None]
	pol_lon = np.array(pol_ds['lon'].data)[:, None]

	df_lat = chunk['latitude'].values[None, :]
	df_lon = chunk['longitude'].values[None, :]

	dist_lat = np.abs(df_lat - pol_lat)
	index_lat = np.argmin(dist_lat.T, 1)
	lat.extend(pol_ds['lat'].data[index_lat])

	dist_lon = np.abs(df_lon - pol_lon)
	index_lon = np.argmin(dist_lon.T, 1)
	lon.extend(pol_ds['lon'].data[index_lon])

	chunk.drop(["latitude", "longitude"], axis=1, inplace=True)
	chunk["lat"] = lat
	chunk["lon"] = lon
	chunk = chunk.groupby(["lat", "lon"]).agg({"population": "sum"}).reset_index()

	pol_lat = pol_lat.squeeze(1)
	pol_lon = pol_lon.squeeze(1)
	pol_lat_lon = np.dstack(np.meshgrid(pol_lat, pol_lon)).reshape(-1, 2)
	pol_lat_lon = pd.DataFrame(pol_lat_lon, columns=["lat", "lon"])
	chunk = chunk.merge(pol_lat_lon, on=["lat", "lon"], how='outer').fillna(0.0)

	if len(df) == 0:
		df = chunk.copy(deep=True)
	else:
		df = pd.concat([df, chunk])
		# aggregate
		df = df.groupby(["lat", "lon"]).agg({"population": "sum"}).reset_index()

	return df 

def register_lat_lon(df, chunk_size=1e+5):
	"""
	Align Meta population data with SILAM pollution data.
	For all points in Meta datasets coordinate space, find the
	nearest point in SILAM coordinates."""

	if len(df) <= chunk_size:
		chunk_size = len(df)

	lat = []
	lon = []
	start = 0
	end = chunk_size-1
	pol_lat = np.array(pol_ds['lat'].data)[:, None]
	pol_lon = np.array(pol_ds['lon'].data)[:, None]
	while True:
		df_lat = df.loc[start: end]['latitude'].values[None, :]
		df_lon = df.loc[start: end]['longitude'].values[None, :]

		dist_lat = np.abs(df_lat - pol_lat)
		index_lat = np.argmin(dist_lat.T, 1)
		lat.extend(pol_ds['lat'].data[index_lat])

		dist_lon = np.abs(df_lon - pol_lon)
		index_lon = np.argmin(dist_lon.T, 1)
		lon.extend(pol_ds['lon'].data[index_lon])

		if start >= len(df):
			break

		start += chunk_size
		end += chunk_size

	df.drop(["latitude", "longitude"], axis=1, inplace=True)
	df["lat"] = lat
	df["lon"] = lon
	df = df.groupby(["lat", "lon"]).agg({"population": "sum"}).reset_index()

	pol_lat = pol_lat.squeeze(1)
	pol_lon = pol_lon.squeeze(1)
	pol_lat_lon = np.dstack(np.meshgrid(pol_lat, pol_lon)).reshape(-1, 2)
	pol_lat_lon = pd.DataFrame(pol_lat_lon, columns=["lat", "lon"])
	df = df.merge(pol_lat_lon, on=["lat", "lon"], how='outer').fillna(0.0)

	return df 


################################# 
# ETL Meta population estimates #
#################################
print("Running etl: Meta population density dataset...")

# meta dataforgood aws path 
bucket_name_pop = 'dataforgood-fb-data'
# set path prefix, at the time of writing, meta has only added one month so far
# but this could potentially be updated to month={this year}-{this month}
path_prefix_pop = 'csv/month=2019-06'
# aws cli command
cmd = f'aws s3 ls --no-sign-request s3://{bucket_name_pop}/{path_prefix_pop}/'
# query s3 bucket, return results
stdout = subprocess.check_output([cmd], shell=True)
# parse results for s3 paths to read
rows_pop = stdout.decode().split("\n")
# iterating through country sub dirs, organize dataset by country
sub_dirs_pop  = [each.strip().split("PRE ")[1][:-1] \
				for each in rows_pop if "=" in each]
countries = [each.split("=")[1].split("/")[0] \
			 for each in rows_pop if "=" in each]  

# set up list for world pop estimate
pop_countries = []

# s3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

for i, country in tqdm(enumerate(countries), total=len(countries)):

	# template meta's filename convention
	fn = f"{country}_total_population.csv.gz"
	key = os.path.join(path_prefix_pop, sub_dirs_pop[i], "type=total_population", fn)
	# destination filepath
	dst = os.path.join(dir_output_pop_reg, f"pop_{country}.nc")
	if os.path.exists(dst):
		continue

	try:
		obj = s3.get_object(Bucket=bucket_name_pop, Key=key)
	except:
		print(f'Not found {key}')
		continue

	# once data is pulled down, load local file
	local_dst = os.path.join(dir_output_pop_raw, fn)
	if os.path.exists(local_dst):
		pass
	else:
		cmd = f'aws s3 cp --no-sign-request s3://{bucket_name_pop}/{key} {local_dst}'
		stdout = subprocess.check_output([cmd], shell=True)

	# USA dataset is larger, requires registration in chunks
	if country == "USA":
		pop_country = pd.DataFrame([])
		reader = pd.read_csv(local_dst, compression="gzip", sep="\t", chunksize=1e+5)
		for chunk in reader:
			pop_country = register_lat_lon_chunks(chunk, pop_country)
	else:
		# read file, register lat lon, clean data
		pop_country = pd.read_csv(local_dst, compression="gzip", sep="\t")
		pop_country = register_lat_lon(pop_country)
	
 	# fill na population with 0 pop. density
	pop_country.fillna(1e-6, inplace=True)
	
	if not 'population' in pop_country.columns:
		print("No 'population' column found for {local_dst}")
		continue

	# write registered netcdf to local 
	pop_country.set_index(['lat', 'lon']).to_xarray().fillna(1e-6).to_netcdf(dst)
	pop_country['country'] = country
	pop_countries.append(pop_country)

	# aggregate to worldwide population dataset
	pop = pd.concat(pop_countries)
	pop = pop.groupby(["lat", "lon"]).agg({"population": "sum"}).reset_index()

# Write complete file to netcdf and csv
pop.to_csv(os.path.join(dir_output_pop, "pop.csv"))
pop.set_index(['lat', 'lon']).to_xarray().fillna(1e-6).to_netcdf(os.path.join(dir_output_pop, "pop.nc"))