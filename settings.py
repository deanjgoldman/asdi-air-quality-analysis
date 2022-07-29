import os

this_wd = "/".join(os.path.abspath(__file__).split("/")[:-1])
data_dir = os.path.join(this_wd, "data")
dir_output = os.path.join(this_wd, "output")
data_dir_pop = os.path.join(data_dir, "population")
data_dir_pop_raw = os.path.join(data_dir_pop, "raw")
data_dir_pop_reg = os.path.join(data_dir_pop, "registered")
data_dir_pol = os.path.join(data_dir, "air_quality")
