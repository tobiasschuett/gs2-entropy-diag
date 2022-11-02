import netCDF4 as nc
import numpy.ma as ma
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = nc.Dataset("nlinear1/test2/c1.out.nc")
print("data at last time step")
print(data["entropy_transfer"][-1,:,:])
