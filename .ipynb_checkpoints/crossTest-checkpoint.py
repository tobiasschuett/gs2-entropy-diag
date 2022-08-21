import netCDF4 as nc
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = nc.Dataset("nlinear1/test2/c1.out.nc")
S_transfer = np.array(data["entropy_transfer"][:])

S_transfer_python = np.load("Stransfer-test2_entropy.npz")
S_transfer_python_sum = np.sum(S_transfer_python["entropy_result"],axis=(0,1,2,3))

A = np.round(S_transfer[-1,:,:,:,:],8)
B = np.round(S_transfer_python_sum,8)

resultsAgree = (A==B).all()

print(resultsAgree)
