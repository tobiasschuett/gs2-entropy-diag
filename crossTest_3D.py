import netCDF4 as nc
import numpy as np
import xarray as xr
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

data = nc.Dataset("nlinear1/test4/c1.out.nc") #test4 folder is baseline that works
S_transfer = np.array(data["entropy_transfer_3D"][:])

S_transfer_python = xr.open_dataset("entropy_transfer.nc")  
#sum along energy,lamda,sign; left over theta,kys,kyt,kxs,kxt
S_transfer_python_sum = np.sum(S_transfer_python["entropy_transfer"],axis=(1,2,3)) #leave 0 which is theta

#make 3D
kx = S_transfer_python["kxs"].values
nkx = len(kx)
ky = S_transfer_python["kys"].values
nky = len(ky)

ikx0 = np.argmin(np.abs(kx))
iky0 = np.argmin(np.abs(ky))
ikyt = iky0

#set zonal ky target and only use result of ky >= 0 values, symmetric anyways
S_transfer_python_sum = np.array(S_transfer_python_sum[:,iky0:,ikyt,:,:])

A = S_transfer[-1,:,:,:,:]
B = S_transfer_python_sum

resultsAgree = np.allclose(A,B)

print("python script and gs2 diagnostic compared with default rel_tol and abs_tol, result")
print(resultsAgree)
