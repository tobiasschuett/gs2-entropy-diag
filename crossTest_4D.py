import netCDF4 as nc
import numpy as np
import xarray as xr
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

data = nc.Dataset("nlinear/test/c1.out.nc") #test4 folder is baseline that works
S_transfer_python = xr.open_dataset("entropy_transfer.nc")

S_transfer = np.array(data["entropy_transfer_4D"][:])

python_sym = S_transfer_python["symmetrised"].values
print("python script symmtrisation setting set to: ",python_sym,"-- make sure this agrees with GS2 symmtrisation setting here")

#select theta = midplane because this is 4D calculation
theta = S_transfer_python["theta"].values
imidplane = np.argmin(np.abs(theta))

S_transfer_python = S_transfer_python.isel(theta=imidplane).sum(dim=["energy","lambda","sign"])

#take out result of ky >= 0, negative kys result is symmetric to positive ones
ky = S_transfer_python["kys"].values
nky = len(ky)
iky0 = np.argmin(np.abs(ky))

S_transfer_python = S_transfer_python["entropy_transfer"][iky0:,iky0:,:,:].values

A = S_transfer[-1,:,:,:,:] #select last timestep of GS2 result
B = S_transfer_python

if True:
	print("A.shape",A.shape)
	print("B.shape",B.shape)
	#for actually comparing values to manually check some of it to not rely fully on np.allclose
	print("A",A[1,0,:3,:]) 
	print("B",B[1,0,:3,:])

resultsAgree = np.allclose(A,B)

print("python script and gs2 diagnostic compared with default rel_tol and abs_tol, result")
print(resultsAgree)
