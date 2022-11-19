import netCDF4 as nc
import numpy as np
import xarray as xr
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

data = nc.Dataset("nlinear/test/c1.out.nc")
S_transfer_python = xr.open_dataset("entropy_transfer.nc")  

S_transfer = np.array(data["entropy_transfer_3D"][:])

#sum along energy,lamda,sign; left over theta,kys,kyt,kxs,kxt
S_transfer_python = S_transfer_python.sum(dim=["energy","lambda","sign"])

python_sym = S_transfer_python["symmetrised"].values
print("python script symmtrisation setting set to: ",python_sym,"-- make sure this agrees with GS2 symmtrisation setting here")

#make 3D
kx = S_transfer_python["kxs"].values
nkx = len(kx)
ky = S_transfer_python["kys"].values
nky = len(ky)

ikx0 = np.argmin(np.abs(kx))
iky0 = np.argmin(np.abs(ky))
ikyt = iky0

#for debug print
theta = S_transfer_python["theta"].values
imidplane = np.argmin(np.abs(theta))

#set zonal ky target and only use result of ky >= 0 values, symmetric anyways
S_transfer_python = S_transfer_python["entropy_transfer"][:,iky0:,ikyt,:,:].values

A = S_transfer[-1,:,:,:,:] #select last timestep of GS2 output
B = S_transfer_python

if True:
	print("A.shape: ",A.shape)
	print("B.shape: ",B.shape)
	#for actually comparing values to manually check some of it to not rely fully on np.allclose
	print("A",A[imidplane,1,:3,:])
	print("B",B[imidplane,1,:3,:])

resultsAgree = np.allclose(A,B)

print("python script and gs2 diagnostic compared with default rel_tol and abs_tol, result")
print(resultsAgree)
