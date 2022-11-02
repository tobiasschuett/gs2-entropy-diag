import netCDF4 as nc
import numpy as np
import xarray as xr
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


data = nc.Dataset("nlinear1/test4/c1.out.nc") #test4 folder is baseline that works
S_transfer = np.array(data["entropy_transfer_4D"][:])

S_transfer_python = xr.open_dataset("entropy_transfer.nc")  
S_transfer_python_sum = np.sum(S_transfer_python["entropy_transfer"],axis=(0,1,2,3)) #sum over theta,energy,lambda,sign

#take out result of ky >= 0, negative kys result is symmetric to positive ones
ky = S_transfer_python["kys"].values
nky = len(ky)
iky0 = np.argmin(np.abs(ky))


S_transfer_python_sum = S_transfer_python_sum[iky0:,iky0:,:,:]

A = S_transfer[-1,:,:,:,:]
B = S_transfer_python_sum

#print(A.shape)
#print(B.shape)
#for actually comparing values to manually check some of it to not rely fully on np.allclose
#print("A",A) 
#print("B",B)

resultsAgree = np.allclose(A,B)

print("python script and gs2 diagnostic compared with default rel_tol and abs_tol, result")
print(resultsAgree)
