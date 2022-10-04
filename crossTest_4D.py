import netCDF4 as nc
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = nc.Dataset("nlinear1/test4/c1.out.nc") #test2 folder is baseline that works
S_transfer = np.array(data["entropy_transfer_4D"][:])

S_transfer_python = np.load("Stransfer-test2_entropy.npz")
S_transfer_python_sum = np.sum(S_transfer_python["entropy_result"],axis=(0,1,2,3))

#take out result of ky >= 0, negative kys result is symmetric to positive ones
ky = S_transfer_python["ky"]
nky = len(ky)
iky0 = np.argmin(np.abs(ky))


S_transfer_python_sum = S_transfer_python_sum[iky0:,iky0:,:,:]

A = S_transfer[-1,:,:,:,:]
B = S_transfer_python_sum

#print("A",A)
#print("B",B)

resultsAgree = np.allclose(A,B)

print("python script and gs2 diagnostic compared with default rel_tol and abs_tol, result")
print(resultsAgree)
