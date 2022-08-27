import netCDF4 as nc
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = nc.Dataset("nlinear1/test4/c1.out.nc") #test2 folder is baseline that works
S_transfer = np.array(data["entropy_transfer_3D"][:])

S_transfer_python = np.load("Stransfer-test2_entropy.npz")
#sum along theta,energy,lamda,sign; left over kys,kyt,kxs,kxt
S_transfer_python_sum = np.sum(S_transfer_python["entropy_result"],axis=(0,1,2,3))

#make 3D
kx = S_transfer_python["kx"]
nkx = len(kx)
ky = S_transfer_python["ky"]
nky = len(ky)

ikx0 = np.argmin(np.abs(kx))
iky0 = np.argmin(np.abs(ky))
ikyt = iky0

S_transfer_python_sum = S_transfer_python_sum[:,ikyt,:,:]

digit_precision = 8

# set last timestep with "-1" and set kyt = 0 with "0"
A = np.round(S_transfer[-1,:,:,:],digit_precision)
B = np.round(S_transfer_python_sum,digit_precision)

resultsAgree = (A==B).all()

print("python script and gs2 diagnostic compared up to "+str(digit_precision)+" digits precision, result:")
print(resultsAgree)
