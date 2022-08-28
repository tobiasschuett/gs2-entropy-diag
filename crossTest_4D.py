import netCDF4 as nc
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

data = nc.Dataset("nlinear1/test4/c1.out.nc") #test2 folder is baseline that works
S_transfer = np.array(data["entropy_transfer_4D"][:])

S_transfer_python = np.load("Stransfer-test2_entropy.npz")
S_transfer_python_sum = np.sum(S_transfer_python["entropy_result"],axis=(0,1,2,3))

digit_precision = 5

#normalise to take powers of ten away and to make decimal rounding sensible
TransferMax = S_transfer.max()
S_transfer /= TransferMax
S_transfer_python_sum /= TransferMax

A = np.round(S_transfer[-1,:,:,:,:],digit_precision)
B = np.round(S_transfer_python_sum,digit_precision)

resultsAgree = (A==B).all()

print("python script and gs2 diagnostic compared up to "+str(digit_precision)+" digits precision, result:")
print(resultsAgree)
