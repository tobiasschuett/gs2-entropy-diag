# file to crossTest computation of h
# can be recovered with python from the save_distfn printouts
# compared with the implemented diagnostic "h_t"
# Due to analogous implementation this should then also guarantee that "g_t" is implemented correctly

import numpy as np
import xarray as xr
import sys
sys.path.append('/users/tms535/Developer/zonal-transfer-functions/entropy_transfer/local_Mac_testing/indexing-sym')
import helpers_sym

filepath = "nlinear/test/c1.out.nc"
data = xr.open_dataset(filepath)

nproc_gs2 = 4
filepath = "nlinear/test/c1"
h_python = helpers_sym.get_g(nproc_gs2,filepath)

h_python = np.sum(h_python,axis=(3,4,5,6)) #dimensions are ky,kx,theta,spec,egrid,lambda,sign
h_python_real = np.real(h_python)
h_python_imag = np.imag(h_python)

h_GS2 = data["h_t"].isel(t=-1,species=0)
h_GS2_real = h_GS2.isel(ri=0)
h_GS2_imag = h_GS2.isel(ri=1)

thetas = data["theta"].values
midplane = np.argmin(np.abs(thetas))

if True:
    print("h_python.shape: ",h_python_real.shape)
    print("h_GS2.shape: ",h_GS2_real.shape)
    #for actually comparing values to manually check some of it to not rely fully on np.allclose
    print("h_python_real: ",h_python_real[0,:,midplane+1].values)
    print("h_GS2_real: ",h_GS2_real[0,:,midplane+1].values)
    print("h_python_imag: ",h_python_imag[0,:,midplane+1].values)
    print("h_GS2_imag: ",h_GS2_imag[0,:,midplane+1].values)

resultsAgree = np.allclose(h_python_real,h_GS2_real)
resultsAgree = np.allclose(h_python_imag,h_GS2_imag)

print("python script and gs2 diagnostic compared with default rel_tol and abs_tol, result")
print(resultsAgree)
