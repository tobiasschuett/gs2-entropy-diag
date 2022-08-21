import multiprocessing as mp
import numpy as np
import sys
import xarray as xr
from itertools import product
import helpers as h

# computes only entropy transfer and not the moment based transfer
# all functions based on Steve's work
#TO DO: -use xarray throughout and write to netCDF file for better documantion of layer meaning

def parse_args():
    input_filename, nproc, output_filename, nproc_gs2 = sys.argv[1:]
    nproc = int(nproc)
    nproc_gs2 = int(nproc_gs2)
    return input_filename, nproc, output_filename, nproc_gs2


def parallel_compute_ENTROPY_transfer_along_theta_energy_lambda_sign():
    # Loop over theta in parallel
    with mp.Pool(processes=nproc) as p:
        results = np.array(
            p.starmap(parallel_compute_ENTROPY_transfer_wrapper, 
                         product(range(ntheta),range(nenergy),range(nlambda),range(nsign))), 
                         #first one, i.e. ntheta, most parallel, see reshape before writing output in main
            dtype=float,
        )
    return results


def parallel_compute_ENTROPY_transfer_wrapper(i_theta,i_energy,i_lambda,i_sign):
    #print("Computing net transfer for theta,energy,lambda,sign = {},{},{},{}" \
    #      .format(ds["theta"].values[i_theta],ds["energy"].values[i_energy],ds["lambda"].values[i_lambda],i_sign))
    result1 = 1 #get_Maxwellian_pre_factor(ds["lambda"][i_lambda].values,ds["energy"][i_energy].values,ds["bmag"][i_theta].values)
    result2 = compute_ENTROPY_transfer_this_theta(
       g[:,:,i_theta,0,i_energy,i_lambda,i_sign].values,
       ds["phi_t"][:, :, :, i_theta, :].values,
    )
    return result1*result2

def compute_ENTROPY_transfer_this_theta(g,phi): #basically same as the function above
    # ri to complex already done before in get_g()
    phi = ri_to_complex(phi)
    
    # no t axis to be moved for g since we're at last time step here
    phi = move_first_axis_to_last(phi)
    
    # Make full versions of arrays to simplify indexing
    phi = make_full(phi)
    g_s = make_full_g(g)
    
    # Pre-shift arrays so that indexing works correctly
    g_s = np.fft.fftshift(g_s, axes=(0, 1))
    phi = np.fft.fftshift(phi, axes=(0, 1))
    
    # Pre-compute complex conjugate array: not needed based on equation in paper, but set zonal target
    g_t = g_s
    
    #compute phi at last time step
    phi = phi[:,:,-1] 
                
    #Pre-prepare array of mediators for performance
    phi_m = compute_phi_m_at_last_t(phi, ikx0, iky0)
    g_m = compute_g_m(g_s,ikx0,iky0)
    
    # Compute transfer and return
    return compute_net_entropy_transfer(
        g_t,g_s,phi,g_m,phi_m
    )


def ri_to_complex(array_ri):
    array_complex = np.zeros(array_ri.shape[:-1], dtype=complex)
    array_complex.real = array_ri[..., 0]
    array_complex.imag = array_ri[..., 1]
    return array_complex


def move_first_axis_to_last(array_orig):
    # Roll first axis to last place, e.g. for improved cache performance
    # NB: Cannot use np.moveaxis(...) as the return value is just a view of the original
    array_new = np.zeros(
        np.concatenate((array_orig.shape[1:], [array_orig.shape[0]])),
        dtype=array_orig.dtype,
    )
    for i in range(array_orig.shape[0]):
        array_new[..., i] = array_orig[i, ...]
    return array_new

def make_full_g(array):
    # Assumes dimension order is ky, kx
    # For concatenated array:
    # - ky slice = -1:0:-1 so that we include non-zero kys in reverse order
    # - kx slice = -1::-1 so that we include all kxs in reverse order but this puts kx = 0
    #     to the end even though we still want it at the start, hence use of
    #     np.roll(..., 1, axis=1) so that kx = 0 is at ikx = 0
    # - np.conj(...) of the reversed and rolled array as the negative kys are the conjugate
    #     of the positive kys
    # - np.concatenate(..., axis=0) so that we concatenate in the ky direction
    return np.concatenate(
        (array, np.conj(np.roll(array[-1:0:-1, -1::-1], 1, axis=1))),
        axis=0,
    )

def make_full(array):
    # Assumes dimension order is ky, kx, t
    # For concatenated array:
    # - ky slice = -1:0:-1 so that we include non-zero kys in reverse order
    # - kx slice = -1::-1 so that we include all kxs in reverse order but this puts kx = 0
    #     to the end even though we still want it at the start, hence use of
    #     np.roll(..., 1, axis=1) so that kx = 0 is at ikx = 0
    # - np.conj(...) of the reversed and rolled array as the negative kys are the conjugate
    #     of the positive kys
    # - np.concatenate(..., axis=0) so that we concatenate in the ky direction
    return np.concatenate(
        (array, np.conj(np.roll(array[-1:0:-1, -1::-1, :], 1, axis=1))),
        axis=0,
    )
        
def compute_phi_m_at_last_t(phi, ikx0, iky0):
    # Pre-allocate array
    # NB: Cannot pre-allocate this variable in the global scope and re-use it here
    phi_m = np.zeros((nky, nky, nkx, nkx), dtype=complex)
    # Loop over target and source wavenumbers
    for ikys in range(nky):
        for ikyt in range(nky):
            for ikxs in range(nkx):
                for ikxt in range(nkx):
                    # Work out index of mediator
                    ikxm = ikxt - ikxs + ikx0
                    ikym = ikyt - ikys + iky0
                    # Check mediator index exists
                    if not (0 <= ikxm and ikxm < nkx and 0 <= ikym and ikym < nky):
                        # Just don't set a value to avoid unnecessary cache misses
                        continue
                    # Store mediator value in mediator array
                    phi_m[ikys, ikyt, ikxs, ikxt] = phi[ikym, ikxm]
    # Return output array
    return phi_m

def compute_g_m(g, ikx0, iky0):
    # Pre-allocate array
    # NB: Cannot pre-allocate this variable in the global scope and re-use it here
    g_m = np.zeros((nky, nky, nkx, nkx), dtype=complex)
    # Loop over target and source wavenumbers
    for ikys in range(nky):
        for ikyt in range(nky):
            for ikxs in range(nkx):
                for ikxt in range(nkx):
                    # Work out index of mediator
                    ikxm = ikxt - ikxs + ikx0
                    ikym = ikyt - ikys + iky0
                    # Check mediator index exists
                    if not (0 <= ikxm and ikxm < nkx and 0 <= ikym and ikym < nky):
                        # Just don't set a value to avoid unnecessary cache misses
                        continue
                    # Store mediator value in mediator array
                    g_m[ikys, ikyt, ikxs, ikxt] = g[ikym, ikxm]
    # Return output array
    return g_m

def compute_net_entropy_transfer(g_t,g_s,phi,g_m,phi_m): 
    T_s = (
        z_hat_dot_k_cross_k_prime
        * (
             np.reshape(g_t,(1,nky,1,nkx))
              * phi_m
              * np.reshape(g_s,(nky,1,nkx,1)) -
              np.reshape(g_t,(1,nky,1,nkx))
              *g_m
              *np.reshape(phi,(nky,1,nkx,1))
          ).real
    )
    return T_s

def get_Maxwellian_pre_factor(lambdaa,energyy,bmagg):
    c = (((2*np.pi)**(3/2))/(2*bmagg))
    return c*np.exp(energyy*(1+bmagg/(2*lambdaa)))

if __name__ == "__main__":
    ''' Parse command line inputs:
    input_filename_without_ending: i.e "/users/tms535/scratch/nonlinear_test_runs/cyclone33/c33"
    it_start: start of cutting for moments based transfer
    it_end: end for cutting for moments based transfer
    nproc: number of processors for parallelising this computation
    out_filename: without ending, start name of output files
    nproc_gs2: number of processors that was used in simulation of input file, important for getting g
    '''
    
    input_filename_without_ending, nproc, output_filename, nproc_gs2 = parse_args()

    input_filename = input_filename_without_ending+".out.nc"

    # Open Dataset
    if "," in input_filename:
        input_filename_list = [f for f in input_filename.split(",")]
        ds = xr.open_mfdataset(input_filename_list)
    else:
        ds = xr.open_dataset(input_filename)
        
    # Remove un-used variables
    required_vars = ["theta", "kx", "ky", "density_t", "ntot_t", "phi_t","energy","lambda","bmag"]
    vars_to_delete = []  
    for v in ds.data_vars:
        if v not in required_vars:
            vars_to_delete.append(v)
    ds = ds.drop_vars(vars_to_delete)
    
    # Load data for performance and to avoid concurrent read problems
    ds.load()
    
    # Pre-compute various quantities for performance
    ntheta = len(ds["theta"])
    kx = np.fft.fftshift(ds["kx"].values)
    nkx = len(kx)
    ky = np.fft.fftshift(np.concatenate((ds["ky"].values, -ds["ky"].values[-1:0:-1])))
    nky = len(ky)
    ikx0 = np.argmin(np.abs(kx))
    iky0 = np.argmin(np.abs(ky))

    #precomputute quantities for entropy transfer
    nenergy = len(ds["energy"])
    nsign = 2
    nlambda = len(ds["lambda"])

    #get distr function g
    g = h.get_g(nproc_gs2,input_filename_without_ending)

    #compute constant prefactors
    z_hat_dot_k_cross_k_prime = np.reshape(
        np.reshape(
            np.reshape(kx, (1, nkx)) * np.reshape(ky, (nky, 1)),
            (nky, 1, 1, nkx)
        ) -
        np.reshape(
            np.reshape(kx, (1, nkx)) * np.reshape(ky, (nky, 1)),
            (1, nky, nkx, 1)
        ),
        (nky, nky, nkx, nkx)
    )
    
    #Do calculation
    entropy_result = parallel_compute_ENTROPY_transfer_along_theta_energy_lambda_sign()
    entropy_result = entropy_result.reshape((ntheta,nenergy,nlambda,nsign,nky,nky, nkx,nkx))

    # Write output
    output_shape = ["theta","energy","lambda","sign","kys","kyt","kxs","kxt"]
    np.savez(output_filename+"_entropy",entropy_result=entropy_result,kx=kx,ky=ky,output_shape=output_shape)
