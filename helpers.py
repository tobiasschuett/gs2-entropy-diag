import numpy as np
import xarray as xr
from tqdm import tqdm

def get_g(nproc,filepath):
    for i in range(nproc):
        ds = xr.open_dataset(filepath+".nc.dfn."+str(i))
        gr = ds["gr"]
        gi = ds["gi"]
        if i == 0:
            dim_dict = {
            "x": ds["ntheta0"].values,
            "y": ds["naky"].values,
            "l": ds["nlambda"].values,
            "e": ds["negrid"].values,
            "s": ds["nspec"].values,
            "nsign": len(ds["sign"]), #already dimensions in original, just carry through
            "ntheta": len(ds["theta"]) #already dimensions in original, just carry through
            }
            gr_all, gi_all = gr.values, gi.values
        else:
            gr_all = np.concatenate((gr_all,gr.values))
            gi_all = np.concatenate((gi_all,gi.values))
            
    layout_reshape_array, reshape_dims_dict = get_layout_reshape_array(ds,dim_dict)
    print("shape of reshape array: ",layout_reshape_array)
    print(gr_all.shape)

    gr_all = np.reshape(gr_all,layout_reshape_array)
    gi_all = np.reshape(gi_all,layout_reshape_array)
    
    g_all = np.zeros(gr_all.shape, dtype=complex)
    g_all.real = gr_all
    g_all.imag = gi_all
    
    g_all = xr.DataArray(
        data = g_all,
        dims = reshape_dims_dict,
    )
    
    #print(g_all.dims)
    
    #current oder: reshape_dims_dict = flip(layout_string),sign,theta
    #order I want for g: ky,kx,theta,species,energy,lambda,sign
    #use this because np.transpose is broken
    g_all = g_all.transpose("y","x","theta","s","e","l","sign")
    
    #print(g_all.dims)
    
    return g_all

def get_layout_reshape_array(ds,dim_dict):
    layout_string = ds["layout"].values.astype(str).flatten()[0]
    #print("detected layout string: ",layout_string)
    
    #mirror the layout_string as "the rightmost dimensions in layout string are paralleised first" in gs2.
    #The first dimension in the reshape array is the once that splits the entire array up, i.e. the most paralleised dimension
    #given that we obtained the total array by stacking the results from all processors.
    #Hence, we want the rightmost dimension to be in first position in the newshape array
    #as we iterate through and effectively append in for loop below, the flip is needed
    #xy are chose to be on the left since fourier analysis needs access to full domain on each processor
    layout_string = layout_string[::-1] 
    #print("dimesions ordered from left to right: ",layout_string,",sign,theta")
    
    reshape_array = []
    reshape_dims_dict = []
    for i in layout_string:
        reshape_array = np.concatenate((reshape_array,[dim_dict[i]])).astype(int)
        reshape_dims_dict = np.append(reshape_dims_dict,i)
        
    reshape_array = np.concatenate((reshape_array,[dim_dict["nsign"]])).astype(int)
    reshape_array = np.concatenate((reshape_array,[dim_dict["ntheta"]])).astype(int)
    reshape_dims_dict = np.append(reshape_dims_dict,["sign","theta"])
    
    #print(reshape_dims_dict)
    
    return reshape_array, reshape_dims_dict

def get_g_at_timestep(nproc,filepath,timestep):
    for i in range(nproc):
        ds = xr.open_dataset(filepath+".nc"+str(timestep).zfill(5)+".dfn."+str(i))
        gr = ds["gr"]
        gi = ds["gi"]
        if i == 0:
            dim_dict = {
            "x": ds["ntheta0"].values,
            "y": ds["naky"].values,
            "l": ds["nlambda"].values,
            "e": ds["negrid"].values,
            "s": ds["nspec"].values,
            "nsign": len(ds["sign"]), #already dimensions in original, just carry through
            "ntheta": len(ds["theta"]) #already dimensions in original, just carry through
            }
            gr_all, gi_all = gr.values, gi.values
        else:
            gr_all = np.concatenate((gr_all,gr.values))
            gi_all = np.concatenate((gi_all,gi.values))
            
    layout_reshape_array, reshape_dims_dict = get_layout_reshape_array(ds,dim_dict)
    print("shape of reshape array: ",layout_reshape_array)
    print(gr_all.shape)

    gr_all = np.reshape(gr_all,layout_reshape_array)
    gi_all = np.reshape(gi_all,layout_reshape_array)
    
    g_all = np.zeros(gr_all.shape, dtype=complex)
    g_all.real = gr_all
    g_all.imag = gi_all
    
    g_all = xr.DataArray(
        data = g_all,
        dims = reshape_dims_dict,
    )
    
    #print(g_all.dims)
    
    #current oder: reshape_dims_dict = flip(layout_string),sign,theta
    #order I want for g: ky,kx,theta,species,energy,lambda,sign
    #use this because np.transpose is broken
    g_all = g_all.transpose("y","x","theta","s","e","l","sign")
    
    #print(g_all.dims)
    
    return g_all


def symmetry_test1(kx,ky,result,roundDigits=5): # 1.) check that J[k,p,q] = J[k,q,p] with k target, p source and q mediator
    nkx, nky = len(kx), len(ky)
    mediator_indices = compute_mediatorField_index(kx,ky)
    invalid_mdeiators = 0
    testPassed = True
    for i in tqdm(range(nky)):
        for m in range(nky):
            for j in range(nkx):
                for l in range(nkx):
                    if checkValidMediator(kx,ky,i,m,j,l):
                        ikym, ikxm = mediator_indices[i,m,j,l]
                        ikym, ikxm = int(ikym), int(ikxm)

                        if round(result[i,m,j,l],roundDigits) != round(result[ikym,m,ikxm,l],roundDigits):
                            print("error",i,m,j,l,ikym,ikxm)
                            testPassed = False
                            return testPassed
                    else:
                        invalid_mdeiators += 1
                        
    return testPassed

def symmetry_test2(kx,ky,result,roundDigits=5): # 2.) check that J[k,p,q] = J[-k,-p,-q]
    nkx, nky = len(kx), len(ky)
    testPassed = True
    for i in tqdm(range(nky)):
        for m in range(nky):
            for j in range(nkx):
                for l in range(nkx):
                    a1 = result[i,j,m,l]
                    i = flipIndex(kx,i)
                    m = flipIndex(kx,m)
                    j = flipIndex(kx,j)
                    l = flipIndex(kx,l)
                    a2 = result[i,j,m,l]
                    if round(a1,roundDigits) != round(a2,roundDigits):
                        print("error",a1,a2)
                        testPassed = False
                        return testPassed
                        
    return testPassed

def symmetry_test3(kx,ky,result,roundDigits=5): 
    #check that J[k,p,q] + J[p,q,k] + J[q,k,p] = J1 + J2 + J3 = 0 with k target, p source and q mediator 
    #this is  equal to J[k,-k',-k''] + J[-k',-k'',k] + J[-k'',k,-k'] = J1 + J2 + J3 = 0 with k target, -k' source and -k'' mediator
    #due to p = -k' and q = -k''
    mediator_indices = compute_mediatorField_index(kx,ky)
    nkx, nky = len(kx), len(ky)
    testPassed = True
    for i in tqdm(range(nky)):
        for m in range(nky):
            for j in range(nkx):
                for l in range(nkx):
                    if checkValidMediator(kx,ky,i,m,j,l):
                        ikym, ikxm = mediator_indices[i,m,j,l]
                        ikym, ikxm = int(ikym), int(ikxm)
                        
                        # CRUCIAL 2 LINES
                        i, j = flipIndex(ky,i), flipIndex(kx,j) #create -k' and -k''
                        ikym, ikxm = flipIndex(ky,ikym), flipIndex(kx,ikxm) #create -k' and -k''
                        J1 = result[i,m,j,l]
                        if checkValidMediator(kx,ky,ikym,i,ikxm,j):
                            J2 = result[ikym,i,ikxm,j] #source becomes target & mediator becomes source w.r.t J1
                            if checkValidMediator(kx,ky,m,ikym,l,ikxm):
                                J3 = result[m,ikym,l,ikxm] #mediator becomes target & target becomes source w.r.t J1
                                if round(J1+J2+J3,roundDigits) != 0:
                                    print("error",J1+J2+J3)
                                    testPassed = False
                                    return testPassed
                            
    return testPassed

def compute_mediatorField_index(kx,ky):
    ikx0 = np.argmin(np.abs(kx))
    iky0 = np.argmin(np.abs(ky))
    nky = len(ky)
    nkx = len(kx)
    
    mediator_indices = np.zeros((nky,nky,nkx,nkx,2))
    
    for ikxs in range(nkx):
        for ikxt in range(nkx):
            for ikys in range(nkx):
                for ikyt in range(nkx):
                    # Work out index of mediator
                    ikxm = ikxt - ikxs + ikx0
                    ikym = ikyt - ikys + iky0
                    # Check mediator index exists
                    if not (0 <= ikxm and ikxm < nkx and 0 <= ikym and ikym < nky):
                        # Just don't set a value to avoid unnecessary cache misses
                        mediator_indices[ikys,ikyt,ikxs,ikxt] = [99,99]
                    else:
                        mediator_indices[ikys,ikyt,ikxs,ikxt] = [ikym,ikxm]
    # Return output array
    return mediator_indices

def checkValidMediator(kx,ky,i,m,j,l):
    ikx0 = np.argmin(np.abs(kx))
    iky0 = np.argmin(np.abs(ky))
    nkx = len(kx)
    nky = len(ky)
    
    ikxm = l - j + ikx0
    ikym = m - i + iky0
    
    if (0 <= ikxm and ikxm < nkx and 0 <= ikym and ikym < nky):
        return True
    else:
        return False
    
def flipIndex(kx,ix): #convert index of ky into index of -ky
    nkx = len(kx)
    ikx0 = np.argmin(np.abs(kx))
    delta = ix-ikx0
    return ikx0-delta

def has_correct_symmetry(kx,ky,array,roundDigits=5): 
    nkx, nky = len(kx), len(ky)
    #we want array(kx,ky) = conj(array(-kx,-ky)) for reality condition
    ikx0 = int((nkx-1)/2)
    iky0 = int((nky-1)/2)
    result = True
    
    for i in range(nky):
        for j in range(nkx):
            if i == iky0 and j == ikx0:
                continue
            else:
                if round(array[i,j],roundDigits) != round(np.conj(array[-i-1,-j-1]),roundDigits):
                    result = False
    
    return result

def zhat_test(kx,ky):
    #these terms must be zero by equations
    nkx,nky = len(kx),len(ky)
    ikx0, iky0 = int((nkx-1)/2), int((nky-1)/2)
    testPassed = True
    for i in tqdm(range(nky)):
        for m in range(nky):
            for j in range(nkx):
                for l in range(nkx):
                    if (i == m and l == j):
                        if z_hat_dot_k_cross_k_prime[i,m,j,l] != 0:
                            testPassed = False
                    if ((l == ikx0 and m == iky0) or (i == iky0 and j == ikx0)):
                        if z_hat_dot_k_cross_k_prime[i,m,j,l] != 0:
                            testPassed = False
    return testPassed