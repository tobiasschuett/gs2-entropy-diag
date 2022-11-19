#!/bin/bash
inputPath="/users/tms535/Developer/gs2-entropy-diag/nlinear/test/c1.out.nc"
filename_distfn_gs2="/users/tms535/Developer/gs2-entropy-diag/nlinear/test/c1"
scriptPath="/users/tms535/Developer/zonal-transfer-functions/entropy_transfer/compute_entropy_transfer.py"
outPath="entropy_transfer"
nproc=2
nproc_gs2=4
sym_mode="sym" #"sym" or "non-sym"

python3 $scriptPath $inputPath $filename_distfn_gs2 $nproc $outPath $nproc_gs2 $sym_mode
