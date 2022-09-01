#!/bin/bash

inputPath="/users/tms535/Developer/gs2-entropy-diag/nlinear1/test4/c1"
scriptPath="/users/tms535/Developer/gs2-entropy-diag/entropy_transfer.py"
outPath="Stransfer-test2"
nproc=2
nproc_gs2=4

python3 $scriptPath $inputPath $nproc $outPath $nproc_gs2
