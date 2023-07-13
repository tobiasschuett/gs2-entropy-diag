# !/bin/bash
# ~/scratch/tms535/python_environments/envs/my_first_environment/bin/python

# bash script to cross-validate GS2 entropy transfer output against python output
# python script is tested against the expected symmetries for the non-sym and sym-case
# can be used to see if changes in GS2 implementation break something

echo 'running GS2 test simulation...'

bash ./nlinear/test/run.sh

echo 'finished test run...'
wait
echo 'updating python output based on phi and g from new sim output...'

bash run-python-script.sh

echo 'finished updating python output'
echo 'running cross check for 3D case...'

python crossTest_3D.py

echo 'running cross check for 4D case...'

python crossTest_4D.py

echo 'running cross check for 4D_extended case...'

python crossTest_4D_extended.py

echo 'running cross check for write h...'

python crossTest_h.py
