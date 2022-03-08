#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --time 7:0:0
#SBATCH --account watkinjs-supercon
#SBATCH --qos bbdefault
#SBATCH --mail-type ALL

cd /rds/projects/w/watkinjs-supercon/O_VortexAvalances

set -e
module purge; module load bluebear
module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2
# python code/run_avalanche_sim.py -s 1001 -n "test5.5" --print_after 1000000 -i 250 -v 500
python code/avalanche_sim.py --start_from "test5.5_init" -p 1000000 -c 10 -n "test5.5cont" -v 10 --max_time 10000000
# python code/avalanche_sim.py -n "continued_2.5" -p 1000000 -c 10 --max_time 10000000 -v 500 --start_from "density_sweep_2.5"