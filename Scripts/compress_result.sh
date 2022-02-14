#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --time 10
#SBATCH --account watkinjs-supercon
#SBATCH --qos bbshort
#SBATCH --mail-type ALL

cd /rds/projects/w/watkinjs-supercon/O_VortexAvalances

set -e
module purge; module load bluebear
module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2
python code/compress.py  -r -n density_sweep_4.5 -d /rds/projects/w/watkinjs-supercon/O_VortexAvalances/results/Simulation_results/AvalancheResult