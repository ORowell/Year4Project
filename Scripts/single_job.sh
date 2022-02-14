#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --time 2:0:0
#SBATCH --account watkinjs-supercon
#SBATCH --qos bbdefault
#SBATCH --mail-type ALL

cd /rds/projects/w/watkinjs-supercon/O_VortexAvalances

set -e
module purge; module load bluebear
module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2
python code/avalanche_sim.py -d 4.5 -c 10 -n 'density_4.5_nowrap' -v 20