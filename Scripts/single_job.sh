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
# python code/vortex_avalanches.py -n density_sweep_6.0 -d 6.0
python code/short_scripts.py