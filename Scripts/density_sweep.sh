#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --mem 16G
#SBATCH --time 48:0:0
#SBATCH --account watkinjs-supercon
#SBATCH --qos bbdefault
#SBATCH --mail-type ALL
#SBATCH --array 0-4

cd /rds/projects/w/watkinjs-supercon/O_VortexAvalances

# densities=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0)
densities=(4.0 4.5 5.0 5.5 6.0)

set -e
module purge; module load bluebear
module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2
python code/vortex_avalanches.py -d ${densities[${SLURM_ARRAY_TASK_ID}]} -n "density_sweep_${densities[${SLURM_ARRAY_TASK_ID}]}" -p 100000 --rel_stop_speed 0.1