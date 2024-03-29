#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --time 300
#SBATCH --account watkinjs-supercon
#SBATCH --qos bbdefault
#SBATCH --mail-type ALL
#SBATCH --array 0-31

cd /rds/projects/w/watkinjs-supercon/O_VortexAvalances

set -e
module purge; module load bluebear
module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2
python code/vortex_avalanches.py -s ${SLURM_ARRAY_TASK_ID} 