#!/bin/bash
#SBATCH --ntasks 32
#SBATCH --time 10
#SBATCH --qos bbshort
#SBATCH --mail-type ALL
#SBATCH --array 0-31

set -e
module purge; module load bluebear
module load Python/3.8.2-GCCcore-9.3.0
python code/vortex_avalanches.py -s ${SLURM_ARRAY_TASK_ID} -v 50 --dt 1e-4