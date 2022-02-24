#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --mem 32G
#SBATCH --time 10-0:0:0
#SBATCH --account watkinjs-supercon
#SBATCH --qos bbdefault
#SBATCH --mail-type ALL
#SBATCH --array 0-11

cd /rds/projects/w/watkinjs-supercon/O_VortexAvalances

# densities=(0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0)
densities=($(seq 0.5 0.5 6.0))

set -e
module purge; module load bluebear
module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2
python code/avalanche_sim.py --start_from "new_pins_density_sweep_${densities[${SLURM_ARRAY_TASK_ID}]}" -p 1000000 -c 10 -n "new_pins_continued_${densities[${SLURM_ARRAY_TASK_ID}]}" -v 500 --max_time 10000000