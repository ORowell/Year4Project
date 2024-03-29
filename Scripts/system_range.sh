#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --mem 40G
#SBATCH --time 10-0:0:0
#SBATCH --account watkinjs-supercon
#SBATCH --qos bbdefault
#SBATCH --mail-type ALL
#SBATCH --array 0-39

cd /rds/projects/w/watkinjs-supercon/O_VortexAvalances

set -e
module purge; module load bluebear
module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2
seed=$((1100+${SLURM_ARRAY_TASK_ID}))
python code/run_avalanche_sim.py -s $seed -n "system_seed_${seed}_lnoise" --print_after 1000000 --print_after 1000000 -t 864000 --noise 0.1 -v 700 #--start_from "system_seed_${seed}_init"