#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --mem 40G
#SBATCH --time 6-0:0:0
#SBATCH --account watkinjs-supercon
#SBATCH --qos bbdefault
#SBATCH --mail-type ALL
#SBATCH --array 0-39

cd /rds/projects/w/watkinjs-supercon/O_VortexAvalances

set -e
module purge; module load bluebear
module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2
seed=$((1000+${SLURM_ARRAY_TASK_ID}))

if [ -f "results/Simulation_results/BasicAvalancheResult/system_seed_$seed" ];
then
echo "Found pre-existing result - continuing"
python code/run_avalanche_sim.py -s $seed -n "system_seed_${seed}_cont" --print_after 1000000 --max_time 50000000 -t 518400 --start_from "system_seed_$seed" -v 500
else
echo "No pre-existing result - using init result"
python code/run_avalanche_sim.py -s $seed -n "system_seed_$seed" --print_after 1000000 --max_time 50000000 -t 518400 --start_from "system_seed_${seed}_init" -v 500
fi