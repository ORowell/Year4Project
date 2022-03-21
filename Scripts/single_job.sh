#!/bin/bash
#SBATCH --ntasks 2
#SBATCH --time 10-0:0:0
#SBATCH --mem 40G
#SBATCH --account watkinjs-supercon
#SBATCH --qos bbdefault
#SBATCH --mail-type ALL

cd /rds/projects/w/watkinjs-supercon/O_VortexAvalances

set -e
module purge; module load bluebear
module load BEAR-Python-DataScience/2020a-foss-2020a-Python-3.8.2
python code/run_avalanche_sim.py -s 1001 -n "big5.5" --print_after 1000000 -i 800 -v 100 --length 15 --width 15 -t 864000 --compress 10
# python code/avalanche_sim.py -n "new_continued_3.0" -p 1000000 -c 10 -v 300 --start_from "density_sweep_3.0"