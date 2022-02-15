python -m cProfile -o Profiles/%1.prof code/avalanche_sim.py --profile -v 20
snakeviz Profiles/%1.prof