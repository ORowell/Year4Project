import getopt
import sys
from typing import Optional

import start_time
from avalanche_analysis_classes import AvalancheResult, BasicAvalancheResult
from avalanche_sim import StepAvalancheSim

PROFILING           = False         #     --profile
LENGTH              = 10            #     --length
WIDTH               = 4             #     --width
REPEATS             = 1             #     --repeats
PIN_DENSITY         = 5.5           # -d, --density
PIN_SIZE            = 0.15          #     --pin_radius
PIN_STRENGTH        = 3.            #     --pin_force
NOISE_FACTOR        = 0.2           #     --noise
SEED                = 1001          # -s, --seed
DT                  = 1e-4          #     --dt
REL_STOP_SPEED      = 0.1           #     --rel_stop_speed
INIT_REL_STOP_SPEED = 0.5           #     --init_rel_stop_speed
INIT_NUM_VORTICES   = 250           # -i, --init_vortices
NUM_VORTICES        = 1000          # -v, --vortices
NAME                = ''            # -n, --name
WALL_TIME           = 0             # -t, --wall_time
START_FROM          = None          #     --start_from
COMPRESS            = None          #     --compress
PRINT_AFTER         = None          #     --print_after
MAX_TIME            = None          #     --max_time

BUFFER_TIME = 1*60*60 # Time in seconds to allow for saving + timing discrepancies

if __name__ == '__main__':
    argv = sys.argv[1:]
    print(f'Running {sys.argv[0]} with arguments', argv, flush=True)
    opts, args = getopt.getopt(argv, 'd:s:i:v:n:t:',
                               ['profile', 'length=', 'width=', 'repeats=',
                               'density=', 'pin_radius=', 'pin_force=',
                               'seed=', 'dt=', 'rel_stop_speed=', 'init_rel_stop_speed=',
                               'init_vortices=', 'vortices=', 'name=', 'compress=',
                               'print_after=', 'max_time=', 'start_from=', 'wall_time=',
                               'noise='])
    for opt, arg in opts:
        if opt == '--profile':
            PROFILING = True
            print(f'Setting {PROFILING = }')
        elif opt == '--length':
            LENGTH = int(arg)
            print(f'Setting {LENGTH = }')
        elif opt == '--width':
            WIDTH = int(arg)
            print(f'Setting {WIDTH = }')
        elif opt == '--repeats':
            REPEATS = int(arg)
            print(f'Setting {REPEATS = }')
        elif opt in ('-d', '--density'):
            PIN_DENSITY = float(arg)
            print(f'Setting {PIN_DENSITY = }')
        elif opt == '--pin_radius':
            PIN_SIZE = float(arg)
            print(f'Setting {PIN_SIZE = }')
        elif opt == '--pin_force':
            PIN_STRENGTH = float(arg)
            print(f'Setting {PIN_STRENGTH = }')
        elif opt == '--noise':
            NOISE_FACTOR = float(arg)
            print(f'Setting {NOISE_FACTOR = }')
        elif opt in ('-s', '--seed'):
            SEED = int(arg)
            print(f'Setting {SEED = }')
        elif opt == '--dt':
            DT = float(arg)
            print(f'Setting {DT = }')
        elif opt == '--init_rel_stop_speed':
            INIT_REL_STOP_SPEED = float(arg)
            print(f'Setting {INIT_REL_STOP_SPEED = }')
        elif opt in ('-i', '--init_vortices'):
            INIT_NUM_VORTICES = int(arg)
            print(f'Setting {INIT_NUM_VORTICES = }')
        elif opt == '--rel_stop_speed':
            REL_STOP_SPEED = float(arg)
            print(f'Setting {REL_STOP_SPEED = }')
        elif opt in ('-v', '--vortices'):
            NUM_VORTICES = int(arg)
            print(f'Setting {NUM_VORTICES = }')
        elif opt in ('-t', '--wall_time'):
            WALL_TIME = int(arg)
            print(f'Setting {WALL_TIME = }')
        elif opt in ('-n', '--name'):
            NAME = arg
            print(f'Setting {NAME = }')
        elif opt == '--compress':
            COMPRESS = int(arg)
            print(f'Setting {COMPRESS = }')
        elif opt == '--print_after':
            PRINT_AFTER = int(arg)
            print(f'Setting {PRINT_AFTER = }')
        elif opt == '--max_time':
            MAX_TIME = int(arg)
            print(f'Setting {MAX_TIME = }')
        elif opt == '--start_from':
            START_FROM = arg
            print(f'Setting {START_FROM = }')
    sys.stdout.flush()

MOVEMENT_CUTOFF = REL_STOP_SPEED * PIN_STRENGTH * DT
INIT_MOVEMENT_CUTOFF = INIT_REL_STOP_SPEED * PIN_STRENGTH * DT
if WALL_TIME != 0:
    WALL_TIME -= BUFFER_TIME
    if WALL_TIME < 0:
        print('Wall time is greater than buffer time. Removing wall time', flush=True)
        WALL_TIME = 0
if NAME == '':
    NAME = f'test{SEED}'

def main(*, length: int = LENGTH, width: int = WIDTH, repeats: int = REPEATS, density: float = PIN_DENSITY,
         pin_size: float = PIN_SIZE, pin_strength: float = PIN_STRENGTH, seed: int = SEED, dt: float = DT,
         movement_cutoff: float = MOVEMENT_CUTOFF, num_vortices: int = NUM_VORTICES, max_time: Optional[int] = MAX_TIME,
         init_movement_cutoff: float = INIT_MOVEMENT_CUTOFF, init_num_vortices: int = INIT_NUM_VORTICES,
         name: str = NAME, compress: Optional[int] = COMPRESS, print_after: Optional[int] = PRINT_AFTER,
         start_from: Optional[str] = START_FROM, wall_time: float = WALL_TIME, noise_factor: float = NOISE_FACTOR):
    if start_from is None:
        print('Creating initial simulation', flush=True)
        init_sim = StepAvalancheSim.create_system(length, width, repeats, density, pin_size,
                                                  pin_strength, seed, noise_factor)
        print('Running initial simulation', flush=True)
        init_result = init_sim.run_vortex_sim(init_num_vortices, dt, 9, init_movement_cutoff, 100,
                                              print_after=print_after, max_time_steps=max_time,
                                              wall_time=wall_time)
        
        print('Creating main simulation', flush=True)
        main_sim = StepAvalancheSim.continue_from(init_result)
        init_result.save(name+'_init')
        del init_result
    else:
        # Continue from a past state
        print(f'Loading past result at {start_from}', flush=True)
        try:
            past_result = AvalancheResult.load(start_from)
            main_sim = StepAvalancheSim.continue_from(past_result)
        except FileNotFoundError:
            past_result = BasicAvalancheResult.load(start_from)
            main_sim = StepAvalancheSim.continue_from_basic(past_result)
    print('Running main simulation', flush=True)
    result = main_sim.run_vortex_sim(num_vortices, dt, 9, movement_cutoff, 100,
                                     print_after=print_after, max_time_steps=max_time, save_comp=10,
                                     wall_time=wall_time)
    
    # Compress the results before saving
    if compress is None:
        print('Storing results in basic form')
        result = result.to_basic_result()
    elif compress != 1:
        print(f'Compressing results file by factor of {compress}', flush=True)
        result = result.compress(compress)
    result.save(name)


if __name__ == '__main__':
    main()
