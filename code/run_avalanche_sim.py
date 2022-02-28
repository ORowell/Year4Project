from avalanche_sim import StepAvalancheSim
from avalanche_analysis_classes import AvalancheResult, BasicAvalancheResult

from typing import Optional
import sys
import getopt

PROFILING           = False         #     --profile
LENGTH              = 10            #     --length
WIDTH               = 4             #     --width
REPEATS             = 1             #     --repeats
PIN_DENSITY         = 5.5           # -d, --density
PIN_SIZE            = 0.15          #     --pin_radius
PIN_STRENGTH        = 3.            #     --pin_force
SEED                = 1001          # -s, --seed
DT                  = 1e-4          #     --dt
REL_STOP_SPEED      = 0.1           #     --rel_stop_speed
INIT_REL_STOP_SPEED = 0.5           #     --init_rel_stop_speed
INIT_NUM_VORTICES   = 200           # -i, --init_vortices
NUM_VORTICES        = 3000          # -v, --vortices
NAME                = ''            # -n, --name
COMPRESS            = None          #     --compress
PRINT_AFTER         = None          #     --print_after
MAX_TIME            = None          #     --max_time

if __name__ == '__main__':
    argv = sys.argv[1:]
    print(argv, flush=True)
    opts, args = getopt.getopt(argv, 'd:s:i:v:n:',
                            ['profile', 'length=', 'width=', 'repeats=',
                             'density=', 'pin_radius=', 'pin_force=',
                             'seed=', 'dt=', 'rel_stop_speed=', 'init_rel_stop_speed=',
                             'init_vortices=', 'vortices=', 'name=', 'compress=',
                             'print_after=', 'max_time='])
    for opt, arg in opts:
        if opt == '--profile':
            PROFILING = True
            print(f'Setting {PROFILING = }')
        elif opt in ('-l', '--length'):
            LENGTH = int(arg)
            print(f'Setting {LENGTH = }')
        elif opt in ('-w', '--width'):
            WIDTH = int(arg)
            print(f'Setting {WIDTH = }')
        elif opt == '--repeats':
            REPEATS = int(arg)
            print(f'Setting {REPEATS = }')
        elif opt in ('-d', '--density'):
            PIN_DENSITY = float(arg)
            print(f'Setting {PIN_DENSITY = }')
        elif opt in ('-r', '--pin_radius'):
            PIN_SIZE = float(arg)
            print(f'Setting {PIN_SIZE = }')
        elif opt in ('-f', '--pin_force'):
            PIN_STRENGTH = float(arg)
            print(f'Setting {PIN_STRENGTH = }')
        elif opt in ('-s', '--seed'):
            SEED = int(arg)
            print(f'Setting {SEED = }')
        elif opt in ('-t', '--dt'):
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
        elif opt in ('-n', '--name'):
            NAME = arg
            print(f'Setting {NAME = }')
        elif opt in ('-c', '--compress'):
            COMPRESS = int(arg)
            print(f'Setting {COMPRESS = }')
        elif opt in ('-p', '--print_after'):
            PRINT_AFTER = int(arg)
            print(f'Setting {PRINT_AFTER = }')
        elif opt == '--max_time':
            MAX_TIME = int(arg)
            print(f'Setting {MAX_TIME = }')
    sys.stdout.flush()

MOVEMENT_CUTOFF = REL_STOP_SPEED * PIN_STRENGTH * DT
INIT_MOVEMENT_CUTOFF = INIT_REL_STOP_SPEED * PIN_STRENGTH * DT
if NAME == '':
    NAME = f'test{SEED}'

def main(length: int = LENGTH, width: int = WIDTH, repeats: int = REPEATS, density: float = PIN_DENSITY,
         pin_size: float = PIN_SIZE, pin_strength: float = PIN_STRENGTH, seed: int = SEED, dt: float = DT,
         movement_cutoff: float = MOVEMENT_CUTOFF, num_vortices: int = NUM_VORTICES, max_time: Optional[int] = MAX_TIME,
         init_movement_cutoff: float = INIT_MOVEMENT_CUTOFF, init_num_vortices: int = INIT_NUM_VORTICES,
         name: str = NAME, compress: Optional[int] = COMPRESS, print_after: Optional[int] = PRINT_AFTER):
    print('Creating initial simulation', flush=True)
    init_sim = StepAvalancheSim.create_system(length, width, repeats, density, pin_size, pin_strength, seed)
    print('Running initial simulation', flush=True)
    init_result = init_sim.run_vortex_sim(init_num_vortices, dt, 9, init_movement_cutoff, 100,
                                          print_after=print_after, max_time_steps=max_time, save_comp=10)
    
    print('Creating main simulation', flush=True)
    main_sim = StepAvalancheSim.continue_from(init_result)
    print('Running main simulation', flush=True)
    result = main_sim.run_vortex_sim(num_vortices, dt, 9, movement_cutoff, 100,
                                     print_after=print_after, max_time_steps=max_time, save_comp=10)
    
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