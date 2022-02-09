from vortex_avalanches import AvalancheResult

import sys
import getopt

REPLACE_OLD = False     # -r, --replace_old
NAME        = None      # -n, --name
FREQ        = 10        # -f, --frequency
DIRECTORY   = None      # -d, --directory

argv = sys.argv[1:]
opts, args = getopt.getopt(argv, 'rn:f:d:',
                           ['replace_old', 'name=', 'frequency=',
                            'directory='])
for opt, arg in opts:
    if opt in ('-n', '--name'):
        NAME = arg
        print(f'Setting {NAME = }')
    if opt in ('-d', '--directory'):
        DIRECTORY = arg
        print(f'Setting {DIRECTORY = }')
    elif opt in ('-f', '--frequency'):
        FREQ = int(arg)
        print(f'Setting {FREQ = }')
    elif opt in ('-r', '--replace_old'):
        REPLACE_OLD = True
        print(f'Setting {REPLACE_OLD = }')
        
if NAME is None:
    raise RuntimeError('Name must be specified')

result = AvalancheResult.load(NAME, DIRECTORY)
if not isinstance(result, AvalancheResult):
    raise FileNotFoundError(f'Could not find file: {NAME}')
comp_result = result.compress(FREQ)
if REPLACE_OLD:
    comp_result.save(NAME, DIRECTORY)
else:
    comp_result.save(f'comp_{NAME}', DIRECTORY)