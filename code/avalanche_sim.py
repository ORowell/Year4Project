from typing import List, Optional
from simulation import Simulation, BAR_FORMAT, HALF_ROOT_3
from avalanche_analysis_classes import AvalancheResult, AvalancheAnimator

import sys
import getopt
from warnings import warn
import numpy as np
import tqdm
from math import ceil
from abc import ABC, abstractmethod

NP_FORMATTER = {'float': '{: .5e}'.format}

PROFILING       = False         #     --profile
LENGTH          = 10            # -l, --length
WIDTH           = 4             # -w, --width
REPEATS         = 1             #     --repeats
PIN_DENSITY     = 4.4           # -d, --density
PIN_SIZE        = 0.15          # -r, --pin_radius
PIN_STRENGTH    = 3.            # -f, --pin_force
SEED            = 1001          # -s, --seed
DT              = 1e-4          # -t, --dt
REL_STOP_SPEED  = 0.1           #     --rel_stop_speed
NUM_VORTICES    = 20           # -v, --vortices
NAME            = 'pin_cell_list_test'            # -n, --name
COMPRESS        = 1             # -c, --compress
START_FROM      = None          #     --start_from
ANIMATE         = True         # -a, --animate
LOAD_FILE       = False         #     --load
PRINT_AFTER     = None          #     --verbose

if __name__ == '__main__':
    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'l:w:d:r:f:s:t:v:n:c:ap:',
                            ['profile', 'length=', 'width=', 'repeats=',
                             'density=', 'pin_radius=', 'pin_force=',
                             'seed=', 'dt=', 'rel_stop_speed=', 'vortices=',
                             'name=', 'compress=', 'animate', 'load', 'print_after='])
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
        elif opt in ('-a', '--animate'):
            ANIMATE = True
            print(f'Setting {ANIMATE = }')
        elif opt == '--start_from':
            START_FROM = arg
            print(f'Setting {START_FROM = }')
        elif opt == '--load':
            LOAD_FILE = True
            print(f'Setting {LOAD_FILE = }')
        elif opt in ('-p', '--print_after'):
            PRINT_AFTER = int(arg)
            print(f'Setting {PRINT_AFTER = }')

MOVEMENT_CUTOFF = REL_STOP_SPEED * PIN_STRENGTH * DT
if PROFILING:
    ANIMATE = False
    LOAD_FILE = False
if NAME == '':
    NAME = f'test{SEED}_short'

class VortexAvalancheBase(Simulation, ABC):
    X_NEG_ARY = np.array([-1,1])
    
    def __init__(self, x_num: int, y_num: int, x_repeats: int, y_repeats: int,
                 pin_size: float, pin_strength: float):
        super().__init__(x_num, y_num, x_repeats, y_repeats)
        
        self.pinning_size = pin_size
        self.pinning_strength = pin_strength
        
        self._pin_x_cells = ceil(self.x_size/self.pinning_size)
        self._pin_y_cells = ceil(self.y_size/self.pinning_size)
        self._pinning_sites = np.empty(shape=(0, 2))
        self._celled_pin_site: List[List[np.ndarray]] = [[]]
        
    @property
    def pinning_sites(self):
        return self._pinning_sites
    
    @pinning_sites.setter
    def pinning_sites(self, pin_sites: np.ndarray):
        self._pinning_sites = pin_sites
        
        self._celled_pin_site = [[np.empty(shape=(0, 2)) for j in range(self._pin_x_cells)]
                                 for i in range(self._pin_y_cells)]
        cell_loc = np.floor_divide(pin_sites, self.pinning_size).astype(int)
        for pin_pos, pin_cell in zip(pin_sites, cell_loc):
            cell_x, cell_y = pin_cell
            self._celled_pin_site[cell_y][cell_x] = np.append(self._celled_pin_site[cell_y][cell_x],
                                                              pin_pos[np.newaxis, :], axis=0)
        
    @classmethod
    def create_system(cls, x_size: int, y_size: int, repeats: int, pinned_density: float,
                      pin_size: float, pin_strength: float, random_seed: Optional[int] = None):
        obj = cls(x_size, y_size, 0, repeats, pin_size, pin_strength)
        num_pins = int(obj.x_size * obj.y_size * pinned_density)
        obj.pinning_sites = np.append(obj.pinning_sites, obj._generate_random_pos(num_pins, random_seed, 1), axis=0)
        
        return obj
        
    @classmethod
    def continue_from(cls, past_result: AvalancheResult):
        obj = cls(int(past_result.x_size), int(past_result.y_size/HALF_ROOT_3), 0,
                  past_result.repeats, past_result.pinning_size, past_result.pinning_strength)
        obj.pinning_sites = past_result.pinning_sites
        obj.vortices = past_result.values[-1][-1]
        obj.random_gen = past_result.random_gen
        
    def _get_all_vortices(self):
        # Add mirror images to stop particles leaving the left side
        mirror_images = self.vortices * self.X_NEG_ARY
        all_vortices = np.append(self.vortices, mirror_images, axis=0)
        images = self._get_images(all_vortices)
        return np.concatenate((all_vortices, images))
        
    def _vortices_force(self, vortex_pos, other_pos, vortex_index, cutoff):
        repulsive_force = super()._vortices_force(vortex_pos, other_pos, vortex_index, cutoff)
        attractive_force = self._pinning_force(vortex_pos)
        
        return repulsive_force + attractive_force

    @abstractmethod 
    def _pinning_force(self, vortex_pos: np.ndarray):
        """Calculates the force on a vortex due to the attractive pinning sites."""
        raise NotImplementedError
    
    def _get_pin_sites(self, vortex_pos: np.ndarray):
        # return self._pinning_sites
        cell_x, cell_y = np.floor_divide(vortex_pos, self.pinning_size).astype(int)
        x_range = [0 if cell_x-1 < 0 else cell_x-1, -1 if cell_x+2 >= self._pin_x_cells else cell_x+2]
        # TODO: Allow for wrapping
        y_range = [0 if cell_y-1 < 0 else cell_y-1, -1 if cell_y+2 >= self._pin_y_cells else cell_y+2]
        pin_sites = np.empty(shape=(0, 2))
        for rows in self._celled_pin_site[y_range[0]:y_range[1]]:
            for cell in rows[x_range[0]:x_range[1]]:
                pin_sites = np.append(pin_sites, cell, axis=0)
        return pin_sites
    
    def _handle_edges(self):
        # Only wrap in the y-direction
        np.mod(self.vortices, self.size_ary, out=self.vortices, where=[False, True])
        
    def run_vortex_sim(self, total_added: int, dt: float, force_cutoff: float, movement_cutoff: float,
                       cutoff_time: int = 1, include_pbar: bool = True, print_after: Optional[int] = None):
        # if dt*self.pinning_strength > movement_cutoff:
        #     warn(f'Pinning force is greater than allowed movement for given dt ({dt*self.pinning_strength} > {movement_cutoff}). \
        #         Pinned vortices may move too much to be deemed stationary.')
        # Record the positions of the vortices at each time step
        result_vals = []
        removed_vortices: List[List[int]] = []
        if self.random_gen is None:
            self.random_gen: np.random.Generator = np.random.default_rng()
        
        if include_pbar:
            # Loop with progress bar
            iterator = tqdm.tqdm(range(total_added), desc='Simulating', bar_format=BAR_FORMAT)
        else:
            iterator = range(total_added)
        for i in iterator:
            new_removed_vortex_lst: List[int] = []
            # Add a new vortex within the first lattice spacing
            self.add_vortex(self.random_gen.uniform(0, 1), self.random_gen.uniform(0, self.y_size))
            new_result_lst = []
            new_result_ary = self.vortices.copy()[np.newaxis, ...]
            count = 0
            while True:
                count += 1
                self._step(dt, force_cutoff)
                new_result_ary = np.append(new_result_ary, self.vortices[np.newaxis, ...], axis=0)
                
                # Check for no movement
                if new_result_ary.shape[0] > cutoff_time:
                    displacement = new_result_ary[-1] - new_result_ary[-(cutoff_time+1)]
                    displacement = np.mod(displacement + self.size_ary/2, self.size_ary) - self.size_ary/2
                    distance = np.linalg.norm(displacement, axis=1)
                    
                    # If this event is taking a long time print out updates
                    if count == print_after:
                        print()
                    if print_after is not None and count >= print_after:
                        np_out = np.array2string(displacement[np.argmax(distance)], formatter=NP_FORMATTER)
                        print(f'Vortex {np.argmax(distance):>3}: {np_out:<35}{movement_cutoff:.2e}', end='\r')
                    
                    if np.all(distance < movement_cutoff*cutoff_time):
                        new_result_lst.append(new_result_ary)
                        if include_pbar and print_after is not None and count >= print_after:
                            sys.stdout.write("\x1b[1A")
                        break
                
                # Check for vortices that have gone over the right side
                to_keep = self.vortices[:, 0] < self.x_size
                self.vortices: np.ndarray = self.vortices[to_keep]
                # If any vortices have been removed save the current result array
                # and start a new array of different dimensions
                if not np.all(to_keep):
                    new_result_lst.append(new_result_ary)
                    new_result_ary = self.vortices.copy()[np.newaxis, ...]
                    # Get the indices of all the vortices that have been removed
                    removed_indices, = np.nonzero(np.logical_not(to_keep))
                    new_removed_vortex_lst.extend(removed_indices[::-1])
            else:
                new_result_lst.append(new_result_ary)
            result_vals.append(new_result_lst)
            removed_vortices.append(new_removed_vortex_lst)
            
        return AvalancheResult(result_vals, removed_vortices, self.pinning_sites, dt,
                               self.x_size, self.y_size, self.y_images, self.random_gen,
                               force_cutoff, movement_cutoff, cutoff_time,
                               self.pinning_size, self.pinning_strength)

class StepAvalancheSim(VortexAvalancheBase):
    def _pinning_force(self, vortex_pos):
        displacement = self._get_pin_sites(vortex_pos) - vortex_pos
        distances = np.linalg.norm(displacement, axis=1)
        directions = displacement/distances[:, np.newaxis]
        
        active_pins = directions[distances<self.pinning_size]
        forces = self.pinning_strength*active_pins
        
        return np.sum(forces, axis=0)
    
def main(length: int = LENGTH, width: int = WIDTH, repeats: int = REPEATS, density: float = PIN_DENSITY,
         pin_size: float = PIN_SIZE, pin_strength: float = PIN_STRENGTH, seed: int = SEED, dt: float = DT,
         movement_cutoff: float = MOVEMENT_CUTOFF, num_vortices: int = NUM_VORTICES, name: str = NAME,
         compress: int = COMPRESS, animate: bool = ANIMATE, start_from: Optional[str] = None,
         load_file: bool = LOAD_FILE, print_after: Optional[int] = PRINT_AFTER):
    if not load_file:
        if start_from is None:
            sim = StepAvalancheSim.create_system(length, width, repeats, density, pin_size, pin_strength, seed)
        else:
            # Continue from a past state
            past_result = AvalancheResult.load(start_from)
            sim = StepAvalancheSim.continue_from(past_result)
            dt = past_result.dt
        result = sim.run_vortex_sim(num_vortices, dt, 9, movement_cutoff=movement_cutoff, print_after=print_after, cutoff_time=100)
        # Compress the results before saving
        if compress != 1:
            result = result.compress(compress)
        if not PROFILING:
            result.save(name)
    else:
        result = AvalancheResult.load(name)
        if result is None:
            raise FileNotFoundError(f'Could not locate stored result {name}')
        
    if animate:
        print(f'{result.movement_cutoff = }, {result.movement_cutoff_time = }')
        print(f'{result.dt = }, {result.vortices_added = }')
        freq = int(input(f'{result.flattened_num_t} to animate. Enter frequency: '))
        
        animator = AvalancheAnimator()
        animator.animate(result, f'vortex_{name}.gif', freq)
    
if __name__ == '__main__':
    main()
