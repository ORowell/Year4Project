from typing import List, Optional
from simulation import Simulation, PickleClass, BAR_FORMAT

import sys
import os
import getopt
from operator import itemgetter
from warnings import warn
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tqdm
from abc import ABC, abstractmethod

PROFILING       = False         # -p, --profile
LENGTH          = 10            # -l, --length
WIDTH           = 4             # -w, --width
REPEATS         = 2             #     --repeats
PIN_DENSITY     = 4.4           # -d, --density
PIN_SIZE        = 0.15          # -r, --pin_radius
PIN_STRENGTH    = 3.            # -f, --pin_force
SEED            = 1001          # -s, --seed
DT              = 1e-4          # -t, --dt
REL_STOP_SPEED  = 1.            #     --rel_stop_speed
ADDED_VORTICES  = 250           # -v, --vortices
NAME            = f'test{SEED}' # -n, --name
ANIMATE         = False         # -a, --animate
LOAD_FILE       = False         #     --load

argv = sys.argv[1:]
opts, args = getopt.getopt(argv, 'pl:w:d:r:f:s:t:v:n:a',
                           ['profile', 'length=', 'width=', 'repeats=',
                            'density=', 'pin_radius=', 'pin_force=',
                            'seed=', 'dt=', 'rel_stop_speed=', 'vortices=',
                            'name=', 'animate', 'load'])

for opt, arg in opts:
    if opt in ('-p', '--profile'):
        PROFILING = True
    elif opt in ('-l', '--length'):
       LENGTH = int(arg)
    elif opt in ('-w', '--width'):
       WIDTH = int(arg)
    elif opt == '--repeats':
       REPEATS = int(arg)
    elif opt in ('-d', '--density'):
       PIN_DENSITY = float(arg)
    elif opt in ('-r', '--pin_radius'):
       PIN_SIZE = float(arg)
    elif opt in ('-f', '--pin_force'):
       PIN_STRENGTH = float(arg)
    elif opt in ('-s', '--seed'):
       SEED = int(arg)
    elif opt in ('-t', '--dt'):
       DT = float(arg)
    elif opt == '--rel_stop_speed':
       REL_STOP_SPEED = float(arg)
    elif opt in ('-v', '--vortices'):
       ADDED_VORTICES = int(arg)
    elif opt in ('-n', '--name'):
       NAME = arg
    elif opt in ('-a', '--animate'):
        ANIMATE = True
    elif opt == '--load':
        LOAD_FILE = True

MOVEMENT_CUTOFF = REL_STOP_SPEED * PIN_STRENGTH * DT
if PROFILING:
    ANIMATE = False
    LOAD_FILE = False

@dataclass
class AvalancheResult(PickleClass):
    """Data class to store the results of a simulation"""
    values: List[List[np.ndarray]]
    removed_vortices: List[List[int]]
    pinning_sites: np.ndarray
    dt: float
    x_size: float
    y_size: float
    force_cutoff: float
    movement_cutoff: float
    movement_cutoff_time: int
    pinning_size: float
    pinning_strength: float
    size_ary: np.ndarray = field(init=False)
    vortices_added: int = field(init=False)
    flattened_num_t: int = field(init=False)
    
    def __post_init__(self):
        self.size_ary = np.array([self.x_size, self.y_size])
        self.vortices_added = len(self.values)
        flattened_values = self.flatten
        self.flattened_num_t = len(flattened_values)
        shapes = [values.shape for values in flattened_values]
        self.max_vortices = max(shapes, key=itemgetter(0))[0]
        
    @property
    def flatten(self):
        """Return the result as just a list of varying array shape.
        
        This will remove the result data for just before a vortex is removed
        after going off the end so should not be used for analysis of vortex
        movement.
        """
        output: List[np.ndarray] = []
        for vortex_add_lst in self.values:
            for ary in vortex_add_lst:
                # Don't include the in between data (just before a vortex is deleted)
                ary = ary[:-1]
                for data in ary:
                    output.append(data)
            # But do include the final one after movement has ended
            output.append(vortex_add_lst[-1][-1])
        return output
    
    def compress(self, freq, keep_final=False):
        """Creates a new object only including a every `freq`'th time-step.
        The `dt` attribute will also be multiplied by `freq` accordingly.
        
        If `keep_final` is `True` the last time step of each array will always
        be retained to so that comparisons between initial and final states
        remain valid. However, this will result in a different `dt` between the
        last two data points.
        """
        new_vals = []
        for lst in self.values:
            new_lst = []
            for ary in lst:
                if keep_final:
                    # Remove final value before shortening and then add back on
                    shortened_vals = np.append(ary[:-1:freq, ...], ary[np.newaxis, :-1, ...], axis=0)
                else:
                    shortened_vals = ary[::freq, :, ...]
                new_lst.append(shortened_vals)
            new_vals.append(new_lst)
        
        return AvalancheResult(new_vals, self.removed_vortices,self.pinning_sites, self.dt*freq,
                               self.x_size, self.y_size, self.force_cutoff, self.movement_cutoff,
                               self.movement_cutoff_time, self.pinning_size, self.pinning_strength)

class VortexAvalancheBase(Simulation, ABC):
    def __init__(self, x_num: int, y_num: int, x_repeats: int, y_repeats: int,
                 pin_size: float, pin_strength: float):
        super().__init__(x_num, y_num, x_repeats, y_repeats)
        
        self.pinning_sites = np.empty(shape=(0, 2))
        self.pinning_size = pin_size
        self.pinning_strength = pin_strength
        
    @classmethod
    def create_system(cls, x_size: int, y_size: int, repeats: int, pinned_density: float,
                      pin_size: float, pin_strength: float, random_seed: Optional[int] = None):
        obj = cls(x_size, y_size, 0, repeats, pin_size, pin_strength)
        num_pins = int(obj.x_size * obj.y_size * pinned_density)
        obj.pinning_sites = np.append(obj.pinning_sites, obj._generate_random_pos(num_pins, random_seed, 1), axis=0)
        
        return obj
        
    def _vortices_force(self, vortex_pos, other_pos, vortex_index, cutoff):
        # Add an image to stop particles leaving the left side
        image_force_size = self._bessel_func(2*vortex_pos[0])
        image_force = np.array([image_force_size, 0])
        
        repulsive_force = super()._vortices_force(vortex_pos, other_pos, vortex_index, cutoff)
        attractive_force = self._pinning_force(vortex_pos)
        
        return repulsive_force + attractive_force + image_force
       
    @abstractmethod 
    def _pinning_force(self, vortex_pos: np.ndarray):
        """Calculates the force on a vortex due to the attractive pinning sites."""
        raise NotImplementedError
    
    def _handle_edges(self):
        # Only wrap in the y-direction
        np.mod(self.vortices, self.size_ary, out=self.vortices, where=[False, True])
        
    def run_vortex_sim(self, total_added: int, dt: float, force_cutoff: float, movement_cutoff: float,
                       cutoff_time: int = 1, include_pbar: bool = True, quiet: bool = False):
        # if dt*self.pinning_strength > movement_cutoff:
        #     warn(f'Pinning force is greater than allowed movement for given dt ({dt*self.pinning_strength} > {movement_cutoff}). Pinned vortices may move too much to be deemed stationary.')
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
        count = 0
        for i in iterator:
            new_removed_vortex_lst: List[int] = []
            # Add a new vortex within the first lattice spacing
            self.add_vortex(self.random_gen.uniform(0, 1), self.random_gen.uniform(0, self.y_size))
            new_result_lst = []
            new_result_ary = self.vortices.copy()[np.newaxis, ...]
            if include_pbar and not quiet:
                print()
            while True:
                count += 1
                self._step(dt, force_cutoff)
                new_result_ary = np.append(new_result_ary, self.vortices[np.newaxis, ...], axis=0)
                
                # Check for no movement
                if new_result_ary.shape[0] > cutoff_time:
                    displacement = new_result_ary[-1] - new_result_ary[-(cutoff_time+1)]
                    distance = np.linalg.norm(displacement, axis=1)
                    if not quiet:
                        print(f'Vortex {np.argmax(distance):>3}: {str(displacement[np.argmax(distance)]):<35}{movement_cutoff}', end='\r')
                    if np.all(distance < movement_cutoff):
                        new_result_lst.append(new_result_ary)
                        if include_pbar and not quiet:
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
            
        return AvalancheResult(result_vals, removed_vortices,self.pinning_sites, dt,
                               self.x_size, self.y_size, force_cutoff, movement_cutoff,
                               cutoff_time, self.pinning_size, self.pinning_strength)
            
class StepAvalancheSim(VortexAvalancheBase):
    def _pinning_force(self, vortex_pos):
        displacement = self.pinning_sites - vortex_pos
        # distances = np.linalg.norm(displacement, axis=1, keepdims=True)
        # force_strength = self.pinning_strength*(distances<self.pinning_size)
        
        # forces = force_strength*displacement/distances
        distances = np.linalg.norm(displacement, axis=1)
        directions = displacement/distances[:, np.newaxis]
        active_pins = directions[distances<self.pinning_size]
        forces = self.pinning_strength*active_pins
        
        return np.sum(forces, axis=0)
            
class AvalancheAnimator:    
    def animate(self, result: AvalancheResult, filename, anim_freq=1):
        self._result = result
        
        n_steps = self._result.flattened_num_t//anim_freq
        self._anim_freq = anim_freq
        
        fig, _ = self._anim_init(self._result.max_vortices)
        
        animator = anim.FuncAnimation(fig, self._anim_update, n_steps, blit=True)
        
        self._p_bar = tqdm.tqdm(total=n_steps+1, desc='Animating ', unit='fr', bar_format=BAR_FORMAT)
        if not os.path.exists('results\\gifs'):
            os.makedirs('results\\gifs')
        animator.save(f'results\\gifs\\{filename}', fps=30)
        self._p_bar.close()
        
    def _anim_init(self, max_vortices):
        fig = plt.figure(figsize=(10, 10*self._result.y_size/self._result.x_size))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([0, self._result.x_size])
        ax.set_ylim([0, self._result.y_size])
        
        self._dots = [ax.plot([], [], 'o', c='r')[0] for i in range(max_vortices)]
        
        for vortex in self._result.pinning_sites:
            ax.add_artist(plt.Circle(vortex, self._result.pinning_size, color='grey', alpha=0.3))
        
        return fig, ax
        
    def _anim_update(self, frame_num):
        self._p_bar.update(1)
        values = self._result.flatten[frame_num*self._anim_freq]
        num_vortices = values.shape[0]
        for i, dot in enumerate(self._dots):
            if i < num_vortices:
                dot.set_data(values[i])
            else:
                dot.set_data([], [])
            
        return self._dots
    
def main(length: int = LENGTH, width: int = WIDTH, repeats: int = REPEATS, density: float = PIN_DENSITY,
         pin_size: float = PIN_SIZE, pin_strength: float = PIN_STRENGTH, seed: int = SEED, dt: float = DT,
         movement_cutoff: float = MOVEMENT_CUTOFF, n_vortices: int = ADDED_VORTICES, name: str = NAME,
         animate: bool = ANIMATE, load_file: bool = LOAD_FILE):
    if not load_file:
        sim = StepAvalancheSim.create_system(length, width, repeats, density, pin_size, pin_strength, seed)
        result = sim.run_vortex_sim(n_vortices, dt, 9, movement_cutoff=movement_cutoff, quiet=True)
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