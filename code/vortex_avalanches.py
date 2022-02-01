from typing import List, Optional
from simulation import Simulation, PickleClass, SAVE_LOCATION, BAR_FORMAT

import sys
from operator import itemgetter
from warnings import warn
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tqdm
from abc import ABC, abstractmethod

if any(arg in sys.argv for arg in ('--profile', '-p')):
    PROFILING = True
else:
    PROFILING = False

@dataclass
class AvalancheResult(PickleClass):
    """Data class to store the results of a simulation"""
    values: list
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
    flattened_values: List[np.ndarray] = field(init=False)
    flattened_num_t: int = field(init=False)
    
    def __post_init__(self):
        self.size_ary = np.array([self.x_size, self.y_size])
        self.vortices_added = len(self.values)
        self.flattened_values = self._flatten()
        self.flattened_num_t = len(self.flattened_values)
        shapes = [values.shape for values in self.flattened_values]
        self.max_vortices = max(shapes, key=itemgetter(0))[0]
        
    def _flatten(self):
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
        
    def _vortices_force(self, vortex_pos, other_pos, cutoff):
        # Add an image to stop particles leaving the left side
        other_pos = np.append(other_pos, [[-vortex_pos[0], vortex_pos[1]]], axis=0)
        repulsive_force = super()._vortices_force(vortex_pos, other_pos, cutoff)
        attractive_force = self._pinning_force(vortex_pos)
        
        return repulsive_force + attractive_force
       
    @abstractmethod 
    def _pinning_force(self, vortex_pos: np.ndarray):
        """Calculates the force on a vortex due to the attractive pinning sites."""
        raise NotImplementedError
    
    def _handle_edges(self):
        # Only wrap in the y-direction
        np.mod(self.vortices, self.size_ary, out=self.vortices, where=[False, True])
        
    def run_vortex_sim(self, total_added: int, dt: float, force_cutoff: float, movement_cutoff: float,
                       cutoff_time: int = 1, leave_pbar: bool = True, quiet: bool = False):
        # if dt*self.pinning_strength > movement_cutoff:
        #     warn(f'Pinning force is greater than allowed movement for given dt ({dt*self.pinning_strength} > {movement_cutoff}). Pinned vortices may move too much to be deemed stationary.')
        # Record the positions of the vortices at each time step
        result_vals = []
        if self.random_gen is None:
            self.random_gen: np.random.Generator = np.random.default_rng()
        
        if quiet:
            iterator = range(total_added)
        else:
            # Loop with progress bar
            iterator = tqdm.tqdm(range(total_added), desc='Simulating', bar_format=BAR_FORMAT, leave=leave_pbar)
        count = 0
        for i in iterator:
            # Add a new vortex within the first lattice spacing 
            self.add_vortex(self.random_gen.uniform(0, 1), self.random_gen.uniform(0, self.y_size))
            new_result_lst = []
            new_result_ary = self.vortices.copy()[np.newaxis, ...]
            print()
            while True:
                count += 1
                self._step(dt, force_cutoff)
                new_result_ary = np.append(new_result_ary, self.vortices[np.newaxis, ...], axis=0)
                
                # Check for no movement
                if new_result_ary.shape[0] > cutoff_time:
                    displacement = new_result_ary[-1] - new_result_ary[-(cutoff_time+1)]
                    distance = np.linalg.norm(displacement, axis=1)
                    print(f'Vortex {np.argmax(distance):>3}: {str(displacement[np.argmax(distance)]):<35}{movement_cutoff}', end='\r')
                    if np.all(distance < movement_cutoff):
                        new_result_lst.append(new_result_ary)
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
            else:
                new_result_lst.append(new_result_ary)
            result_vals.append(new_result_lst)
            
        return AvalancheResult(result_vals, self.pinning_sites, dt, self.x_size, self.y_size,
                               force_cutoff, movement_cutoff, cutoff_time,
                               self.pinning_size, self.pinning_strength)
            
class StepAvalancheSim(VortexAvalancheBase):
    def _pinning_force(self, vortex_pos):
        displacement = self.pinning_sites - vortex_pos
        distances = np.linalg.norm(displacement, axis=1, keepdims=True)
        force_strength = self.pinning_strength*(distances<self.pinning_size)
        
        forces = force_strength*displacement/distances
        
        return np.sum(forces, axis=0)
            
class AvalancheAnimator:    
    def animate(self, result: AvalancheResult, filename, anim_freq=1):
        self._result = result
        
        n_steps = self._result.flattened_num_t//anim_freq
        self._anim_freq = anim_freq
        
        fig, _ = self._anim_init(self._result.max_vortices)
        
        animator = anim.FuncAnimation(fig, self._anim_update, n_steps, blit=True)
        
        self._p_bar = tqdm.tqdm(total=n_steps+1, desc='Animating ', unit='fr', bar_format=BAR_FORMAT)
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
        values = self._result.flattened_values[frame_num*self._anim_freq]
        num_vortices = values.shape[0]
        for i, dot in enumerate(self._dots):
            if i < num_vortices:
                dot.set_data(values[i])
            else:
                dot.set_data([], [])
            
        return self._dots
    
def test():
    sim = StepAvalancheSim.create_system(10, 4, 2, 4.4, 0.15, 3, 1005)
    result = sim.run_vortex_sim(50, 1e-3, 9, movement_cutoff=3e-3, cutoff_time=1)
    result.save('test_short')
    # result = AvalancheResult.load('test_short')
    print(f'{result.movement_cutoff = }, {result.movement_cutoff_time = }')
    freq = int(input(f'{result.flattened_num_t} to animate. Enter frequency: '))
    
    animator = AvalancheAnimator()
    animator.animate(result, 'vortex_test_short.gif', freq)
    
if __name__ == '__main__':
    test()
    