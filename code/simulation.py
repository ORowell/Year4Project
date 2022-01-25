from typing import Optional, Type, TypeVar
import os.path
import numpy as np
import scipy.special as scipy_s
import tqdm
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from dataclasses import dataclass, field

BAR_FORMAT = '{desc}: {percentage:5.1f}%|{bar}{r_bar}'
HALF_ROOT_3 = np.sqrt(3)/2
SAVE_LOCATION = 'results\\Simulation_results\\{cls.__name__}'
FILE_LOCATION = SAVE_LOCATION + '\\{filename}'


T = TypeVar('T', bound='PickleClass')
class PickleClass:
    def save(self, filename: str):
        with open(FILE_LOCATION.format(cls=self.__class__, filename=filename), 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls: Type[T], filename: str, quiet=False) -> Optional[T]:
        path = FILE_LOCATION.format(cls=cls, filename=filename)
        if not os.path.exists(path):
            if not quiet:
                print(f'{path} not found')
            return None
        with open(FILE_LOCATION.format(cls=cls, filename=filename), 'rb') as f:
            return pickle.load(f)

@dataclass
class SimResult(PickleClass):
    """Data class to store the results of a simulation"""
    values: np.ndarray
    dt: float
    x_size: float
    y_size: float
    cutoff: float
    num_t: int = field(init=False)
    num_vortices: int = field(init=False)
    t_max: float = field(init=False)
    size_ary: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.num_t, self.num_vortices, _ = self.values.shape
        self.t_max = self.dt * self.num_t
        self.size_ary = np.array([self.x_size, self.y_size])
        
    def get_average_velocity(self, start_index=0):
        diff = self.values[start_index+1:, :, :] - self.values[start_index:-1, :, :]
        diff = np.mod(diff + self.size_ary/2, self.size_ary) - self.size_ary/2
        avg_diff = np.mean(diff, (0, 1))
        
        return avg_diff / self.dt


class Simulation:
    result_type = SimResult
    
    def __init__(self, x_num: int, y_num: int, x_repeats: int, y_repeats: int):
        # Give the vortices array the right shape to allow appending
        self.vortices: np.ndarray = np.empty(shape=(0, 2))
        
        self.x_size = x_num
        self.y_size = y_num * HALF_ROOT_3
        self.x_size_ary = np.array([self.x_size, 0])
        self.y_size_ary = np.array([0, self.y_size])
        self.size_ary = np.array([self.x_size, self.y_size])
        
        self.x_images = x_repeats
        self.y_images = y_repeats
        
        self.current_force = np.array([0, 0])
        self.random_gen: Optional[np.random.Generator] = None
    
    def add_vortex(self, x_pos: float, y_pos: float):
        """Add a vortex at the given x and y position"""
        self.vortices = np.append(self.vortices, [[x_pos, y_pos]], axis=0)
        
    def create_random_vortices(self, num_vortices: int, seed: Optional[int] = None):
        """Create a given number of vortices at random locations"""
        self.vortices = np.append(self.vortices,
                                  self._generate_random_pos(num_vortices, seed), axis=0)
        
    def _generate_random_pos(self, num_pos: int, seed: Optional[int] = None,
                             min_x: float = 0, min_y: float = 0):
        """Create a given number of random positions"""
        if self.random_gen is None or seed is not None:
            # Allow usage of a seed for consistent results, eg. benchmarking
            self.random_gen = np.random.default_rng(seed)
        
        x_vals = self.random_gen.uniform(min_x, self.x_size, (num_pos, 1))
        y_vals = self.random_gen.uniform(min_y, self.y_size, (num_pos, 1))
        
        return np.append(x_vals, y_vals, axis=1)
        
    def add_triangular_lattice(self, corner, rows: int,
                               cols: int, offset: bool=False):
        """Generate a triangular vortex lattice at a given location.
        
        offset controls whether the bottom left vortex is at corner
        or offset by half a unit"""
        corner = np.array(corner)
        self.vortices = np.append(self.vortices,
                                  self._generate_lattice_pos(corner, rows, cols, offset), axis=0)
            
    def _generate_lattice_pos(self, corner: np.ndarray, rows: int,
                              cols: int, offset: bool=False):
        new_vortices: np.ndarray = np.empty(shape=(0, 2))
        for i in range(rows):
            # X vals should be offset every other row (ie. mod 2)
            x_vals = np.arange(cols) + 0.5*((i+offset)%2)
            y_vals = np.full_like(x_vals, i*HALF_ROOT_3)
            
            # Join the values and add the overall shift
            new_lattice = np.stack((x_vals, y_vals), axis=1) + corner
            
            new_vortices = np.append(new_vortices, new_lattice, axis=0)
        
        return new_vortices
    
    def run_sim(self, total_time: float, dt: float, cutoff: float,
                leave_pbar: bool=True, quiet: bool=False):
        num_steps = int(total_time/dt)
        # Record the positions of the vortices at each time step
        result_vals = np.empty((num_steps+1, *self.vortices.shape))
        result_vals[0] = self.vortices.copy()
        
        if quiet:
            iterator = range(num_steps)
        else:
            # Loop with progress bar
            iterator = tqdm.tqdm(range(num_steps), desc='Simulating', bar_format=BAR_FORMAT, leave=leave_pbar)
        
        for i in iterator:
            self._step(dt, cutoff)
            result_vals[i+1] = self.vortices.copy()
            
        return SimResult(result_vals, dt, self.x_size, self.y_size, cutoff)
        
    def _step(self, dt: float, cutoff: float):
        all_vortices = self._get_all_vortices()
        new_vortices = self.vortices.copy()
        # Could be done by numpy without a loop or in parallel?
        for i, vortex in enumerate(self.vortices):
            # Don't allow a vortex to act on itself
            acting_vortices = np.delete(all_vortices, i, axis=0)
            force = self._vortices_force(vortex, acting_vortices, cutoff)
            
            new_vortices[i] += dt*force
        
        new_vortices += dt * self.current_force
        self.vortices = new_vortices
        self._handle_edges()
        
    def _get_all_vortices(self):
        """Get an array of every vortex (real or images) that could apply a force.
        This must start with the real vortices in order.
        """
        images = self._get_images(self.vortices)
        return np.concatenate((self.vortices, images))
    
    def _get_images(self, vortices):
        """Return an array of the images of each vortex in vortices"""
        # Find the offsets for each "tile" of images
        x_offset = np.repeat(np.arange(-self.x_images, self.x_images+1), 2*self.y_images+1) * self.x_size
        y_offset = np.tile(np.arange(-self.y_images, self.y_images+1), 2*self.x_images+1) * self.y_size
        
        # Remove the (0, 0) cases (non-image)
        x_offset = np.delete(x_offset, x_offset.size//2, axis=0)
        y_offset = np.delete(y_offset, y_offset.size//2, axis=0)
        
        # Add the offsets to each vortex
        x_images = (x_offset[np.newaxis, :] + vortices[:, 0][:, np.newaxis]).flatten()
        y_images = (y_offset[np.newaxis, :] + vortices[:, 1][:, np.newaxis]).flatten()
              
        return np.stack((x_images, y_images), axis=1)
    
    def _vortices_force(self, vortex_pos, other_pos, cutoff):
        rel_pos = vortex_pos - other_pos
        distances = np.linalg.norm(rel_pos, axis=1, keepdims=True)
        directions = rel_pos/distances
        
        # Don't calculate for long distances
        if cutoff:
            large_distances = distances > cutoff
            distances[large_distances] = 0
        force_sizes = self._bessel_func(distances)
        if cutoff:
            force_sizes[large_distances] = 0
        forces = force_sizes * directions
        
        return np.sum(forces, axis=0)
    
    @staticmethod
    def _bessel_func(val):
        return scipy_s.k1(val)
    
    def _handle_edges(self):
        """Handle any particles that have gone off the edge of the simulation.
        By default they are just wrapped around to the other side."""
        np.mod(self.vortices, self.size_ary, out=self.vortices)
        
        
class SimAnimator:    
    def animate(self, result: SimResult, filename, anim_freq=1):
        self._result = result
        
        n_steps = self._result.num_t//anim_freq
        self._anim_freq = anim_freq
        
        fig, _ = self._anim_init(self._result.num_vortices)
        
        animator = anim.FuncAnimation(fig, self._anim_update, n_steps, blit=True)
        
        self._p_bar = tqdm.tqdm(total=n_steps+1, desc='Animating ', unit='fr', bar_format=BAR_FORMAT)
        animator.save(f'results\\gifs\\{filename}', fps=30)
        self._p_bar.close()
        
    def _anim_init(self, num_vortices):
        fig = plt.figure(figsize=(10, 10*self._result.y_size/self._result.x_size))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([0, self._result.x_size])
        ax.set_ylim([0, self._result.y_size])
        
        self._dots = [ax.plot([], [], 'o', c='r')[0] for i in range(num_vortices)]
        
        return fig, ax
        
    def _anim_update(self, frame_num):
        self._p_bar.update(1)
        for i, dot in enumerate(self._dots):
            dot.set_data(self._result.values[frame_num*self._anim_freq, i])
            
        return self._dots
    
def ground_state():
    images = 8
    sim = Simulation(2, 2, images, images)
    
    sim.add_triangular_lattice((0, 0), 2, 2)
    
    result = sim.run_sim(0.2, 0.0001, 0)
    
    print(images)
    print(np.mean(np.linalg.norm(result.values[-1] - result.values[0], axis=1)))
    
    # animator = SimAnimator()
    # animator.animate(result, 'groundstate.gif', 10)
    
def ground_state_from_rand():
    sim = Simulation(2, 2, 6, 6)
    
    # sim.add_triangular_lattice((0, 0), 2, 2)
    sim.create_random_vortices(4, seed=120)
    
    result = sim.run_sim(15, 0.01, 0)
    
    animator = SimAnimator()
    animator.animate(result, 'groundstate_rand_start.gif', 2)
    
def many_vortices():
    sim = Simulation(15, 10, 1, 1)
    
    sim.create_random_vortices(150, seed=120)
    
    result = sim.run_sim(10, 0.01, 0)
    
    animator = SimAnimator()
    animator.animate(result, 'lots.gif')
    
if __name__ == '__main__':
    # ground_state()
    ground_state_from_rand()
    # many_vortices()
