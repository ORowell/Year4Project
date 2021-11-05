from typing import Iterable, List
from simulation import Simulation, HALF_ROOT_3, SimResult, SimAnimator, SAVE_LOCATION

import os
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Default values
PINNED_DEPTH = 4
CHANNEL_LENGTH = 10
T_MAX = 200
T_STEPS = 1e4
REPEATS = 1

ANALYTICAL_FC = 0.050018

@dataclass
class ChannelSimResult(SimResult):
    pinned_vortices: np.ndarray
    
    @classmethod
    def from_SimResult(cls, simresult: SimResult, pinned_vortices):
        return cls(simresult.values, simresult.dt, simresult.x_size, simresult.y_size, pinned_vortices)


class ChannelSimulation(Simulation):
    def __init__(self, x_num, y_num, x_repeats, y_repeats):
        super().__init__(x_num, y_num, x_repeats, y_repeats)
        
        self.pinned_vortices = np.empty(shape=(0, 2))
        self.pinned_images = np.empty(shape=(0, 2))
        
    @classmethod
    def create_channel(cls, channel_width, pinned_width, channel_length, repeats):
        """Create a system channel of free vortices amongst pinned vortices"""
        obj = cls(channel_length, channel_width + 2*pinned_width, repeats, 0)
        
        obj.add_pinned_lattice((0, HALF_ROOT_3/2), pinned_width, channel_length, pinned_width%2)
        top_channel_base = (channel_width + pinned_width + 0.5) * HALF_ROOT_3
        obj.add_pinned_lattice((0, top_channel_base), pinned_width, channel_length, channel_width%2)
        
        obj.add_triangular_lattice((0, (pinned_width + 0.5)*HALF_ROOT_3), channel_width, channel_length)
        
        return obj
        
    def _get_all_vortices(self):
        real_vortices = super()._get_all_vortices()
        return np.concatenate((real_vortices, self.pinned_vortices, self.pinned_images))
    
    def add_pinned_lattice(self, corner, rows: int, cols: int, offset: bool=False):
        corner = np.array(corner)
        new_vortices = self._generate_lattice_pos(corner, rows, cols, offset)
        new_images = self._get_images(new_vortices)
        self.pinned_vortices = np.concatenate((self.pinned_vortices, new_vortices))
        self.pinned_images = np.concatenate((self.pinned_images, new_images))
        
    def add_pinned_vortex(self, x_pos: float, y_pos: float):
        """Add a pinned vortex at the given x and y position"""
        position = np.array([[x_pos, y_pos]])
        self.pinned_vortices = np.append(self.pinned_vortices, [[x_pos, y_pos]], axis=0)
        images = self._get_images(position)
        self.pinned_vortices = np.concatenate((self.pinned_vortices, images))
        
    def run_sim(self, total_time: float, dt: float):
        sim_result = super().run_sim(total_time, dt)
        return ChannelSimResult.from_SimResult(sim_result, self.pinned_vortices)
        
        
class ChannelSimAnimator(SimAnimator):
    _result: ChannelSimResult
    
    def animate(self, result: ChannelSimResult, filename, anim_freq=1): # type: ignore[override]
        return super().animate(result, filename, anim_freq=anim_freq)
    
    def _anim_init(self, num_vortices):
        fig, ax = super()._anim_init(num_vortices)
        
        for vortex in self._result.pinned_vortices:
            ax.plot(vortex[0], vortex[1], 'x', c='k')
            
        return fig, ax

def plain_channel():
    sim = ChannelSimulation.create_channel(1, PINNED_DEPTH, CHANNEL_LENGTH, REPEATS)
    
    result = sim.run_sim(0.5, 0.0001)
    
    animator = ChannelSimAnimator()
    animator.animate(result, 'channel.gif', 10)
    
def current_channel():
    width = 1
    force = 0.1
    
    result = get_channel_result(width, force)
    
    animator = ChannelSimAnimator()
    animator.animate(result, f'current_channel_{width}w.gif', 10)
    
def get_filename(width, force):
    return f'w={width}, f={round(force, 4)}'
    
def get_channel_result(width: int, force: float, t_max: float=T_MAX, num_steps=T_STEPS,
                       pinned_width: int=PINNED_DEPTH, length: int=CHANNEL_LENGTH, repeats: int=REPEATS):
    dt = t_max/num_steps
    sim = ChannelSimulation.create_channel(width, pinned_width, length, repeats)
    
    result = ChannelSimResult.load(get_filename(width, force))
    if result is not None:
        if (result.dt == dt and result.num_t-1 == int(num_steps)
            and np.all(result.size_ary == sim.size_ary)):
            return result
    sim.current_force = np.array((force, 0))
    
    sim_result = sim.run_sim(t_max, dt)
    sim_result.save(get_filename(width, force))
    
    return sim_result

def get_average_vels(forces: Iterable[float], width, include_saved_results=False, **kwargs):
    vels = []
    output_forces = []
    if include_saved_results:
        output_forces, extra_results = get_saved_results(width, **kwargs)
        vels = [result.get_average_velocity() for result in extra_results]
    
    for force in forces:
        if force in output_forces:
            continue
        result = get_channel_result(width, force, **kwargs)
        output_forces.append(force)
        vels.append(result.get_average_velocity())
        # print(round(force, 4), vels[-1])
        
    return np.array(vels), output_forces

def plot_vels(force_list: Iterable[float], width, include_saved_results=False, **kwargs):
    velocities, force_list = get_average_vels(force_list, width, include_saved_results, **kwargs)
    
    # Sort so that the force list is in order
    force_list, velocities = zip(*sorted(zip(force_list, velocities)))
    velocities = np.array(velocities)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(force_list, velocities[:, 0], '.-', label='x')
    ax.plot(force_list, velocities[:, 1], '.-', label='y')
    
    ax.set_ylabel('Average velocity')
    ax.set_xlabel('Force from current')
    ax.legend()
    
    plt.show()
    
def get_saved_results(width: int, t_max: float=T_MAX, num_steps=T_STEPS, pinned_width: int=PINNED_DEPTH,
                      length: int=CHANNEL_LENGTH, repeats: int=REPEATS):
    file_start = f'w={width}, f='
    start_len = len(file_start)
    
    dt = t_max/num_steps
    # Create blank sim to get correct size
    sim = ChannelSimulation.create_channel(width, pinned_width, length, repeats)
    
    force_list: List[float] = []
    result_list: List[ChannelSimResult] = []
    
    for filename in os.listdir(SAVE_LOCATION.format(cls=ChannelSimResult)):
        if filename[:start_len] != file_start:
            continue
        force = float(filename[start_len:])
        
        result = ChannelSimResult.load(filename)
        if result is not None:
            if (result.dt == dt and result.num_t-1 == int(num_steps)
                and np.all(result.size_ary == sim.size_ary)):
                force_list.append(force)
                result_list.append(result)
                
    return force_list, result_list
    
if __name__ == '__main__':
    # plain_channel()
    # current_channel()
    # plot_vels(np.linspace(0, 0.2, 10, endpoint=False).tolist() + np.linspace(0.2, 0.5, 13).tolist(), 3)
    plot_vels(np.linspace(0.045, 0.055, 201), 1, True)
