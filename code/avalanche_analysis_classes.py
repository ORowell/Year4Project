from typing import List, Tuple
from simulation import PickleClass, BAR_FORMAT

import os
from operator import itemgetter
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tqdm

@dataclass
class AvalancheResult(PickleClass):
    """Data class to store the results of a simulation"""
    values: List[List[np.ndarray]]
    removed_vortices: List[List[int]]
    pinning_sites: np.ndarray
    dt: float
    x_size: float
    y_size: float
    repeats: int
    random_gen: np.random.Generator
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
        """
        """
        This will remove the result data for just before a vortex is removed
        after going off the end so should not be used for analysis of vortex
        movement.
        """
        output: List[np.ndarray] = []
        for vortex_add_lst in self.values:
            for ary in vortex_add_lst:
                # # Don't include the in between data (just before a vortex is deleted)
                # ary = ary[:-1]
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
                               self.x_size, self.y_size, self.repeats, self.random_gen,
                               self.force_cutoff, self.movement_cutoff, self.movement_cutoff_time,
                               self.pinning_size, self.pinning_strength)
    
    def get_event_sizes(self, rel_cutoff: float = 2, time_start: int = 0, x_min: float = 0):
        events: List[int] = []
        is_events = self.get_events(rel_cutoff, time_start, x_min)
        for del_vortices, is_event in is_events:
            # Count all vortices that left the system as being part of the event
            num_event = len(del_vortices)
            num_event += np.count_nonzero(is_event)
            
            events.append(num_event)
        return events
    
    def get_events(self, rel_cutoff: float = 2, time_start: int = 0, x_min: float = 0):
        is_events: List[Tuple[List[int], np.ndarray]] = []
        # Iterate through each added vortex
        for data, del_vortices in zip(self.values[time_start:], self.removed_vortices[time_start:]):
            start_pos = data[0][0, ...]
            end_pos = data[-1][-1, ...]
            
            # Ignore all removed vortices so others can be compared
            for removed_vortex in del_vortices:
                start_pos = np.delete(start_pos, removed_vortex, axis=0)
            
            displacement = end_pos - start_pos
            displacement = np.mod(displacement + self.size_ary/2, self.size_ary) - self.size_ary/2
            distance_moved = np.linalg.norm(displacement, axis=1)
            is_event = np.logical_and(distance_moved > rel_cutoff*self.pinning_size, end_pos[:, 0] >= x_min)
            
            is_events.append((del_vortices, is_event))
        return is_events
    
    def get_event_paths(self, rel_cutoff: float = 2, time_start: int = 0, x_min: float = 0):
        events_paths: List[List[np.ndarray]] = []
        is_events = self.get_events(rel_cutoff, time_start, x_min)
        for (del_vortices, is_event), data in zip(is_events, self.values[time_start:]):
            event_paths: List[np.ndarray] = []
            data = data.copy()
            for removed_vortex in del_vortices:
                path = np.empty(shape=(0, 2))
                for j, data_ary in enumerate(data[:-1]):
                    path = np.append(path, data_ary[:, removed_vortex, :], axis=0)
                    # Remove vortex from the array
                    data[j] = np.delete(data_ary, removed_vortex, axis=1)
                    # Check if this is when vortex was removed
                    if data[j].shape[1] == data[j+1].shape[1]:
                        break
                event_paths.append(path)
            event_data = np.empty(shape=(0, np.count_nonzero(is_event), 2))
            for data_ary in data:
                event_data = np.append(event_data, data_ary[:, is_event, :], axis=0)
            event_paths.extend(np.swapaxes(event_data, 0, 1))
            events_paths.append(event_paths)
            
        return events_paths
            
class AvalancheAnimator:    
    def animate(self, result: AvalancheResult, filename, anim_freq: int = 1):
        self._result = result
        
        n_steps = self._result.flattened_num_t//anim_freq
        self._anim_freq = anim_freq
        self._p_bar = tqdm.tqdm(total=n_steps+1, desc='Animating ', unit='fr', bar_format=BAR_FORMAT)
        
        fig, _ = self._anim_init(self._result.max_vortices)
        animator = anim.FuncAnimation(fig, self._anim_update, n_steps, blit=True)
        
        directory = os.path.join('results', 'gifs')
        if not os.path.exists(directory):
            os.makedirs(directory)
        animator.save(os.path.join(directory, filename), fps=30)
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
        
    def _anim_update(self, frame_num: int):
        self._p_bar.update(1)
        values = self._result.flatten[frame_num*self._anim_freq]
        num_vortices = values.shape[0]
        
        compare_frame = frame_num*self._anim_freq - self._result.movement_cutoff_time
        stationary = np.full(num_vortices, True)
        if compare_frame >= 0:
            last_values = self._result.flatten[compare_frame]
            # Only change colours if a vortex wasn't just added/deleted
            if len(last_values) == num_vortices:
                diff = values - last_values
                diff = np.mod(diff + self._result.size_ary/2, self._result.size_ary) - self._result.size_ary/2
                distance = np.linalg.norm(diff, axis=1)
                stationary = distance < self._result.movement_cutoff*self._result.movement_cutoff_time
        for i, dot in enumerate(self._dots):
            if i < num_vortices:
                dot.set_data(values[i])
                dot.set_color('r' if stationary[i] else 'b')
            else:
                dot.set_data([], [])
            
        return self._dots
