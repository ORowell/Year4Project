import os
import pickle
from dataclasses import dataclass, field
from operator import itemgetter
from typing import List, Optional, Tuple, Type, TypeVar, Union

import numpy as np

SAVE_LOCATION = os.path.join('results', 'Simulation_results', '{cls.__name__}')

_T = TypeVar('_T', bound='PickleClass')
class PickleClass:
    def save(self, filename, directory: Optional[str] = None):
        if directory is None:
            directory = SAVE_LOCATION.format(cls=self.__class__)
        if not os.path.exists(directory):
            print('Making dirs at', directory, flush=True)
            os.makedirs(directory)
        file_location = os.path.join(directory, filename)
        with open(file_location, 'wb') as f:
            print('Saving file to', file_location, flush=True)
            pickle.dump(self, f)
            
    @classmethod
    def load(cls: Type[_T], filename: str, directory: Optional[str] = None,
             quiet=False) -> Optional[_T]:
        if directory is None:
            directory = SAVE_LOCATION.format(cls=cls)
        if not os.path.exists(directory):
            if not quiet:
                print(f'{directory} not found', flush=True)
            return None
        with open(os.path.join(directory, filename), 'rb') as f:
            if not quiet:
                print(f'Loading result found at {os.path.join(directory, filename)}',
                      flush=True)
            result = pickle.load(f)
            if isinstance(result, cls):
                if not quiet:
                    print('Result loaded', flush=True)
                return result
            else:
                return None

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
        flattened_values = self.flatten()
        self.flattened_num_t = len(flattened_values)
        shapes = [values.shape for values in flattened_values]
        self.max_vortices = max(shapes, key=itemgetter(0))[0]
        
    def flatten(self, event_range: Union[int, slice] = slice(None)):
        """Return the result as just a list of varying array shape.
        """
        """
        This will remove the result data for just before a vortex is removed
        after going off the end so should not be used for analysis of vortex
        movement.
        """
        output: List[np.ndarray] = []
        if isinstance(event_range, int):
            vals_to_use = [self.values[event_range]]
        else:
            vals_to_use = self.values[event_range]
        for vortex_add_lst in vals_to_use:
            for ary in vortex_add_lst:
                # # Don't include the in between data (just before a vortex is deleted)
                # ary = ary[:-1]
                for data in ary:
                    output.append(data)
            # # But do include the final one after movement has ended
            # output.append(vortex_add_lst[-1][-1])
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
                               self.force_cutoff, self.movement_cutoff, self.movement_cutoff_time//freq,
                               self.pinning_size, self.pinning_strength)
        
    def to_basic_result(self):
        events = []
        for i, (event_data, removed_vortices) in enumerate(zip(self.values, self.removed_vortices)):
            events.append(Event.from_data(event_data, removed_vortices, self.dt, i))
            
        return BasicAvalancheResult(events, self.pinning_sites, self.x_size, self.y_size,
                                    self.repeats, self.random_gen, self.force_cutoff,
                                    self.movement_cutoff, self.movement_cutoff_time,
                                    self.pinning_size, self.pinning_strength)
    
    def get_event_sizes(self, time_start: int = 0, x_min: float = 1, rel_cutoff: float = 2):
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
    
    def get_event_paths(self, time_start: int = 0, x_min: float = 1, rel_cutoff: float = 2):
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
    
    def get_settled_x(self):
        output: List[np.ndarray] = []
        for vortex_added in self.values:
            end_result_x = vortex_added[-1][-1, :, 0]
            output.append(end_result_x)
        return output
    
@dataclass
class Event:
    moved_vortices: np.ndarray
    removed_vortices: np.ndarray
    time_length: float
    number: int
    
    @classmethod
    def from_data(cls, event_values: List[np.ndarray], removed_vortex_is: List[int], dt: float, number: int):
        start_pos = event_values[0][0, ...]
        end_pos = event_values[-1][-1, ...]
        removed_vortices = np.empty(shape=(0, 2))
        
        # Separate removed vortices from the other data
        for removed_i in removed_vortex_is:
            removed_vortices = np.append(removed_vortices, start_pos[np.newaxis, removed_i, :], axis=0)
            start_pos = np.delete(start_pos, removed_i, axis=0)
        
        event_data = np.stack((start_pos, end_pos), axis=0)
        
        # Calculate the number of time steps in the event
        time_steps = sum(map(lambda ary: len(ary)-1, event_values))
        
        return cls(event_data, removed_vortices, time_steps*dt, number)
    
@dataclass
class BasicAvalancheResult(PickleClass):
    events: List[Event]
    pinning_sites: np.ndarray
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
    
    def __post_init__(self):
        self.size_ary = np.array([self.x_size, self.y_size])
        self.vortices_added = len(self.events)
    
    def get_event_sizes(self, time_start: int = 0, x_min: float = 1, rel_cutoff: float = 2):
        events: List[int] = []
        is_events = self.get_events(rel_cutoff, time_start, x_min)
        for removed_num, is_event in is_events:
            # Count all vortices that left the system as being part of the event
            num_event = removed_num
            num_event += np.count_nonzero(is_event)
            
            events.append(num_event)
        return events
    
    def get_events(self, rel_cutoff: float = 2, time_start: int = 0, x_min: float = 0):
        is_events: List[Tuple[int, np.ndarray]] = []
        # Iterate through each added vortex
        for event in self.events[time_start:]:
            start_pos = event.moved_vortices[0]
            end_pos = event.moved_vortices[-1]
            
            displacement = end_pos - start_pos
            displacement = np.mod(displacement + self.size_ary/2, self.size_ary) - self.size_ary/2
            distance_moved = np.linalg.norm(displacement, axis=1)
            is_event = np.logical_and(distance_moved > rel_cutoff*self.pinning_size, end_pos[:, 0] >= x_min)
            
            is_events.append((len(event.removed_vortices), is_event))
        return is_events
    
    def get_settled_x(self):
        output: List[np.ndarray] = []
        for event in self.events:
            end_result_x = event.moved_vortices[-1]
            output.append(end_result_x)
        return output
    