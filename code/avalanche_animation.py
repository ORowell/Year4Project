import os
from typing import Optional, Literal, List, Union

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.axes import Axes

import avalanche_sim as sims
from avalanche_analysis_classes import AvalancheResult
from simulation import BAR_FORMAT, HALF_ROOT_3
from vortex_analysis import cut_path

class AvalancheAnimator:
    def animate(self, result: AvalancheResult, filename, anim_freq: int = 1,
                event_range: Union[int, slice] = slice(None), **kwargs):
        self._result = result
        self.flat_result = self._result.flatten(event_range)
        self._y_size = self._result.y_size
        
        n_steps = len(self.flat_result)//anim_freq
        self._anim_freq = anim_freq
        self._p_bar = tqdm.tqdm(total=n_steps+1, desc='Animating ', unit='fr', bar_format=BAR_FORMAT)
        
        fig, _ = self._anim_init(self._result.max_vortices)
        animator = anim.FuncAnimation(fig, self._anim_update, n_steps, blit=True)
        
        directory = os.path.join('results', 'gifs')
        if not os.path.exists(directory):
            os.makedirs(directory)
        animator.save(os.path.join(directory, filename), fps=30)
        self._p_bar.close()
        
    def _anim_init(self, max_vortices: int):
        fig = plt.figure(figsize=(10, 10*self._result.y_size/self._result.x_size))
        ax: Axes = fig.add_subplot(1, 1, 1)
        fig.tight_layout()
        ax.set_xlim([0, self._result.x_size])
        ax.set_ylim([0, self._result.y_size])
        
        self._dots = [ax.plot([], [], 'o', c='r', zorder=100)[0] for i in range(max_vortices)]
        
        for vortex in self._result.pinning_sites:
            ax.add_artist(plt.Circle(vortex, self._result.pinning_size, color='grey', alpha=0.3))
            # Double draw vortices that go over the edge
            vortex_y = vortex[1]
            if vortex_y < self._result.pinning_size:
                ax.add_artist(plt.Circle([vortex[0], vortex_y+self._y_size], self._result.pinning_size, color='grey', alpha=0.3))
            elif vortex_y > self._y_size - self._result.pinning_size:
                ax.add_artist(plt.Circle([vortex[0], vortex_y-self._y_size], self._result.pinning_size, color='grey', alpha=0.3))
        
        return fig, ax
        
    def _anim_update(self, frame_num: int):
        self._p_bar.update(1)
        values = self.flat_result[frame_num*self._anim_freq]
        num_vortices = values.shape[0]
        
        compare_frame = frame_num*self._anim_freq - self._result.movement_cutoff_time
        self._stationary = np.full(num_vortices, True)
        if compare_frame >= 0:
            last_values = self.flat_result[compare_frame]
            # Only change colours if a vortex wasn't just added/deleted
            if len(last_values) == num_vortices:
                diff = values - last_values
                diff = np.mod(diff + self._result.size_ary/2, self._result.size_ary) - self._result.size_ary/2
                distance = np.linalg.norm(diff, axis=1)
                self._stationary = distance < self._result.movement_cutoff*self._result.movement_cutoff_time
        for i, dot in enumerate(self._dots):
            if i < num_vortices:
                dot.set_data(values[i])
                dot.set_color('r' if self._stationary[i] else 'b')
            else:
                dot.set_data([], [])
            
        return self._dots

class ImagesAvalancheAnimator(AvalancheAnimator):
    def _anim_init(self, max_vortices: int):
        self._image_num = self._result.repeats
        self.max_real = max_vortices
        self.total_images = 1+4*self._image_num
        fig, ax = super()._anim_init(max_vortices*(1+self.total_images))
        
        # Resize the figure and axes limits
        fig_width, fig_height = fig.get_size_inches()
        fig_width *= 2
        fig_height *= 1 + 2*self._image_num
        fig.set_size_inches(fig_width, fig_height)
        
        ax.set_xlim([-self._result.x_size, self._result.x_size])
        ax.set_ylim([-self._image_num*self._result.y_size, self._result.y_size*(1+self._image_num)])
        
        self._blank_sim = sims.StepAvalancheSim(self._result.x_size, self._result.y_size/HALF_ROOT_3,
                                          0, self._image_num, 0, 0)
        for i, dot in enumerate(self._dots):
            if i >= self.max_real:
                dot.set_alpha(0.3)
                
        return fig, ax
        
    def _anim_update(self, frame_num: int):
        super()._anim_update(frame_num)
        values = self.flat_result[frame_num*self._anim_freq]
        num_vortices = values.shape[0]
        images = self._get_images(values)
        
        for i, dot in enumerate(self._dots):
            if i < num_vortices:
                continue
            elif i % self.max_real < num_vortices:
                real_i = i % self.max_real
                image_num = i // self.max_real - 1
                dot.set_data(images[real_i*self.total_images+image_num])
                dot.set_color('r' if self._stationary[real_i] else 'b')
            else:
                dot.set_data([], [])
                
        return self._dots
    
    def _get_images(self, vortices: np.ndarray):
        self._blank_sim.vortices = vortices
        all_vortices = self._blank_sim._get_all_vortices()
        return all_vortices[len(vortices):]
    
class EventAnimator(AvalancheAnimator):
    def animate(self, result: AvalancheResult, filename, anim_freq: int = 1,
                event_range: Union[int, slice] = slice(None), pause_frames: int = 0,
                event_time_start: int = 0, **kwargs):
        self._result = result
        if isinstance(event_range, int):
            event_range = slice(event_range, event_range+1)
        self._y_size = self._result.y_size
        self._event_paths = self._result.get_event_paths(event_time_start)[event_range]
        
        data = self._result.values[event_range]
        self.flat_values: List[List[np.ndarray]] = []
        for event_vals in data:
            flat_event_vals = []
            for ary in event_vals:
                for data in ary:
                    flat_event_vals.append(data)
            self.flat_values.append(flat_event_vals)
        
        self._event_steps = list(map(len, self.flat_values))
        n_steps = sum(self._event_steps)//anim_freq + (len(self._event_steps) - 1)*pause_frames
        self._anim_freq = anim_freq
        self._pause_time = pause_frames
        self._current_event_i = 0
        self._pausing = 0
        
        self._p_bar = tqdm.tqdm(total=n_steps+1, desc='Animating ', unit='fr', bar_format=BAR_FORMAT)
        
        fig, _ = self._anim_init(self._result.max_vortices)
        animator = anim.FuncAnimation(fig, self._anim_update, n_steps, blit=True)
        
        directory = os.path.join('results', 'gifs')
        if not os.path.exists(directory):
            os.makedirs(directory)
        animator.save(os.path.join(directory, filename), fps=30)
        self._p_bar.close()
        
    def _anim_init(self, max_vortices: int, max_path_lines: float = 5):
        fig, ax = super()._anim_init(max_vortices)
        self._largest_event = max(map(len, self._event_paths))
        
        self._event_lines = [ax.plot([], [])[0] for _ in range(int(self._largest_event*max_path_lines))]
        
        return fig, ax
        
    def _anim_update(self, frame_num: int):
        self._p_bar.update(1)
        if self._pausing:
            if self._pausing == self._pause_time:
                self._pausing = 0
            else:
                self._pausing += 1
                return []
        frame_index = (self._anim_freq * (frame_num - self._pause_time*self._current_event_i) - 
                       sum(self._event_steps[:self._current_event_i]))
        if frame_index >= self._event_steps[self._current_event_i]:
            self._current_event_i += 1
            frame_index = (self._anim_freq * (frame_num - self._pause_time*self._current_event_i) - 
                           sum(self._event_steps[:self._current_event_i]))
            if self._pause_time:
                self._pausing = 1
                return []
        values = self.flat_values[self._current_event_i][frame_index]
        num_vortices = values.shape[0]
        
        # Plot event paths
        line_i = 0
        i = -1
        for i, path in enumerate(self._event_paths[self._current_event_i]):
            cut_paths = cut_path(path[:frame_index], self._result.y_size)
            self._event_lines[i].set_data(*cut_paths[0])
            colour = self._event_lines[i].get_color()
            for path_sec in cut_paths[1:]:
                if len(path_sec[0]) < self._anim_freq:
                    continue
                self._event_lines[self._largest_event+line_i].set_data(*path_sec)
                self._event_lines[self._largest_event+line_i].set_color(colour)
                line_i += 1
        # self._p_bar.write(f'{frame_index = }, {i = }, {line_i = }')
        for line in (self._event_lines[i+1:self._largest_event] +
                     self._event_lines[self._largest_event+line_i:]):
            line.set_data([], [])
        
        compare_frame = frame_index - self._result.movement_cutoff_time
        self._stationary = np.full(num_vortices, True)
        if compare_frame >= 0:
            last_values = self.flat_values[self._current_event_i][compare_frame]
            # Only change colours if a vortex wasn't just added/deleted
            if len(last_values) == num_vortices:
                diff = values - last_values
                diff = np.mod(diff + self._result.size_ary/2, self._result.size_ary) - self._result.size_ary/2
                distance = np.linalg.norm(diff, axis=1)
                self._stationary = distance < self._result.movement_cutoff*self._result.movement_cutoff_time
        for i, dot in enumerate(self._dots):
            if i < num_vortices:
                dot.set_data(values[i])
                dot.set_color('r' if self._stationary[i] else 'b')
            else:
                dot.set_data([], [])
            
        return self._dots + self._event_lines

def animate_file(filename: str, directory: str, output_ext: str = '', freq: Optional[int] = None,
                 event_range: Union[int, slice] = slice(None),
                 style: Literal['normal', 'images', 'events'] = 'normal', **animator_kwargs):
    result = AvalancheResult.load(filename, directory)
    if result is None:
        print(f'Failed to load {filename}', flush=True)
        return
    if freq is None:
        print(f'{result.movement_cutoff = }, {result.movement_cutoff_time = }')
        print(f'{result.dt = }, {result.vortices_added = }')
        freq = int(input(f'{len(result.flatten(event_range))} to animate. Enter frequency: '))
        
    # Compress result before hand for memory management
    if freq >= 100 and freq % 10 == 0:
        comp_result = result.compress(10)
        del result
        result = comp_result
        anim_freq = freq // 10
    else:
        anim_freq = freq
    
    if style == 'normal':
        animator = AvalancheAnimator()
    elif style == 'images':
        animator = ImagesAvalancheAnimator()
    elif style == 'events':
        animator = EventAnimator()
    animator.animate(result, f'{filename}{output_ext}_f{freq}.gif', anim_freq,
                     event_range, **animator_kwargs)
    
    return freq

def animate_folder(directory: str, output_ext: str = '', single_freq: bool = True):
    freq_used = None
    for filename in os.listdir(directory):
        freq_used = animate_file(filename, directory, output_ext, freq_used)
        if not single_freq:
            freq_used = None
            
if __name__ == '__main__':
    # animate_file('new_pins_continued_5.5', os.path.join('results', 'Simulation_results', 'AvalancheResult', 'New_pins'))
    animate_file('new_continued_5.5', os.path.join('results', 'Simulation_results', 'AvalancheResult', 'Density_sweep'),
                 '_events', style='events', pause_frames=15)
                #  '_event261', 5, slice(256, 262), 'events', pause_frames=100)
    # animate_file('big5.5_init', os.path.join('results', 'Simulation_results', 'AvalancheResult'))
    # animate_folder(os.path.join('results', 'Simulation_results', 'AvalancheResult', 'Density_sweep'))
    pass
