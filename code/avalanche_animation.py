import os
from typing import Union

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.axes import Axes

import avalanche_sim as sims
from avalanche_analysis_classes import AvalancheResult
from simulation import BAR_FORMAT, HALF_ROOT_3


class AvalancheAnimator:
    def animate(self, result: AvalancheResult, filename, anim_freq: int = 1,
                event_range: Union[int, slice] = slice(None)):
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
        
    def _anim_init(self, max_vortices):
        fig = plt.figure(figsize=(10, 10*self._result.y_size/self._result.x_size))
        ax: Axes = fig.add_subplot(1, 1, 1)
        fig.tight_layout()
        ax.set_xlim([0, self._result.x_size])
        ax.set_ylim([0, self._result.y_size])
        
        self._dots = [ax.plot([], [], 'o', c='r')[0] for i in range(max_vortices)]
        
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
    def _anim_init(self, max_vortices):
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
