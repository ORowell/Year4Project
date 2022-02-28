from avalanche_analysis_classes import AvalancheResult, BasicAvalancheResult
from short_scripts import animate_file, animate_folder

import os
from typing import List, Optional, Tuple, Type, Union
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from scipy.optimize import fsolve

def phase_plot(events_lst: List[int], title=None, exclude_zero: bool = False,
               show: bool = True):
    largest_event = max(events_lst)
    event_freq = [0]*(largest_event+1)
    for event_size in events_lst:
        event_freq[event_size] += 1
    if exclude_zero:
        event_freq[0] = 0
    
    fig = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.bar(range(largest_event+1), event_freq)
    ax.set_xlim(0, 60)
    ax.set_ylim(0)
    
    if title is not None:
        plt.title(title)
    
    if show:
        plt.show(block=False)
    
    return fig, ax

def gen_phase_plot(filename: str, exclude_zero: bool = False, save_dir: Optional[str] = None,
                   time_start: int = 0, show: bool = True, s_max: Optional[int] = None,
                   result_type: Union[Type[AvalancheResult], Type[BasicAvalancheResult]] = AvalancheResult):
    result = result_type.load(os.path.join('New_pins', filename))
    _, ax = phase_plot(result.get_event_sizes(time_start), filename, exclude_zero, False)
    if s_max is not None:
        add_power_law(result, ax, s_max)
        plt.legend()
    if show:
        plt.show(block=False)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir,  f'{filename}_powerlaw.jpg'))
    
def gen_path_plots(save_dir: str, filename: str, inc_pins: bool = True, time_start: int = 0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result = AvalancheResult.load(os.path.join('New_pins', filename))
    paths = result.get_event_paths(time_start)
    sizes = result.get_event_sizes(time_start)
    for i, path_lst in enumerate(paths):
        fig = plt.figure(figsize=(10, 10*result.y_size/result.x_size))
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_xlim([0, result.x_size])
        ax.set_ylim([0, result.y_size])
        
        if inc_pins:
            for pinned_vortex in result.pinning_sites:
                ax.add_artist(plt.Circle(pinned_vortex, result.pinning_size, color='grey', alpha=0.3))
                # Double draw vortices that go over the edge
                vortex_y = pinned_vortex[1]
                if vortex_y < result.pinning_size:
                    ax.add_artist(plt.Circle([pinned_vortex[0], vortex_y+result.y_size], result.pinning_size, color='grey', alpha=0.3))
                elif vortex_y > result.y_size - result.pinning_size:
                    ax.add_artist(plt.Circle([pinned_vortex[0], vortex_y-result.y_size], result.pinning_size, color='grey', alpha=0.3))
        
        for path in path_lst:
            path_y = path[:, 1]
            y_diff = path_y[1:] - path_y[:-1]
            # Work out when (if at all) the particle wraps
            cuts, = np.nonzero(np.abs(y_diff) > result.y_size/2)
            cuts += 1
            if cuts.size == 0:
                ax.plot(path[:, 0], path_y)
            else:
                p = ax.plot(path[:cuts[0], 0], path_y[:cuts[0]])
                colour = p[0].get_color()
                for j in range(cuts.size-1):
                    if cuts[j+1] == cuts[j] + 1:
                        continue
                    ax.plot(path[cuts[j]:cuts[j+1], 0], path_y[cuts[j]:cuts[j+1]], color=colour)
                ax.plot(path[cuts[-1]:, 0], path_y[cuts[-1]:], color=colour)
        plt.savefig(os.path.join(save_dir,  f'vortex_add{i+time_start}-size-{sizes[i]}.jpg'))
        plt.close(fig)
        
def gen_density_plot(save_dir: str, filename: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result = AvalancheResult.load(os.path.join('Density_sweep_avg', filename))
    x_pos_lst = result.get_settled_x()
    for i, x_pos in enumerate(x_pos_lst):
        fig = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        
        hist, bins = np.histogram(x_pos, int(result.x_size*2), (0, result.x_size))
        bin_centres = (bins[1:] + bins[:-1])/2
        
        ax.plot(bin_centres, hist)
        ax.set_xlim(0, result.x_size)
        ax.set_ylim(0)
        
        plt.savefig(os.path.join(save_dir,  f'vortex_add{i}.jpg'))
        plt.close(fig)
        
def alpha_solve(alpha: float, s_max: int, suff_stat: float = 0):
    s_vals = np.arange(1, s_max+1)
    powered_vals = s_vals**(-alpha)
    h = np.sum(powered_vals)
    h_prime = -alpha*np.sum(powered_vals/s_vals)
    
    return h_prime/h + suff_stat

def zeta_ish(alpha: float, s_max: int) -> float:
    s_vals = np.arange(1, s_max+1)
    powered_vals = s_vals**(-alpha)
    h = np.sum(powered_vals)
    
    return h
        
def power_law_fit(event_sizes: List[int], s_max: int, init_guess: float = 2.) -> Tuple[float, float]:
    sufficient_stat = 0.
    total_events = 0
    for size in event_sizes:
        if size == 0:
            continue
        sufficient_stat += 1/size
        total_events += 1
    sufficient_stat /= total_events
    
    alpha = fsolve(alpha_solve, init_guess, (s_max, sufficient_stat))[0]
    norm_factor = total_events / zeta_ish(alpha, s_max)
    return alpha, norm_factor

def add_power_law(result: Union[AvalancheResult, BasicAvalancheResult], ax: Axes,
                  s_max: int):
    alpha, norm_factor = power_law_fit(result.get_event_sizes(10), s_max)
    print(f'{-alpha = }')
    x_min, x_max = ax.get_xlim()
    x_vals = np.linspace(x_min, x_max)
    power_vals = x_vals**(-alpha) * norm_factor
    
    ax.plot(x_vals, power_vals, 'r', label=f'$s^{{-{alpha:.3f}}}$')
    
if __name__ == '__main__':
    # plots = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0]
    plots = [4.0, 4.5, 5.0, 5.5, 6.0]
    for d in plots:
        print(d)
        gen_phase_plot(f'new_pins_continued_{d:.1f}', True, os.path.join('results', 'Figures', 'Phase_plots'), 10, False, 50)
    # gen_phase_plot('density_4.5_spread', True, os.path.join('results', 'Figures', 'Phase_plots'), 100)
    plt.show(block=False)
    input('Press enter to exit')
    
    # animate_file('new_pins_continued_5.0', os.path.join('results', 'Simulation_results', 'AvalancheResult', 'New_pins'))#, event_range=193, output_ext='_event93')
    # animate_folder(os.path.join('results', 'Simulation_results', 'AvalancheResult', 'Density_sweep'))
    # gen_path_plots(os.path.join('results', 'Figures', 'Event_paths', 'NewPins5.0_cont_events'), 'new_pins_continued_5.0', time_start=10)
    # gen_density_plot(os.path.join('results', 'Figures', 'Density_gradients', 'Density6.0_gradient'), 'density_sweep_6.0')
