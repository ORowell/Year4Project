import os
from typing import List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import fsolve
from scipy.stats import chisquare

from avalanche_analysis_classes import AvalancheResult, BasicAvalancheResult
from short_scripts import animate_file, animate_folder


def phase_plot(events_lst: List[int], title=None, exclude_zero: bool = False,
               show: bool = True, log: bool = False):
    largest_event = max(events_lst)
    event_freq = [0]*(largest_event+1)
    for event_size in events_lst:
        event_freq[event_size] += 1
    if exclude_zero:
        event_freq[0] = 0
    
    fig = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    if log:
        ax.plot(range(largest_event+1), event_freq, 'x')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 60)
    else:
        ax.bar(range(largest_event+1), event_freq)
        ax.set_xlim(0, 60)
    ax.set_ylim(0.1)
    
    if title is not None:
        plt.title(title)
    
    if show:
        plt.show(block=False)
    
    return fig, ax

def gen_phase_plot(filename: str, exclude_zero: bool = False, save_dir: Optional[str] = None,
                   time_start: int = 0, show: bool = True, s_max: Optional[int] = None, log: bool = False,
                   result_type: Union[Type[AvalancheResult], Type[BasicAvalancheResult]] = AvalancheResult):
    result = result_type.load(os.path.join('New_pins', filename))
    sizes = result.get_event_sizes(time_start)
    del result # Delete for memory management
    # sizes = rand_power_law_vals(0.7, s_max, 300)
    _, ax = phase_plot(sizes, filename, exclude_zero, False, log)
    if s_max is not None:
        add_power_law(sizes, ax, s_max)
        plt.legend()
    if show:
        plt.show(block=False)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        end = 'log' if log else ''
        plt.savefig(os.path.join(save_dir, f'{filename}_powerlaw{end}.jpg'))
    
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
    h_prime = -np.sum(powered_vals*np.log(s_vals))
    
    return h_prime/h + suff_stat

def zeta_ish(alpha: float, s_max: int) -> float:
    s_vals = np.arange(1, s_max+1)
    powered_vals = s_vals**(-alpha)
    h = np.sum(powered_vals)
    
    return h
    
def power_law_fit(event_sizes: List[int], s_max: int, init_guess: float = 0.5) -> Tuple[float, float]:
    sufficient_stat = 0.
    total_events = 0
    for size in event_sizes:
        if size == 0:
            continue
        sufficient_stat += np.log(size)
        total_events += 1
    sufficient_stat /= total_events
    
    alpha = fsolve(alpha_solve, init_guess, (s_max, sufficient_stat))[0]
    norm_factor = total_events / zeta_ish(alpha, s_max)
    return alpha, norm_factor

def add_power_law(event_sizes: List[int], ax: Axes, s_max: int):
    alpha, norm_factor = power_law_fit(event_sizes, s_max)
    print(f'{-alpha = }')
    x_min, x_max = ax.get_xlim()
    x_vals = np.linspace(x_min, x_max)
    power_vals = x_vals**(-alpha) * norm_factor
    
    p_value = fitness_test(event_sizes, s_max, alpha, norm_factor, 15)
    
    ax.plot(x_vals, power_vals, 'r', label=f'$s^{{-{alpha:.3f}}}$\nFit p-value of {p_value:.3e}')
    
def fitness_test(event_sizes: List[int], s_max: int, alpha: float, norm_factor: float,
                 min_bin_size: float):
    size_range = range(1, s_max+1)
    expected_vals = [size**(-alpha) * norm_factor for size in size_range]
    bin_edges: List[int] = [1]
    expected_hist = []
    current_size = 0.
    for i, val in enumerate(expected_vals):
        current_size += val
        if current_size >= min_bin_size:
            bin_edges.append(i+2)
            expected_hist.append(current_size)
            current_size = 0
    bin_edges[-1] = s_max
    expected_hist[-1] += current_size
    
    expected_hist = np.array(expected_hist) #type: ignore
    observed_hist, _ = np.histogram(event_sizes, bins=bin_edges)
    print(f'Using bins {bin_edges}')
    print(f'which give expected values of {expected_hist}, sum={sum(expected_hist)}')
    print(f'compared to observed values {observed_hist}, sum={sum(observed_hist)}')
    
    p_value = chisquare(observed_hist, expected_hist, 1).pvalue
    print(p_value)
    
    return p_value

def rand_power_law_vals(alpha, s_max, num_vals: int,
                        generator: Optional[np.random.Generator] = None):
    poss_s = np.arange(1, s_max+1)
    norm_factor = zeta_ish(alpha, s_max)
    
    pdf = poss_s ** (-alpha) / norm_factor
    cdf = np.cumsum(pdf)
    
    if generator is None:
        generator = np.random.default_rng()
    vals: np.ndarray = generator.uniform(size=num_vals)
    out_vals = np.searchsorted(cdf, vals)+1
    
    return out_vals
    
if __name__ == '__main__':
    # plots = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0]
    plots = [4.0, 4.5, 5.0, 5.5, 6.0]
    for d in plots:
        print(d)
        gen_phase_plot(f'new_pins_continued_{d:.1f}', True, os.path.join('results', 'Figures', 'Phase_plots'), 10, False, 50, True)
    # gen_phase_plot('density_4.5_spread', True, os.path.join('results', 'Figures', 'Phase_plots'), 100)
    plt.show(block=False)
    input('Press enter to exit')
    
    # animate_file('new_pins_continued_5.0', os.path.join('results', 'Simulation_results', 'AvalancheResult', 'New_pins'))#, event_range=193, output_ext='_event93')
    # animate_folder(os.path.join('results', 'Simulation_results', 'AvalancheResult', 'Density_sweep'))
    # gen_path_plots(os.path.join('results', 'Figures', 'Event_paths', 'NewPins5.0_cont_events'), 'new_pins_continued_5.0', time_start=10)
    # gen_density_plot(os.path.join('results', 'Figures', 'Density_gradients', 'Density6.0_gradient'), 'density_sweep_6.0')
