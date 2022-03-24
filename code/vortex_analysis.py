import os
import re
from typing import List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.optimize import fsolve
from scipy.stats import chisquare

from avalanche_analysis_classes import AvalancheResult, BasicAvalancheResult


def phase_plot(events_lst: List[int], title=None, exclude_zero: bool = False,
               show: bool = True, log: bool = False, s_max: Optional[int] = None):
    largest_event = max(events_lst)
    event_freq = [0]*(largest_event+1)
    for event_size in events_lst:
        event_freq[event_size] += 1
    if exclude_zero:
        event_freq[0] = 0
    
    fig = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    # fig.tight_layout()
    if log:
        if s_max is not None and s_max < largest_event:
            ax.plot(range(s_max+1), event_freq[:s_max+1], 'x')
            ax.plot(range(s_max+1, largest_event+1), event_freq[s_max+1:], 'x', color='grey')
        else:
            ax.plot(range(largest_event+1), event_freq, 'x')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 60)
        ax.set_ylim(0.1)
    else:
        if s_max is not None and s_max < largest_event:
            ax.bar(range(s_max+1), event_freq[:s_max+1])
            ax.bar(range(s_max+1, largest_event+1), event_freq[s_max+1:], color='grey')
        else:
            ax.bar(range(largest_event+1), event_freq)
        ax.set_xlim(0, 60)
        ax.set_ylim(0)
    ax.set_xlabel('Event size')
    ax.set_ylabel(f'Frequency')
    
    if title is not None:
        plt.title(title)
    
    if show:
        plt.show(block=False)
    
    return fig, ax

def gen_phase_plot(filename: str, title: Optional[str] = None, save_dir: Optional[str] = None,
                   time_start: int = 0, show: bool = True,
                   s_max: Optional[int] = None, log: bool = False,
                   result_type: Union[Type[AvalancheResult],
                                      Type[BasicAvalancheResult]] = AvalancheResult,
                   exclude_zero: bool = True):
    result = result_type.load(os.path.join('Density_sweep', filename))
    sizes = result.get_event_sizes(time_start)
    del result # Delete for memory management
    if title is None:
        title = filename
    gen_phase_plot_from_sizes(sizes, filename, title,
                              save_dir, show, s_max, log, exclude_zero)

def gen_phase_plot_from_sizes(sizes: List[int], filename: str, title: Optional[str] = None,
                              save_dir: Optional[str] = None, show: bool = True,
                              s_max: Optional[int] = None, log: bool = False,
                              exclude_zero: bool = True, quiet: bool = False):
    # sizes = rand_power_law_vals(0.7, s_max, 300)
    if title is None:
        title = filename
    fig, ax = phase_plot(sizes, title, exclude_zero, False, log, s_max)
    if s_max is not None:
        add_power_law(sizes, ax, s_max, quiet)
        plt.legend()
    if show:
        plt.show(block=False)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        end = ''
        if s_max is not None:
            end += 'powerlaw'
        if log:
            end += 'log'
        if end:
            end += '_'
        plt.savefig(os.path.join(save_dir, f'{end}{filename}.jpg'))
    
def add_pins_to_plot(ax: Axes, result: Union[AvalancheResult, BasicAvalancheResult]):
    for pinned_vortex in result.pinning_sites:
        ax.add_artist(plt.Circle(pinned_vortex, result.pinning_size, color='grey', alpha=0.3))
        # Double draw vortices that go over the edge
        vortex_y = pinned_vortex[1]
        if vortex_y < result.pinning_size:
            ax.add_artist(plt.Circle([pinned_vortex[0], vortex_y+result.y_size],
                                     result.pinning_size, color='grey', alpha=0.3))
        elif vortex_y > result.y_size - result.pinning_size:
            ax.add_artist(plt.Circle([pinned_vortex[0], vortex_y-result.y_size],
                                     result.pinning_size, color='grey', alpha=0.3))
            
def pin_plot(filename: str, load_folder: str = '', save_dir: Optional[str] = None, show: bool = True,
             result_type: Union[Type[AvalancheResult], Type[BasicAvalancheResult]] = AvalancheResult):
    result = result_type.load(os.path.join(load_folder, filename))
    fig = plt.figure(figsize=(10, 10*result.y_size/result.x_size))
    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.set_xlim([0, result.x_size])
    ax.set_ylim([0, result.y_size])
    add_pins_to_plot(ax, result)
    if show:
        plt.show(block=False)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{filename}_pins.jpg'))
    
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
            add_pins_to_plot(ax, result)
        
        for path in path_lst:
            path_cuts = cut_path(path, result.y_size)
            p = ax.plot(*path_cuts[0])
            colour = p[0].get_color()
            for path_sec in path_cuts[1:]:
                ax.plot(*path_sec, color=colour)
        plt.savefig(os.path.join(save_dir,  f'vortex_add{i+time_start}-size-{sizes[i]}.jpg'))
        plt.close(fig)
        
def cut_path(path: np.ndarray, y_size: float) -> List[Tuple[np.ndarray, np.ndarray]]:
    path_y = path[:, 1]
    y_diff = path_y[1:] - path_y[:-1]
    # Work out when (if at all) the particle wraps
    cuts, = np.nonzero(np.abs(y_diff) > y_size/2)
    cuts += 1
    if cuts.size == 0:
        return [(path[:, 0], path_y)]
    path_outs = [(path[:cuts[0], 0], path_y[:cuts[0]])]
    for j in range(cuts.size-1):
        if cuts[j+1] == cuts[j] + 1:
            continue
        path_outs.append((path[cuts[j]:cuts[j+1], 0], path_y[cuts[j]:cuts[j+1]]))
    path_outs.append((path[cuts[-1]:, 0], path_y[cuts[-1]:]))
    return path_outs
        
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
    
def power_law_fit(event_sizes: List[int], s_max: int, init_guess: float = 0.5,
                  min_bin_size: int = 15, quiet: bool = False) -> Tuple[float, float]:
    sufficient_stat = 0.
    total_events = 0
    for size in event_sizes:
        if size == 0 or size > s_max:
            continue
        sufficient_stat += np.log(size)
        total_events += 1
    sufficient_stat /= total_events
    
    alpha = fsolve(alpha_solve, init_guess, (s_max, sufficient_stat))[0]
    norm_factor = total_events / zeta_ish(alpha, s_max)
    
    p_value = fitness_test(event_sizes, s_max, alpha, norm_factor, min_bin_size, quiet)
    
    return alpha, norm_factor, p_value

def add_power_law(event_sizes: List[int], ax: Axes, s_max: int, quiet: bool = False):
    alpha, norm_factor, p_value = power_law_fit(event_sizes, s_max, quiet=quiet)
    if not quiet:
        print(f'{-alpha = }')
    x_min = max(ax.get_xlim()[0], 1.)
    x_vals = np.linspace(x_min, s_max)
    power_vals = x_vals**(-alpha) * norm_factor
    
    ax.plot(x_vals, power_vals, 'r',
            label=f'$s^{{-{alpha:.3f}}}$, $s_0 = {s_max}$\nFit p-value of {p_value:.3g}')
    
def fitness_test(event_sizes: List[int], s_max: int, alpha: float, norm_factor: float,
                 min_bin_size: int, quiet: bool = False):
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
    
    p_value = chisquare(observed_hist, expected_hist, 1).pvalue
    if not quiet:
        print(f'Using bins {bin_edges}')
        print(f'which give expected values of {expected_hist}, sum={sum(expected_hist)}')
        print(f'compared to observed values {observed_hist}, sum={sum(observed_hist)}.')
        print(f'{p_value = }')
    
    return p_value

def naive_fit_smax(event_sizes: List[int]):
    max_p_val = 0
    best_s_max = 0
    for s_max in range(10, 35):
        *_, p_value = power_law_fit(event_sizes, s_max, quiet=True)
        if p_value > max_p_val:
            max_p_val = p_value
            best_s_max = s_max
    return best_s_max

def fit_folder(directory: str):
    save_dir = os.path.join('results', 'Figures', 'Phase_plots', 'System_range5.5')
    for filename in os.listdir(directory):
        seed, = re.match(r'.*_(\d+)', filename).groups()
        result = BasicAvalancheResult.load(filename)
        sizes = result.get_event_sizes(10)
        
        # Plot pins as well
        fig = plt.figure(figsize=(10, 10*result.y_size/result.x_size))
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_xlim([0, result.x_size])
        ax.set_ylim([0, result.y_size])
        add_pins_to_plot(ax, result)
        plt.savefig(os.path.join(save_dir, f'{filename}_pins.jpg'))
        plt.close(fig)
        
        del result
        s_max = naive_fit_smax(sizes)
        gen_phase_plot_from_sizes(sizes, filename, f'Power law fit for seed {seed}',
                                  save_dir, False, s_max, quiet=True)
        gen_phase_plot_from_sizes(sizes, filename, f'Power law fit for seed {seed}',
                                  save_dir, False, s_max, True, quiet=True)
        plt.close('all')

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
    # plots = [4.0, 4.5, 5.0, 5.5, 6.0]
    # plots = [3.0, 5.5]
    # for d in plots:
    #     gen_phase_plot(f'new_continued_{d:.1f}',f'Event sizes for density {d:.1f}',
    #                    os.path.join('results', 'Figures', 'Phase_plots'), 10, False)
    #     gen_phase_plot(f'new_pins_continued_{d:.1f}', True,
    #                    os.path.join('results', 'Figures', 'Phase_plots'), 10, False, 50, True)
    # plt.show(block=False)
    
    # s_maxes = list(range(10, 21, 2)) + list(range(21, 30))+ list(range(30, 51, 5))
    # name = 'new_pins_continued_6.0'
    # result = AvalancheResult.load(os.path.join('New_pins', name))
    # sizes = result.get_event_sizes(10)
    # del result
    # for s_max in s_maxes:
    #     print(f'{s_max = }')
    #     gen_phase_plot_from_sizes(sizes, f'density60_smax_{s_max}', f'Power law fit for density 6.0',
    #                               os.path.join('results', 'Figures', 'Phase_plots'),
    #                               False, s_max, True)
    #     gen_phase_plot_from_sizes(sizes, f'density60_smax_{s_max}', f'Power law fit for density 6.0',
    #                               os.path.join('results', 'Figures', 'Phase_plots'),
    #                               False, s_max, False)
    # plt.show(block=False)
    fit_folder(os.path.join('results', 'Simulation_results', 'BasicAvalancheResult'))
    
    # pin_plot('new_pins_continued_5.5', 'New_pins', os.path.join('results', 'Figures'))
    # pin_plot('continued_5.5', 'Density_sweep', os.path.join('results', 'Figures'))
    
    # input('Press enter to exit')
    # gen_path_plots(os.path.join('results', 'Figures', 'Event_paths', 'NewPins5.5_cont_events'),
    #                'new_pins_continued_5.5', time_start=10)
    # gen_density_plot(os.path.join('results', 'Figures', 'Density_gradients', 'Density6.0_gradient'),
    #                  'density_sweep_6.0')
    pass
