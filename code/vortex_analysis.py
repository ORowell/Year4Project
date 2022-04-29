import copy
import os
import re
from typing import List, Literal, Optional, Sequence, Tuple, Type, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from scipy.optimize import fsolve
from scipy.stats import chisquare

from avalanche_analysis_classes import AvalancheResult, BasicAvalancheResult

def truncate_colormap(cmap: Colormap, minval: float = 0.0,
                      maxval: float = 1.0, n: int = -1):
    if n == -1:
        n = cmap.N
    new_cmap = LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

# Edit a np colourmap to show NaN values as white
_cm = copy.copy(cm.get_cmap('gnuplot2'))
_cm.set_bad('w')
MY_CM = truncate_colormap(_cm, 0.15, 0.9)

def phase_plot(events_lst: List[int], title=None, exclude_zero: bool = False,
               show: bool = True, log: bool = False, s_range: Optional[Tuple[int, int]] = None):
    largest_event = max(events_lst)
    event_freq = [0]*(largest_event+1)
    for event_size in events_lst:
        event_freq[event_size] += 1
    if exclude_zero:
        event_freq[0] = 0
    
    fig = plt.figure(figsize=(4, 3))
    ax: Axes = fig.add_subplot(1, 1, 1)
    # fig.tight_layout()
    if log:
        if s_range is not None and s_range[0] < largest_event:
            s_min, s_max = s_range
            ax.plot(range(s_min), event_freq[:s_min], 'x', color='grey')
            ax.plot(range(s_min, s_max+1), event_freq[s_min:s_max+1], 'x')
            ax.plot(range(s_max+1, largest_event+1), event_freq[s_max+1:], 'x', color='grey')
        else:
            ax.plot(range(largest_event+1), event_freq, 'x')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 60)
        ax.set_ylim(0.1)
    else:
        if s_range is not None and s_range[0] < largest_event:
            s_min, s_max = s_range
            ax.bar(range(s_min), event_freq[:s_min], color='grey')
            ax.bar(range(s_min, s_max+1), event_freq[s_min:s_max+1])
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
                   s_range: Optional[Tuple[int, int]] = None,
                   log: Literal['linear', 'log', 'both'] = 'linear',
                   result_type: Union[Type[AvalancheResult],
                                      Type[BasicAvalancheResult]] = AvalancheResult,
                   exclude_zero: bool = True):
    result = result_type.load(os.path.join('New_pins', filename))
    sizes = result.get_event_sizes(time_start)
    del result # Delete for memory management
    if title is None:
        title = filename
    if log == 'linear' or log == 'both':
        gen_phase_plot_from_sizes(sizes, filename, title,
                                save_dir, show, s_range, False, exclude_zero)
    if log == 'log' or log == 'both':
        gen_phase_plot_from_sizes(sizes, filename, title,
                                save_dir, show, s_range, True, exclude_zero)

def gen_phase_plot_from_sizes(sizes: List[int], filename: str, title: Optional[str] = None,
                              save_dir: Optional[str] = None, show: bool = True,
                              s_range: Optional[Tuple[int, int]] = None, log: bool = False,
                              exclude_zero: bool = True, quiet: bool = False):
    # sizes = rand_power_law_vals(0.7, s_max, 300)
    if title is None:
        title = filename
    fig, ax = phase_plot(sizes, title, exclude_zero, False, log, s_range)
    if s_range is not None:
        add_power_law(sizes, ax, *s_range, quiet)
        plt.legend()
    fig.tight_layout()
    if show:
        plt.show(block=False)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        end = ''
        if s_range is not None:
            end += 'powerlaw'
        if log:
            end += 'log'
        if end:
            end += '_'
        plt.savefig(os.path.join(save_dir, f'{end}{filename}.png'))
        if not show:
            plt.close(fig)
        
def gen_time_plot_from_sizes(sizes: List[int], filename: str, title: Optional[str] = None,
                             save_dir: Optional[str] = None, show: bool = True):
    if title is None:
        title = filename
        
    fig = plt.figure()
    ax: Axes = fig.add_subplot(1, 1, 1)
    
    ax.plot(range(len(sizes)), sizes)
    
    ax.set_xlim(0, len(sizes))
    ax.set_ylim(0)
    ax.set_xlabel('Event number')
    ax.set_ylabel('Event size')
    # plt.title(title)
        
    if show:
        plt.show(block=False)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{filename}_time.png'))
        
def gen_time_plot(filename: str, title: Optional[str] = None, save_dir: Optional[str] = None,
                  time_start: int = 0, show: bool = True,
                  result_type: Union[Type[AvalancheResult],
                                     Type[BasicAvalancheResult]] = AvalancheResult):
    result = result_type.load(os.path.join('New_pins', filename))
    sizes = result.get_event_sizes(time_start)
    del result # Delete for memory management
    if title is None:
        title = filename
    gen_time_plot_from_sizes(sizes, filename, title, save_dir, show)
    
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
        plt.savefig(os.path.join(save_dir, f'{filename}_pins.png'))
    
def gen_path_plots(save_dir: str, filename: str, inc_pins: bool = True, 
                   inc_stat_vortices: bool = False, time_start: int = 0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result = AvalancheResult.load(os.path.join('New_pins', filename))
    paths = result.get_event_paths(time_start)
    sizes = result.get_event_sizes(time_start)
    if inc_stat_vortices:
        non_events = [~is_event for _, is_event in result.get_events(time_start)]
    for i, path_lst in enumerate(paths):
        fig = plt.figure(figsize=(10, 10*result.y_size/result.x_size))
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_xlim([0, result.x_size])
        ax.set_ylim([0, result.y_size])
        
        if inc_pins:
            add_pins_to_plot(ax, result)
            
        if inc_stat_vortices:
            stat_vortices = result.values[i+time_start][-1][-1, non_events[i], :]
            for x, y in stat_vortices:
                ax.plot(x, y, 'o', c='r')
        
        for path in path_lst:
            path_cuts = cut_path(path, result.y_size)
            p = ax.plot(*path_cuts[0])
            colour = p[0].get_color()
            for path_sec in path_cuts[1:]:
                ax.plot(*path_sec, color=colour)
        plt.savefig(os.path.join(save_dir,  f'vortex_add{i+time_start}-size-{sizes[i]}.png'))
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
    result = AvalancheResult.load(filename)
    x_pos_lst = result.get_settled_x()
    for i, x_pos in enumerate(x_pos_lst):
        fig = plt.figure()
        ax: Axes = fig.add_subplot(1, 1, 1)
        
        hist, bins = np.histogram(x_pos, int(result.x_size*2), (0, result.x_size))
        bin_centres = (bins[1:] + bins[:-1])/2
        
        ax.plot(bin_centres, hist)
        ax.set_xlim(0, result.x_size)
        ax.set_ylim(0)
        
        plt.savefig(os.path.join(save_dir,  f'vortex_add{i}.png'))
        plt.close(fig)
        
def alpha_solve(alpha: float, s_min: int, s_max: int, suff_stat: float = 0):
    s_vals = np.arange(s_min, s_max+1)
    powered_vals = s_vals**(-alpha)
    h = np.sum(powered_vals)
    h_prime = -np.sum(powered_vals*np.log(s_vals))
    
    return h_prime/h + suff_stat

def zeta_ish(alpha: float, s_min: int, s_max: int) -> float:
    s_vals = np.arange(s_min, s_max+1)
    powered_vals = s_vals**(-alpha)
    h = np.sum(powered_vals)
    
    return h
    
def power_law_fit(event_sizes: List[int], s_min: int, s_max: int, init_guess: float = 0.5,
                  min_bin_size: int = 15, quiet: bool = False) -> Tuple[float, float, float]:
    sufficient_stat = 0.
    total_events = 0
    for size in event_sizes:
        if size < s_min or size > s_max:
            continue
        sufficient_stat += np.log(size)
        total_events += 1
    if total_events == 0:
        return (np.nan, np.nan, 0.)
    sufficient_stat /= total_events
    
    alpha = fsolve(alpha_solve, init_guess, (s_min, s_max, sufficient_stat))[0]
    norm_factor = total_events / zeta_ish(alpha, s_min, s_max)
    
    p_value = fitness_test(event_sizes, s_min, s_max, alpha, norm_factor, min_bin_size, quiet)
    
    return alpha, norm_factor, p_value

def add_power_law(event_sizes: List[int], ax: Axes, s_min: int, s_max: int, quiet: bool = False):
    alpha, norm_factor, p_value = power_law_fit(event_sizes, s_min, s_max, quiet=quiet)
    if not quiet:
        print(f'{-alpha = }')
    x_vals = np.linspace(s_min, s_max)
    power_vals = x_vals**(-alpha) * norm_factor
    
    ax.plot(x_vals, power_vals, 'r',
            label=f'$s^{{-{alpha:.3f}}}$, $s_{{min}} = {s_min}$, $s_{{max}} = {s_max}$'
            + f'\nFit p-value of {p_value:.3g}')
    
def fitness_test(event_sizes: List[int], s_min: int, s_max: int, alpha: float,
                 norm_factor: float, min_bin_size: int, quiet: bool = False):
    size_range = range(s_min, s_max+1)
    expected_vals = [size**(-alpha) * norm_factor for size in size_range]
    bin_edges: List[int] = [s_min]
    expected_hist = []
    current_size = 0.
    for i, val in enumerate(expected_vals):
        current_size += val
        if current_size >= min_bin_size - 1e-5:
            bin_edges.append(i+1+s_min)
            expected_hist.append(current_size)
            current_size = 0
    if len(bin_edges) == 1:
        bin_edges.append(s_max)
        expected_hist.append(current_size)
    else:
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
        *_, p_value = power_law_fit(event_sizes, 1, s_max, quiet=True)
        if p_value > max_p_val:
            max_p_val = p_value
            best_s_max = s_max
    return best_s_max

def exponent_plot(event_sizes: List[int], s_mins: Sequence[int],
                  s_maxes: Sequence[int], colourmap: Union[Colormap, str],
                  save_info: Optional[Tuple[str, str]] = None,
                  contour_spacing: Optional[float] = None,
                  p_val_lim: Optional[float] = None, show: bool = False):
    end = f'_plim{p_val_lim}' if p_val_lim is not None else ''
    max_p_val = 0
    max_p_pos: Optional[Tuple[int, int]] = None
    largest_good_range: Optional[Tuple[int, int]] = None
    largest_range = 0
    event_sizes = np.array(event_sizes)
    
    exponent_grid = np.full((len(s_mins), len(s_maxes)), np.nan)
    filtered_exponent_grid = np.full((len(s_mins), len(s_maxes)), np.nan)
    
    for i, s_min in enumerate(s_mins):
        for j, s_max in enumerate(s_maxes):
            if s_min >= s_max - 1:
                continue
            alpha, _, p_val = power_law_fit(event_sizes, s_min, s_max, quiet=True)
            if p_val > max_p_val:
                max_p_pos = (s_min, s_max)
                max_p_val = p_val
            exponent_grid[i, j] = alpha
            if p_val_lim is None or p_val >= p_val_lim:
                filtered_exponent_grid[i, j] = alpha
                num_data = len(event_sizes[(event_sizes >= s_min) & (event_sizes <= s_max)])
                if num_data > largest_range:
                    largest_range = num_data
                    largest_good_range = (s_min, s_max)
    
    fig, ax = plt.subplots()
    im = ax.imshow(filtered_exponent_grid, cmap=colourmap, interpolation='nearest', origin='lower',
                   vmin=0, vmax=5, extent=(min(s_maxes)-0.5, max(s_maxes)+0.5,
                                           min(s_mins)-0.5, max(s_mins)+0.5))
    
    if contour_spacing is not None:
        x_grid, y_grid = np.meshgrid(s_mins, s_maxes)
        contours_vals = np.arange(0, 5, contour_spacing)
        ax.contour(x_grid, y_grid, exponent_grid, contours_vals, alpha=0.7, colors='w')
        end += '_contour'
    
    if max_p_pos is not None:
        ax.plot(max_p_pos[1], max_p_pos[0], 'x', color='grey', label='Maximum p-value')
    if largest_good_range is not None:
        ax.plot(largest_good_range[1], largest_good_range[0], '.', color='grey',
                label='Largest number of data\npoints with acceptable p-value')

    fig.colorbar(im)
    ax.legend()
    ax.set_xlim(min(s_maxes)-0.5, max(s_maxes)+0.5)
    ax.set_ylim(min(s_mins)-0.5, max(s_mins)+0.5)
    ax.set_xlabel('$s_{{max}}$')
    ax.set_ylabel('$s_{{min}}$')
    if show:
        plt.show()
    if save_info is not None:
        save_dir, filename = save_info
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'exponent_{filename}{end}.png'))
        
    return largest_good_range

def fit_folder(directory: str):
    save_dir = os.path.join('results', 'Figures', 'Phase_plots', 'System_range5.5', 'First runs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for filename in os.listdir(directory):
        if 'cont' in filename or 'noise' in filename:
            continue
        seed, = re.match(r'.*_(\d+)', filename).groups()
        result = BasicAvalancheResult.load(filename)
        # density, = re.match(r'.*_(\d\.\d)', filename).groups()
        # result = AvalancheResult.load(filename, directory)
        sizes = result.get_event_sizes(10)
        
        # Plot pins as well
        fig = plt.figure(figsize=(10, 10*result.y_size/result.x_size))
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.set_xlim([0, result.x_size])
        ax.set_ylim([0, result.y_size])
        add_pins_to_plot(ax, result)
        plt.savefig(os.path.join(save_dir, f'{filename}_pins.png'))
        plt.close(fig)
        
        del result
        max_event = max(sizes)
        plim = 0.2
        s_range = exponent_plot(sizes, range(1, max_event), range(2, max_event+1), MY_CM,
                                (save_dir, filename), 0.2, plim)
        # s_max = naive_fit_smax(sizes)
        if s_range is not None:
            s_min, s_max = s_range
            title = f'Power law fit for seed {seed}'
            # title = f'Power law fit for pin density {density}'
            gen_phase_plot_from_sizes(sizes, f'{filename}_plim{plim}', title,
                                      save_dir, False, [s_min, s_max], quiet=True)
            gen_phase_plot_from_sizes(sizes, f'{filename}_plim{plim}', title,
                                      save_dir, False, [s_min, s_max], True, quiet=True)
        plt.close('all')

def rand_power_law_vals(alpha, s_min, s_max, num_vals: int,
                        generator: Optional[np.random.Generator] = None):
    poss_s = np.arange(s_min, s_max+1)
    norm_factor = zeta_ish(alpha, s_min, s_max)
    
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
    #     gen_phase_plot(f'new_pins_continued_{d:.1f}', f'Event sizes for density {d:.1f}',
    #                    os.path.join('results', 'Figures', 'Phase_plots'), 10, False, log='both')
        # gen_phase_plot(f'new_pins_continued_{d:.1f}', f'Event sizes for density {d:.1f}',
        #                os.path.join('results', 'Figures', 'Phase_plots'), 10, False, log=True)
    # plt.show(block=False)
    
    # s_maxes = list(range(10, 21, 2)) + list(range(21, 30))+ list(range(30, 51, 5))
    # name = 'new_pins_continued_5.5'
    # result = AvalancheResult.load(os.path.join('New_pins', name))
    # sizes = result.get_event_sizes(10)
    # del result
    # max_event = max(sizes)
    # exp_plot_args = (sizes, range(1, max_event), range(2, max_event+1), MY_CM,
    #                  (os.path.join('results', 'Figures', 'Phase_plots', 'Density5.5'), 'density55'))
    # exponent_plot(*exp_plot_args)
    # exponent_plot(*exp_plot_args, 0.2)
    # exponent_plot(*exp_plot_args, p_val_lim=0.05)
    # exponent_plot(*exp_plot_args, 0.2, p_val_lim=0.05)
    # exponent_plot(*exp_plot_args, p_val_lim=0.0)
    # exponent_plot(*exp_plot_args, 0.2, p_val_lim=0.0)
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
    # fit_folder(os.path.join('results', 'Simulation_results', 'AvalancheResult', 'New_pins'))
    
    # pin_plot('new_pins_continued_5.5', 'New_pins', os.path.join('results', 'Figures'))
    # pin_plot('continued_5.5', 'Density_sweep', os.path.join('results', 'Figures'))
    
    # input('Press enter to exit')
    # gen_path_plots(os.path.join('results', 'Figures', 'Event_paths', 'NewPins5.0_cont_events_all_vorts'),
    #                'new_pins_continued_5.0', inc_stat_vortices=True, time_start=10)
    # gen_time_plot('new_pins_continued_5.5', None, os.path.join('results', 'Figures'), 10)
    # gen_density_plot(os.path.join('results', 'Figures', 'Density_gradients', 'Big5.5_init'),
    #                  'big5.5_init')
    
    pass
