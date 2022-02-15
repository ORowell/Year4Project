from avalanche_analysis_classes import AvalancheResult
from short_scripts import animate_file, animate_folder

import os
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def phase_plot(events_lst: List[int], title=None):
    largest_event = max(events_lst)
    event_freq = [0]*(largest_event+1)
    for event_size in events_lst:
        event_freq[event_size] += 1
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(range(largest_event+1), event_freq)
    
    if title is not None:
        plt.title(title)
    
    plt.show(block=False)

def gen_phase_plot(filename: str):
    result = AvalancheResult.load(os.path.join('Density_sweep_avg', filename))
    phase_plot(result.get_event_sizes(x_min=1, time_start=100), filename)
    
def gen_path_plots(save_dir: str, filename: str, inc_pins: bool = True):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result = AvalancheResult.load(os.path.join('Density_sweep_avg', filename))
    paths = result.get_event_paths(x_min=1, time_start=100)
    sizes = result.get_event_sizes(x_min=1, time_start=100)
    for i, path_lst in enumerate(paths):
        fig = plt.figure(figsize=(10, 10*result.y_size/result.x_size))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([0, result.x_size])
        ax.set_ylim([0, result.y_size])
        
        for pinned_vortex in result.pinning_sites:
            ax.add_artist(plt.Circle(pinned_vortex, result.pinning_size, color='grey', alpha=0.3))
        
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
        plt.savefig(os.path.join(save_dir,  f'vortex_add{i}-size-{sizes[i]}.jpg'))
        plt.close(fig)
    
if __name__ == '__main__':
    # gen_phase_plot('density_sweep_4.0')
    # gen_phase_plot('density_sweep_4.5')
    # gen_phase_plot('density_sweep_5.0')
    # gen_phase_plot('density_sweep_5.5')
    # gen_phase_plot('density_sweep_6.0')
    # animate_file('density_4.5_fixed', os.path.join('results', 'Simulation_results', 'AvalancheResult', 'Density_sweep'))
    animate_folder(os.path.join('results', 'Simulation_results', 'AvalancheResult', 'Density_sweep'), '-fixed')
    # gen_path_plots(os.path.join('results', 'Figures', 'Event_paths', 'Density6.0_events'), 'density_sweep_6.0')
    
    input('Press enter to exit')
