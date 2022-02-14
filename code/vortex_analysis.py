from avalanche_analysis_classes import AvalancheResult
from short_scripts import animate_file, animate_folder

import os
from typing import List
import matplotlib.pyplot as plt

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
    
if __name__ == '__main__':
    # gen_phase_plot('density_sweep_4.5')
    # gen_phase_plot('density_sweep_5.0')
    # gen_phase_plot('density_sweep_5.5')
    # gen_phase_plot('density_sweep_6.0')
    animate_file('density_sweep_5.0', os.path.join('results', 'Simulation_results', 'AvalancheResult', 'Density_sweep_avg'), '-avg')
    
    input('Press enter to exit')
