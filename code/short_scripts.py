from typing import Optional
from avalanche_analysis_classes import AvalancheAnimator, AvalancheResult

import os

def animate_file(filename: str, directory: str, output_ext: str = '', freq: Optional[int] = None):
    result = AvalancheResult.load(filename, directory)
    if result is None:
        print(f'Failed to load {filename}')
        return
    if freq is None:
        print(f'{result.movement_cutoff = }, {result.movement_cutoff_time = }')
        print(f'{result.dt = }, {result.vortices_added = }')
        freq = int(input(f'{result.flattened_num_t} to animate. Enter frequency: '))
    
    animator = AvalancheAnimator()
    animator.animate(result, f'{filename}{output_ext}.gif', freq)
    
    return freq

def animate_folder(directory: str, output_ext: str = '', single_freq: bool = True):
    freq_used = None
    for filename in os.listdir(directory):
        freq_used = animate_file(filename, directory, output_ext, freq_used)
        if not single_freq:
            freq_used = None
        
if __name__ == '__main__':
    # animate_folder('results\\Simulation_results\\AvalancheResult\\Density_sweep')
    animate_file('density_sweep_3.0', 'results\\Simulation_results\\AvalancheResult\\Density_sweep')
