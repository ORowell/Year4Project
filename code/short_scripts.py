from typing import Optional, Union
from avalanche_analysis_classes import AvalancheAnimator, AvalancheResult

import os

def animate_file(filename: str, directory: str, output_ext: str = '', freq: Optional[int] = None,
                 event_range: Union[int, slice] = slice(None)):
    result = AvalancheResult.load(filename, directory)
    if result is None:
        print(f'Failed to load {filename}', flush=True)
        return
    if freq is None:
        print(f'{result.movement_cutoff = }, {result.movement_cutoff_time = }')
        print(f'{result.dt = }, {result.vortices_added = }')
        freq = int(input(f'{len(result.flatten(event_range))} to animate. Enter frequency: '))
    
    animator = AvalancheAnimator()
    animator.animate(result, f'{filename}{output_ext}.gif', freq, event_range)
    
    return freq

def animate_folder(directory: str, output_ext: str = '', single_freq: bool = True):
    freq_used = None
    for filename in os.listdir(directory):
        freq_used = animate_file(filename, directory, output_ext, freq_used)
        if not single_freq:
            freq_used = None
        
if __name__ == '__main__':
    animate_folder('results\\Simulation_results\\AvalancheResult\\New_pins', single_freq=True)
    # animate_file('continued_6.0', 'results\\Simulation_results\\AvalancheResult\\Density_sweep')
