from vortex_avalanches import AvalancheAnimator, AvalancheResult

import os

def animate_file(filename: str, directory: str):
    result = AvalancheResult.load(filename, directory)
    if result is None:
        print(f'Failed to load {filename}')
        return
    print(f'{result.movement_cutoff = }, {result.movement_cutoff_time = }')
    print(f'{result.dt = }, {result.vortices_added = }')
    freq = int(input(f'{result.flattened_num_t} to animate. Enter frequency: '))
    
    animator = AvalancheAnimator()
    animator.animate(result, f'{filename}.gif', freq)

def animate_folder(directory: str):
    for filename in os.listdir(directory):
        animate_file(filename, directory)
        
if __name__ == '__main__':
    # animate_folder('results\\Simulation_results\\AvalancheResult\\Density_sweep')
    animate_file('density_sweep_6.0', 'results\\Simulation_results\\AvalancheResult\\Density_sweep')