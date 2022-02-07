from vortex_avalanches import AvalancheAnimator, AvalancheResult

import os

def animate_folder(directory: str):
    for filename in os.listdir(directory):
        result = AvalancheResult.load(filename, directory)
        if result is None:
            print(f'Failed to load {filename}, continuing to next file')
            continue
        
        print(f'{result.movement_cutoff = }, {result.movement_cutoff_time = }')
        print(f'{result.dt = }, {result.vortices_added = }')
        freq = int(input(f'{result.flattened_num_t} to animate. Enter frequency: '))
        
        animator = AvalancheAnimator()
        animator.animate(result, f'{filename}.gif', freq)
        
if __name__ == '__main__':
    animate_folder('results\\Simulation_results\\AvalancheResult\\Density_sweep')