from simulation import Simulation, HALF_ROOT_3

import numpy as np

class ChannelSimulation(Simulation):
    def __init__(self, x_num, y_num, x_repeats, y_repeats):
        super().__init__(x_num, y_num, x_repeats, y_repeats)
        
        self.pinned_vortices = np.empty(shape=(0, 2))
        
    @classmethod
    def create_channel(cls, channel_width, pinned_width, channel_length, repeats):
        obj = cls(channel_length, channel_width + 2*pinned_width, repeats, 0)
        
        obj.add_pinned_lattice((0, HALF_ROOT_3/2), pinned_width, channel_length, pinned_width%2)
        top_channel_base = (channel_width + pinned_width + 0.5) * HALF_ROOT_3
        obj.add_pinned_lattice((0, top_channel_base), pinned_width, channel_length, channel_width%2)
        
        obj.add_triangular_lattice((0, (pinned_width + 0.5)*HALF_ROOT_3), channel_width, channel_length)
        
        return obj
        
    def get_all_vortices(self):
        all_vortices = np.concatenate((self.vortices, self.pinned_vortices))
        images = self.get_images(all_vortices)
        return np.concatenate((all_vortices, images))
    
    def add_pinned_lattice(self, corner, rows: int, cols: int, offset: bool=False):
        corner = np.array(corner)
        self.pinned_vortices = np.append(self.pinned_vortices,
                                         self._generate_lattice_pos(corner, rows, cols, offset), axis=0)
        
    def _anim_init(self, num_vortices):
        fig, ax = super()._anim_init(num_vortices)
        
        for vortex in self.pinned_vortices:
            ax.plot(vortex[0], vortex[1], 'x', c='k')
            
        return fig, ax

def plain_channel():
    sim = ChannelSimulation.create_channel(1, 4, 10, 1)
    
    sim.run_sim(0.5, 0.0001)
    
    sim.animate('channel.gif', 10)
    
def current_channel():
    sim = ChannelSimulation.create_channel(2, 4, 10, 1)
    sim.current_force = np.array((0.1, 0))
    
    T = 200
    sim.run_sim(T, T/1e4)
    sim.animate('current_channel_2w.gif', 10)
    
if __name__ == '__main__':
    # plain_channel()
    current_channel()
