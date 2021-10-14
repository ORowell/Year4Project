import numpy as np


class Simulation:
    def __init__(self, x_size, y_size):
        self.vortices = np.empty(shape=(0,2))
        
        self.x_size = x_size
        self.y_size = y_size
        self.x_size_ary = np.array([x_size, 0])
        self.y_size_ary = np.array([0, y_size])
    
    def create_vortex(self, x_pos, y_pos):
        self.vortices =  np.append(self.vortices, [[x_pos, y_pos]], axis=0)
        
    def create_random_vortices(self, num_vortices):
        x_vals = np.random.uniform(0, self.x_size, (num_vortices, 1))
        y_vals = np.random.uniform(0, self.y_size, (num_vortices, 1))
        
        self.vortices = np.append(x_vals, y_vals, axis=1)
    
    def run_sim(self):
        pass
        
    def step(self):
        pass
    
    def wrap_particles(self):
        x_num_shifts = self.vortices[:, 0]//self.x_size
        x_shift = np.outer(x_num_shifts, self.x_size_ary)
        self.vortices -= x_shift
        
        y_num_shifts = self.vortices[:, 1]//self.y_size
        y_shift = np.outer(y_num_shifts, self.y_size_ary)
        self.vortices -= y_shift
    
    def animate(self):
        pass
    
if __name__ == '__main__':
    pass
