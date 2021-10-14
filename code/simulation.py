from typing import Tuple
import numpy as np


class Simulation:
    def __init__(self, x_size, y_size, x_repeats, y_repeats):
        self.vortices: np.ndarray = np.empty(shape=(0,2))
        
        self.x_size = x_size
        self.y_size = y_size
        self.x_size_ary = np.array([x_size, 0])
        self.y_size_ary = np.array([0, y_size])
        
        self.x_images = x_repeats
        self.y_images = y_repeats
    
    def create_vortex(self, x_pos, y_pos):
        self.vortices = np.append(self.vortices, [[x_pos, y_pos]], axis=0)
        
    def create_random_vortices(self, num_vortices: int):
        x_vals = np.random.uniform(0, self.x_size, (num_vortices, 1))
        y_vals = np.random.uniform(0, self.y_size, (num_vortices, 1))
        
        self.vortices = np.append(x_vals, y_vals, axis=1)
    
    def run_sim(self):
        pass
        
    def step(self, dt: float):
        images = self.get_images()
        new_vortices = self.vortices.copy()
        for i, vortex in enumerate(self.vortices):
            other_vortices = np.delete(self.vortices, i)
            all_acting_vortices = np.concatenate((other_vortices, images))
            force = self.calculate_force(vortex, all_acting_vortices)
            
            new_vortices[i] += dt*force
        
        self.vortices = new_vortices
    
    def get_images(self):
        x_offset = np.repeat(np.arange(-self.x_images, self.x_images+1), 2*self.y_images+1) * self.x_size
        y_offset = np.tile(np.arange(-self.y_images, self.y_images+1), 2*self.x_images+1) * self.y_size
        
        # Remove the (0, 0) cases (non-image)
        x_offset = np.delete(x_offset, x_offset.size//2, axis=0)
        y_offset = np.delete(y_offset, y_offset.size//2, axis=0)
        
        # Add the offsets to each vortex
        x_images = (x_offset[np.newaxis, :] + self.vortices[:, 0][:, np.newaxis]).flatten()
        y_images = (y_offset[np.newaxis, :] + self.vortices[:, 1][:, np.newaxis]).flatten()
              
        return np.stack((x_images, y_images), axis=1)
    
    def calculate_force(self, vortex_pos, other_pos):
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
    sim = Simulation(2, 3, 1, 1)
    sim.create_random_vortices(5)
    
    sim.get_images()
