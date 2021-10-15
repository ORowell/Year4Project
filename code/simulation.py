import numpy as np
import scipy.special as scipy_s
import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as anim

BAR_FORMAT = '{desc}: {percentage:5.1f}%|{bar}{r_bar}'
HALF_ROOT_3 = np.sqrt(3)/2

class Simulation:
    def __init__(self, x_num, y_num, x_repeats, y_repeats):
        self.vortices: np.ndarray = np.empty(shape=(0, 2))
        
        self.x_size = x_num
        self.y_size = y_num * HALF_ROOT_3
        self.x_size_ary = np.array([self.x_size, 0])
        self.y_size_ary = np.array([0, self.y_size])
        
        self.x_images = x_repeats
        self.y_images = y_repeats
    
    def add_vortex(self, x_pos, y_pos):
        self.vortices = np.append(self.vortices, [[x_pos, y_pos]], axis=0)
        
    def create_random_vortices(self, num_vortices: int):
        x_vals = np.random.uniform(0, self.x_size, (num_vortices, 1))
        y_vals = np.random.uniform(0, self.y_size, (num_vortices, 1))
        
        self.vortices = np.append(x_vals, y_vals, axis=1)
        
    def add_triangular_lattice(self, corner, rows: int, cols: int):
        corner = np.array(corner)
        self.vortices = np.append(self.vortices,
                                  self._generate_lattice_pos(corner, rows, cols), axis=0)
            
    def _generate_lattice_pos(self, corner: np.ndarray, rows: int, cols: int):
        new_vortices: np.ndarray = np.empty(shape=(0, 2))
        for i in range(rows):
            x_vals = np.arange(cols) + 0.5*(i%2)
            y_vals = np.full_like(x_vals, i*HALF_ROOT_3)
            
            new_lattice = np.stack((x_vals, y_vals), axis=1) + corner
            
            new_vortices = np.append(new_vortices, new_lattice, axis=0)
        
        return new_vortices
    
    def run_sim(self, total_time: float, dt: float):
        num_steps = int(total_time/dt)
        self.result = np.empty((num_steps+1, *self.vortices.shape))
        self.result[0] = self.vortices.copy()
        
        for i in tqdm.tqdm(range(num_steps), desc='Simulating', bar_format=BAR_FORMAT):
            self.step(dt)
            self.result[i] = self.vortices.copy()
            
        return self.result            
        
    def step(self, dt: float):
        images = self.get_images()
        all_vortices = np.concatenate((self.vortices, images))
        new_vortices = self.vortices.copy()
        for i, vortex in enumerate(self.vortices):
            acting_vortices = np.delete(all_vortices, i, axis=0)
            force = self.vortices_force(vortex, acting_vortices)
            
            new_vortices[i] += dt*force
        
        self.vortices = new_vortices
        self.wrap_particles()
    
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
    
    def vortices_force(self, vortex_pos, other_pos):
        rel_pos = vortex_pos - other_pos
        distances = np.linalg.norm(rel_pos, axis=1, keepdims=True)
        directions = rel_pos/distances
        
        force_sizes = scipy_s.kn(1, distances)
        forces = force_sizes * directions
        
        return np.sum(forces, axis=0)
    
    def wrap_particles(self):
        x_num_shifts = self.vortices[:, 0]//self.x_size
        x_shift = np.outer(x_num_shifts, self.x_size_ary)
        self.vortices -= x_shift
        
        y_num_shifts = self.vortices[:, 1]//self.y_size
        y_shift = np.outer(y_num_shifts, self.y_size_ary)
        self.vortices -= y_shift
    
    def animate(self, filename, anim_freq=1):
        n_steps, num_vortices, _ = self.result.shape
        n_steps //= anim_freq
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([0, self.x_size])
        ax.set_ylim([0, self.y_size])
        
        self._dots = [ax.plot([], [], 'o', c='r')[0] for i in range(num_vortices)]
        self._anim_freq = anim_freq
        
        animator = anim.FuncAnimation(fig, self._anim_update, n_steps, blit=True)
        
        self.p_bar = tqdm.tqdm(total = n_steps+1, desc='Animating ', unit='fr', bar_format=BAR_FORMAT)
        animator.save(f'results\\{filename}', fps=30)
        self.p_bar.close()
        
    def _anim_update(self, frame_num):
        self.p_bar.update(1)
        for i, dot in enumerate(self._dots):
            dot.set_data(self.result[frame_num*self._anim_freq, i])
            
        return self._dots
    
def ground_state():
    sim = Simulation(2, 2, 1, 1)
    
    sim.add_triangular_lattice((0, 0), 2, 2)
    
    sim.run_sim(0.2, 0.0001)
    sim.animate('groundstate.gif', 10)
    
def many_vortices():
    sim = Simulation(4, 3, 2, 2)
    
    sim.create_random_vortices(150)
    
    sim.run_sim(0.4, 0.001)
    sim.animate('lots.gif')
    
if __name__ == '__main__':
    ground_state()
    # many_vortices()
