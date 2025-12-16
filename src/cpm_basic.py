import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import os
from src.util import neighbors_type_dict
from skimage.measure import perimeter as measure_perimeter

WORKING_DIR = os.getcwd()

class CellularPottsModel:
    def __init__(self, 
                 n_cells:int=50, 
                 n_types: int=2, 
                 T: int=1, 
                 L:int=100,
                 type_percentages: list[float]=None,
                 volume_coefficient=0, # 1D array of Cv for each type (length n_types+1)
                 perimeter_coefficient=0, # 1D array of Cp for each type (length n_types+1)  
                 adhessions=None,  # 2D array of J values (shape (n_types+1, n_types+1))
                 lattice_type: str="hex",
                 object_volumes: list[float]=None,
                 periodic: bool=False):
        
        self.L = L 
        self.n_cells = n_cells
        self.T = T
        self.n_types = n_types
        self.type_percentages = type_percentages
        self.volume_coefficient = volume_coefficient
        self.perimeter_coefficient = perimeter_coefficient
        self.adhessions = adhessions  # list of adhessions(should be a flatten [n_typesxn_types] matrix)
        self.periodic = periodic
        
        self.object_volumes = object_volumes  # None or a list of target volume for each cell
        
        self.tau = self.set_cell_type()  # is dict of cells (keys) and the cell types
        if lattice_type is None:
            self.lattice = np.random.randint(1, self.n_cells + 1, size=(self.L, self.L))
        elif lattice_type == "hex":
            self.lattice = self.init_hexlattice()
        elif lattice_type == "surrounded_cell":
            self.lattice = self.init_hex_surroundedcell_lattice()
        elif lattice_type == "circle":
            self.lattice = self.init_circlelattice()
        elif lattice_type == "prerun":
            self.lattice = self.init_prerunlattice()
        elif lattice_type == "two_cells":
            self.lattice = self.init_two_cells_lattice()
        else:
            raise ValueError("Either random, circle, prerun, two_cells, surrounded_cell or hex, or implement other shape init.")

        self.J = self.set_adhesion_coefficient_J() # is a 2D array of J values
        self.C_v = self.set_volume_coefficient_Cv()  # is a 1D array of Cv for each type
        self.C_p = self.set_perimeter_coefficient_Cp()  # is a 1D array of Cp for each type
        self.V = self.set_object_volumes()  # is a dict of the cells (keys) and objective volumes
        self.P = self.set_object_perimeters()  # is a dict of the cells (keys) and objective perimeters
        self.volume_unit = 1
        self.prerunsteps = 1e5

    #-------------------------------------------------------
    # SET UP FOR PRACTICALITIES, COEFFICIENTS AND TYPE/CELL BASED CONSTANTS: 
    #-------------------------------------------------------

    def neighbors_2d_periodic(self, point_index):
        """
        Returns a list of the neighbors (periodic boundaries) for element (i, j).
        """
        i, j = point_index
        neighbors = [
                    ((i - 1) % self.L, (j - 1) % self.L), 
                    ((i - 1) % self.L, j),
                    ((i - 1) % self.L, (j + 1) % self.L),
                    (i, (j - 1) % self.L),
                    (i, (j + 1) % self.L),
                    ((i + 1) % self.L, (j - 1) % self.L),
                    ((i + 1) % self.L, j),
                    ((i + 1) % self.L, (j + 1) % self.L),
                    ]
        return neighbors
    
    def neighbors_2d_non_periodic(self, point_index):
        """
        Returns a list of the neighbors (non-periodic boundaries) for element (i, j).
        """
        i, j = point_index
        potential_neighbors = [
                    (i - 1, j - 1), 
                    (i - 1, j),
                    (i - 1, j + 1),
                    (i, j - 1),
                    (i, j + 1),
                    (i + 1, j - 1),
                    (i + 1, j),
                    (i + 1, j + 1),
                    ]
        neighbors = []
        for ni, nj in potential_neighbors:
            if 0 <= ni < self.L and 0 <= nj < self.L:
                neighbors.append((ni, nj))                
        return neighbors
    
    
    def init_hexlattice(self):
        """
        Function that will init a grid with hexagons (more voronoi-ish actually)
        """
        # Estimate hex spacing
        spacing = int(np.sqrt(self.L*self.L / self.n_cells))

        # Generate hex-grid of candidate centers
        centers = []
        for j in range(0, self.L, spacing):
            shift = (j // spacing) % 2 * (spacing // 2)
            for i in range(shift, self.L, spacing):
                centers.append((i, j))

        centers = np.array(centers)

        # If too many or too few, randomly select 50
        if len(centers) > self.n_cells:
            idx = np.random.choice(len(centers), self.n_cells, replace=False)
            centers = centers[idx]
        elif len(centers) < self.n_cells:
            # pad with random ones if too few
            extra = np.column_stack((
                np.random.randint(0, self.L, self.n_cells - len(centers)),
                np.random.randint(0, self.L, self.n_cells - len(centers))
            ))
            centers = np.vstack([centers, extra])

        yy, xx = np.mgrid[0:self.L, 0:self.L] # (L,L)
        coords = np.stack((xx, yy), axis=-1)  # (L,L,2)

        # Reshape for broadcasting:
        # coords → (L,L,1,2)
        # centers → (1,1,n_cells,2)
        coords_b = coords[:, :, None, :]              # add axis
        centers_b = centers[None, None, :, :]         # add axis
        d2 = np.sum((coords_b - centers_b)**2, axis=3)   

        lattice = np.argmin(d2, axis=2).astype(np.int32) + 1
        return lattice

    def init_circlelattice(self):
        """
        Deterministic placement of circles on a regular grid.
        Overlap is allowed.
        """
        # number of grid points in each dimension
        k = int(np.ceil(np.sqrt(self.n_cells)))

        # spacing between circle centers
        spacing = 1.5 * self.L / k

        # compute deterministic centers
        centers = []
        for iy in range(k):
            for ix in range(k):
                if len(centers) >= self.n_cells:
                    break
                cx = (ix + 0.5) * spacing
                cy = (iy + 0.5) * spacing
                centers.append((cx, cy))

        # create grid
        grid = np.zeros((self.L, self.L), dtype=np.uint8)

        yy, xx = np.indices((self.L, self.L))

        # same radius rule as you used
        radius = np.sqrt((self.L**2/(self.n_cells))/np.pi)/1.8

        for i, (cx, cy) in enumerate(centers):
            mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2
            grid[mask] = i + 1

        return grid


    def init_prerunlattice(self):
        """
        Function that will init a grid with random values for prerun
        """
        lattice = np.load(f"{WORKING_DIR}/saves/init_grids/initial_grid.npy")
        self.L = lattice.shape[0]
        list_of_cells = np.unique(lattice)
        if 0 in list_of_cells:
            self.n_cells = len(list_of_cells) - 1
        else:
            self.n_cells = len(list_of_cells)
        return lattice
    
    def init_hex_surroundedcell_lattice(self):
        """
        Function that will init a grid with hexagons, and cells surrounded by other cells
        """
        if self.n_types != 2:
            raise ValueError("n_types must be 2 for this lattice type to work.")
        if self.type_percentages is None: 
            raise ValueError("Cell Type Percentage must be defined for this lattice type to work.")
        elif isinstance(self.type_percentages, (list, np.ndarray)):
            if len(self.type_percentages) != self.n_types:
                raise ValueError("Length of type_percentages must equal n_types.")
            if not np.isclose(np.sum(self.type_percentages), 1.0):
                raise ValueError("type_percentages must sum to 1.")
            if len(np.unique(self.type_percentages)) != 2:
                raise ValueError("type_percentages must have two unique values for this lattice type to work.")

        grid = self.init_hexlattice()
        majority_type = np.argmax(self.type_percentages) + 1  # +1 because types start at 1
        tau = {i: majority_type for i in range(1, self.n_cells + 1)}  # all cells set to majority type
        
        border_cells = np.unique(np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]]))
        valid_cells = np.setdiff1d(np.arange(1, self.n_cells + 1), border_cells)  # exclude border cells

        minority_cell_number = int(self.n_cells * np.min(self.type_percentages))
        failed_attempts = 0
        for _ in range(minority_cell_number):
            cell_to_change = np.random.choice(valid_cells)
            neighbor_dict = neighbors_type_dict(grid, tau)
            if all(neighbor_dict[cell_to_change] == majority_type):
                tau[cell_to_change] = 3 - majority_type  # switch between 1 and 2
                valid_cells = valid_cells[valid_cells != cell_to_change]  # remove from valid cells
            else:
                failed_attempts += 1

        if failed_attempts > 0:  # try again for failed attempts, does not guarantee success but better than nothing
            for _ in range(failed_attempts):
                cell_to_change = np.random.choice(valid_cells)
                neighbor_dict = neighbors_type_dict(grid, tau)
                if all(neighbor_dict[cell_to_change] == majority_type):
                    tau[cell_to_change] = 3 - majority_type  # switch between 1 and 2
                    valid_cells = valid_cells[valid_cells != cell_to_change]  # remove from valid cells
        self.tau = tau  # update the cell type dictionary
        return grid


    def init_two_cells_lattice(self):
        """
        Function that will init a grid with two cells in the center
        """
        lattice = np.load(f"{WORKING_DIR}/saves/init_grids/initial_grid_two_cells.npy")
        self.L = lattice.shape[0]
        self.n_cells = 2
        self.object_volumes = [0, 1257, 1257]  # set object volumes for the two cells
        return lattice

    def set_cell_type(self):
        tau = {}  # cell type dict
        tau[0] = 0  # background type is 0
        if self.type_percentages is None:
            for i in range(1, self.n_cells + 1):
                tau[i] = np.random.randint(1, self.n_types + 1)  # randomly assign type 1 or 2

        elif len(self.type_percentages) != self.n_types:
            raise ValueError("Length of type_percentages must equal n_types.")
        
        else: # if type percentages are given
            cell_types = []
            for t, p in zip(range(1, self.n_types + 1), self.type_percentages):
                cell_types.extend([t] * int(p * self.n_cells))
            if len(cell_types) < self.n_cells:  # fill in any remaining cells with the last type
                cell_types.extend([self.n_types] * (self.n_cells - len(cell_types)))

            np.random.shuffle(cell_types)
            for i, t in enumerate(cell_types, start=1):
                tau[i] = t
            
        return tau
        
    def set_object_volumes(self):
        V = {}  # volume dict
        V[0] = 0  # background volume is 0
        if self.object_volumes is None:
            for i in range(1, self.n_cells + 1): # for each cell identifier
                #V[i] = ((self.L * self.L) / (self.n_cells))
                V[i] = (np.sqrt(self.L**2/self.n_cells)/2)**2 * np.pi  # approximate volume from radius assuming circular shape
        elif self.object_volumes is not None:
            for i, vol in enumerate(self.object_volumes):
                V[i] = vol
        return V
           
    def set_object_perimeters(self):
        P = {}  # perimeter dict
        P[0] = 0  # background perimeter is 0
        for i in range(1, self.n_cells + 1): # for each cell identifier
            P[i] = 2*np.sqrt(self.V[i]/np.pi)  # approximate perimeter from volume assuming circular shape
        return P

    def set_adhesion_coefficient_J(self):
        if self.adhessions is None:
            J = np.ones([self.n_types+1, self.n_types+1])
        else:
            J = self.adhessions
        return J
    
    def set_volume_coefficient_Cv(self):

        if not isinstance(self.volume_coefficient, (np.ndarray, list)):
            C_v = np.zeros(self.n_types + 1)
        elif self.volume_coefficient is None:
            C_v = np.ones(self.n_types + 1) * 5 # default value 
            C_v[0] = 0  # background has no volume constraint
        else:
            C_v = self.volume_coefficient  # array of C_v for each type
        return C_v

    def set_perimeter_coefficient_Cp(self):
        if not isinstance(self.perimeter_coefficient, (np.ndarray, list)):
            C_p = np.zeros(self.n_types + 1)
        elif self.perimeter_coefficient is None:
            C_p = np.ones(self.n_types + 1) * 5 # default value 
            C_p[0] = 0  # background has no perimeter constraint
        else:
            C_p = self.perimeter_coefficient  # array of C_p for each type
        return C_p

    #-------------------------------------------------------
    # HAMILTONIAN CALCULATION LOGIC
    #-------------------------------------------------------
    
    def adhesion_term(self, 
                      point_index, 
                      point_value, 
                      grid):
        """
        Calculate the adhession term.
        """
        if self.periodic:
            neighbors = self.neighbors_2d_periodic(point_index)
        else:
            neighbors = self.neighbors_2d_non_periodic(point_index)
        neighbor_values = np.array([grid[r, c] for r, c in neighbors])
        different_mask = (neighbor_values != point_value)  # The neighbors that are not the same as the point
        diffs = [self.J[self.tau.get(point_value), self.tau.get(nval)] 
                 for nval in neighbor_values[different_mask]] 
        H_adh = np.sum(diffs)
        return H_adh
    
    def volume_term(self, 
                    point_value, 
                    grid,
                    new: bool=None, 
                    source: bool=None):
        """
        Calculate the volume term.
        """
        if np.all(self.C_v == 0):  # so it does not calculate unnecessarily
            return 0
        else:
            object_vol = self.V.get(point_value)
            if new and not source: # if we were to change the point, and the point we are looking at is the target point
                current_vol = np.sum(grid == point_value) - self.volume_unit
            elif new and source: # if we were to change the point, and the point we are looking at is the source point
                current_vol = np.sum(grid == point_value) + self.volume_unit
            else:
                current_vol = np.sum(grid == point_value)
            H_vol = self.C_v[self.tau.get(point_value)] * (current_vol - object_vol)**2
            return H_vol
    
    def perimeter_term(self,
                       point_value,
                       source_point_value,
                       target_point_index,
                       grid, 
                       new: bool):
        """
        Calculate the perimeter term.
        """
        if np.all(self.C_p == 0):  # so it does not calculate unnecessarily
            return 0
        else:            
            def compute_perimeter(grid, value):
                """
                Compute perimeter of all pixels with a given value,
                It counts one pixel as perimeter zero. 
                """
                # Binary mask for this cell type
                mask = (grid == value)
                if self.periodic:
                    # Pad with wrap (periodic)
                    mask_wrapped = np.pad(mask, pad_width=1, mode='wrap')
                else:
                    mask_wrapped = np.pad(mask, pad_width=1, mode='constant', constant_values=0)

                # Label connected components (in case cell_value appears multiple times)
                perimeter = measure_perimeter(mask_wrapped, neighborhood=8)
                return perimeter
            
            grid_copy = grid.copy()
            if new: # Assume we are changing the target point to the source point
                grid_copy[target_point_index] = source_point_value

            P = compute_perimeter(grid_copy, point_value)
       
            H_perim = self.C_p[self.tau.get(point_value)] * (P - self.P.get(point_value))**2
            return H_perim

    def calculate_H(self,
                    source_point, target_point, 
                    target_point_index, grid, 
                    new: bool):
        """
        Calculate the the Hamiltonian (total energy).
        """
        if new: # Assume we are changing the target point to the source point  
            H = (
                self.adhesion_term(point_index=target_point_index, point_value=source_point, grid=grid) + 
                    (
                    self.volume_term(point_value=target_point, grid=grid, new=new, source=False) + 
                    self.volume_term(point_value=source_point, grid=grid, new=new, source=True)
                    ) +
                    (
                    self.perimeter_term(point_value=source_point,
                                        source_point_value=source_point,
                                        target_point_index=target_point_index,
                                        grid=grid,
                                        new=new) +
                    self.perimeter_term(point_value=target_point,
                                        source_point_value=target_point,
                                        target_point_index=target_point_index,
                                        grid=grid,
                                        new=new)
                    )
                )
        else:
            H = (
                self.adhesion_term(point_index=target_point_index, point_value=target_point, grid=grid) + 
                    (
                    self.volume_term(point_value=target_point, grid=grid, new=new, source=False) + 
                    self.volume_term(point_value=source_point, grid=grid, new=new, source=True)
                    ) +
                    (
                    self.perimeter_term(point_value=source_point,
                                        source_point_value=source_point,
                                        target_point_index=target_point_index,
                                        grid=grid,
                                        new=new) +
                    self.perimeter_term(point_value=target_point,
                                        source_point_value=target_point,
                                        target_point_index=target_point_index,
                                        grid=grid,
                                        new=new)
                    )
                )
        return H

    def total_H(self, grid):
        """
        Calculate the total Energy of the grid.
        """
        total_H = 0
        for i in range(self.L):
            for j in range(self.L):
                point_value = grid[i, j]
                total_H += self.adhesion_term(point_index=(i, j), point_value=point_value, grid=grid)
        
        # Volume term
        for cell_id in range(1, self.n_cells + 1):
            total_H += (self.volume_term(point_value=cell_id, grid=grid, new=False) +
                        self.perimeter_term(point_value=cell_id,
                                        source_point_value=None,
                                        target_point_index=None,
                                        grid=grid,
                                        new=False))
        return total_H

    #---------------------------------------------------------
    # MONTE CARLO STEP AND ANIMATION LOGIC
    #---------------------------------------------------------

    def step(self, grid):
        source_point_index = [np.random.randint(self.L), np.random.randint(self.L)]
        source_point = grid[source_point_index[0], source_point_index[1]]
        if self.periodic:
            neighbors = self.neighbors_2d_periodic(source_point_index)
        else:
            neighbors = self.neighbors_2d_non_periodic(source_point_index)
        target_point_index = neighbors[np.random.randint(len(neighbors))]
        target_point = grid[target_point_index[0], target_point_index[1]]

        if (source_point == target_point) or (source_point == 0):
            pass  # Skip if the target and source are the same cell
        else: 
            H_old = self.calculate_H(source_point=source_point,
                                     target_point=target_point,
                                     target_point_index=target_point_index, 
                                     grid=grid,
                                     new=False)
            
            H_new = self.calculate_H(source_point=source_point,
                                     target_point=target_point,
                                     target_point_index=target_point_index, 
                                     grid=grid,
                                     new=True)
            dH = H_new - H_old

            if dH < 0 or np.random.random() < np.exp(-dH / self.T):
                grid[target_point_index] = source_point


    def run_a_sim(self, steps:int):
        grid = self.lattice.copy()
        for _ in tqdm(range(steps)):
            self.step(grid)
        return grid
    
    def run_time_development_sim(self, steps, interval=1000):
        """
        Run a simulation and store energy at given intervals."
        """
        grid = self.lattice.copy()
        energy = []
        for step in tqdm(range(steps)):
            self.step(grid)
            if step % interval == 0:
                energy.append(self.total_H(grid))
        return energy
    
    def run_animation(self, steps_per_frame=2000, frames=100):
        grid = self.lattice.copy()

        fig, ax = plt.subplots()
        img = ax.imshow(grid, cmap="gist_ncar_r", interpolation="nearest")

        def update(frame):
            for _ in range(steps_per_frame):
                self.step(grid)
            img.set_data(grid)
            return (img,)

        self.anim = FuncAnimation(
            fig,
            update,
            frames=frames,
            interval=16,
            blit=True
        )
        plt.show()

    def save_animation_gif(self, steps_per_frame=2000, frames=100, output_file="animation"):
        grid = self.lattice.copy()

        fig, ax = plt.subplots()
        img = ax.imshow(grid, cmap="gist_ncar_r", interpolation="nearest")
        ax.axis('off')  # hides axes for a cleaner GIF

        def update(frame):
            for _ in range(steps_per_frame):
                self.step(grid)
            img.set_data(grid)
            return (img,)

        anim = FuncAnimation(fig, update, frames=frames, blit=True)

        # Save as GIF
        writer = PillowWriter(fps=10)  # adjust fps as needed
        anim.save(f"{WORKING_DIR}/figures/{output_file}.gif", writer=writer)
        plt.close(fig)
