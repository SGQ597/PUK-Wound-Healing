import numpy as np
from skimage.morphology import convex_hull_image
from collections import defaultdict

#--------------------------------------------------
# Visualization Functions
#--------------------------------------------------

def color_code_cells(grid, type_cell):
    """
    Function that color code each cell based on its type with small random variations.
    Only takes two types plus background (0).
    """
    # Define base colors for each type (as RGB)
    type_base_colors = {
        0: np.array([1, 1, 1]),         # white for background
        1: np.array([1.0, 0.6, 0.6]),    # red-ish for type 1
        2: np.array([0.3, 0.8, 0.3])    # green-ish for type 2
    }
    unique_cells = np.unique(grid)
    # Build an array of base colors for each unique cell
    base_colors = np.array([type_base_colors[type_cell[cell]] for cell in unique_cells])
    variations = (np.random.rand(len(unique_cells), 3) - 0.5) * 0.2

    # Cells with id 0 get no variation
    zero_mask = (unique_cells == 0)
    variations[zero_mask] = 0.0

    # Final colors for each unique cell
    all_colors = np.clip(base_colors + variations, 0, 1)
    # Map: cell_id → row index in unique_cells array
    cell_to_index = {cell: idx for idx, cell in enumerate(unique_cells)}
    # Build an index array matching A
    index_grid = np.vectorize(cell_to_index.get)(grid)

    # Build final color grid (vectorized gather)
    color_grid = all_colors[index_grid]
    return color_grid

#--------------------------------------------------
# Cell Shape Analysis Functions
#--------------------------------------------------

def calculate_convexity_ratio(grid_array, target_value:int):
    """
    Calculates the Area-Based Convexity Ratio for a cell in a 2D grid.
    Convexity Ratio = Area(Shape) / Area(Convex Hull)
    """
    # 1. Define the shape mask (S)
    shape_mask = (grid_array == target_value)
    area_s = np.sum(shape_mask)  # area of the shape
    # 2. Calculate the Convex Hull Image (CH(S))
    convex_hull = convex_hull_image(shape_mask)  # fills any concavities in the shape_mask
    area_ch = np.sum(convex_hull) # area of the convex hull

    # 3. Calculate the Convexity Ratio
    ratio = area_s / area_ch
    return ratio

def non_boundary_convexity(grid, cell_types: dict):
    cell_type_convexity = np.zeros(len(np.unique(list(cell_types.values()))))
    cell_types_count = np.zeros(len(np.unique(list(cell_types.values()))))
    labels = np.unique(grid)
    labels = labels[labels != 0]  # exclude background

    rows, cols = grid.shape
    
    for v in labels:
        # Create mask for this label
        mask = (grid == v)
        coords = np.where(mask)
        if (np.any(coords[0] == 0) or np.any(coords[0] == rows - 1) or 
            np.any(coords[1] == 0) or np.any(coords[1] == cols - 1)):
            continue  # Skip cells touching the boundary
        else:
            ratio = calculate_convexity_ratio(grid, target_value=v)
            cell_type = cell_types.get(v)
            if cell_type is not None:
                cell_type_convexity[cell_type] += ratio
                cell_types_count[cell_type] += 1
    cell_type_convexity = np.divide(cell_type_convexity, cell_types_count, 
                                    out=np.zeros_like(cell_type_convexity), 
                                    where=cell_types_count != 0)  # avoid division by zero
    return cell_type_convexity  # average convexity per cell type


def non_boundary_convexity_two_types(grid, cell_types: dict):
    cell_type_1_convexity  = []
    cell_type_2_convexity  = []
    cell_types_count = np.zeros(len(np.unique(list(cell_types.values()))))
    labels = np.unique(grid)
    labels = labels[labels != 0]  # exclude background

    rows, cols = grid.shape
    
    for v in labels:
        # Create mask for this label
        mask = (grid == v)
        coords = np.where(mask)
        if (np.any(coords[0] == 0) or np.any(coords[0] == rows - 1) or 
            np.any(coords[1] == 0) or np.any(coords[1] == cols - 1)):
            continue  # Skip cells touching the boundary
        else:
            ratio = calculate_convexity_ratio(grid, target_value=v)
            cell_type = cell_types.get(v)
            if cell_type is not None:
                if cell_type == 1:
                    cell_type_1_convexity.append(ratio)
                elif cell_type == 2:
                    cell_type_2_convexity.append(ratio)
                cell_types_count[cell_type] += 1
    return cell_type_1_convexity, cell_type_2_convexity, cell_types_count  # lists of convexity per cell type


#--------------------------------------------------
# Cell Neighbor Functions
#--------------------------------------------------

def neighbors_type_dict(A, A_types):
    """
    A function that returns a dictionary 
    where each cell id maps to a list of cell types of its neighboring cells.
    """
    neighbors = defaultdict(set)
    def add_neighbors(X, Y):
        mask = X != Y
        for a, b in zip(X[mask], Y[mask]):
            neighbors[a].add(b)
            neighbors[b].add(a)
    # vertical
    add_neighbors(A[:-1, :], A[1:, :])
    # horizontal
    add_neighbors(A[:, :-1], A[:, 1:])
    # diagonal upper left to lower right
    add_neighbors(A[:-1, :-1], A[1:, 1:])
    # diagonal lower left to upper right
    add_neighbors(A[:-1, 1:], A[1:, :-1])


    neighbors_type = {cell: [A_types[n] for n in nbs] for cell, nbs in neighbors.items()}
    return neighbors_type

#--------------------------------------------------
# Isolated Punisher Detection
#--------------------------------------------------

def isolated_punisher(grid, value, periodic):
    # Binary mask for this cell type
    mask = (grid == value)
    if periodic:
        # Pad with wrap (periodic)
        mask_wrapped = np.pad(mask, pad_width=1, mode='wrap')
    else:
        mask_wrapped = np.pad(mask, pad_width=1, mode='constant', constant_values=0)

    isolated = (
                mask_wrapped[1:-1, 1:-1] &          # cell itself
                ~mask_wrapped[:-2, 1:-1] &          # up
                ~mask_wrapped[2:, 1:-1] &           # down
                ~mask_wrapped[1:-1, :-2] &          # left
                ~mask_wrapped[1:-1, 2:]  &           # right
                ~mask_wrapped[ :-2, :-2] &        # up-left
                ~mask_wrapped[ :-2, 2:] &         # up-right
                ~mask_wrapped[ 2:, :-2] &         # down-left
                ~mask_wrapped[ 2:, 2:]             # down-right
            )  # moore neighborhood
    return np.any(isolated)  # Returns True if there are isolated pixels