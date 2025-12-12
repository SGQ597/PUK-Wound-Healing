import numpy as np
from skimage.morphology import convex_hull_image


def color_code_cells(grid, type_cell):
    """
    Function that color code each cell based on its type with small random variations.
    Only takes two types plus background (0).
    """
    # Define base colors for each type (as RGB)
    type_base_colors = {
        0: np.array([1, 1, 1]),         # white for background
        1: np.array([0.5, 0.7, 1]),     # blue-ish for type 1
        2: np.array([1.0, 0.6, 0.6])    # red-ish for type 2
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
    return cell_type_convexity