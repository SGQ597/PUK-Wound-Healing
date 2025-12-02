import numpy as np


def color_code_cells(grid, type_cell):
    """
    Function that color code each cell based on its type with small random variations.
    Only takes two types plus background (0).
    """
    # Define base colors for each type (as RGB)
    type_base_colors = {
        0: np.array([1, 1, 1]),         # white for background
        1: np.array([0.5, 0.7, 1]),     # blue-ish for type 1
        2: np.array([1.0, 0.6, 0.6])      # red-ish for type 2
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