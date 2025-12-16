import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from src.cpm_basic import CellularPottsModel
import logging
logging.basicConfig(level=logging.INFO)

def run_one_sim(i, params):
    logging.info(f"Starting simulation {i} in process {os.getpid()}")
    cpm = CellularPottsModel(
        n_cells=params["n_cells"],
        n_types=params["n_types"],
        T=params["T"],
        L=params["L"],
        type_percentages=params["type_percentages"],
        adhessions=params["adhesions"],
        volume_coefficient=params["volume_coefficient"],
        perimeter_coefficient=params["perimeter_coefficient"],
        lattice_type=params["lattice_type"],
        periodic=params["periodic"],
    )

    return (cpm.run_a_sim(steps=params["steps"]), cpm.tau) # the final lattice and cell types


def run_parallel(repeat, params, max_workers=None):
    """
    Run multiple simulations in parallel using ProcessPoolExecutor.
    """
    ctx = mp.get_context("spawn")  # notebook-safe
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx
    ) as executor:
        return list(executor.map(run_one_sim, range(repeat), [params] * repeat))
