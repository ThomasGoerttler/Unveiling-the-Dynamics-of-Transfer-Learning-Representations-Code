import numpy as np
def resize(array):
    array = np.concatenate(array)
    return array

def store_array_to_wandb(wandb, array, base_name='TEST/pool', step=0):
    for i, value in enumerate(array):
        wandb.log({
            base_name + str(i + 1): value,
        }, step=step)
