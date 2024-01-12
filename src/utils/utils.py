import numpy as np
import argparse
import time

def resize(array):
    array = np.concatenate(array)
    return array

def store_array_to_wandb(wandb, array, base_name='TEST/pool', step=0):
    for i, value in enumerate(array):
        wandb.log({
            base_name + str(i + 1): value,
        }, step=step)

def parse_args():
    parser = argparse.ArgumentParser(description="Your script description")

    # Add command-line arguments for each configuration parameter
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--keep_prob', type=float, default=0.0, help='Keep probability')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--model', type=str, default='conv4', help='Model name')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='List of seeds')
    parser.add_argument('--finetuning', action='store_true', help='Enable finetuning')
    parser.add_argument('--finetuning_size', type=int, default=5000, help='Finetuning size')
    parser.add_argument('--pre_trained_dataset', type=str, default='cifar10', help='Pre-trained dataset name')
    parser.add_argument('--degree_of_randomness', type=int, default=0, help='Degree of randomness')
    parser.add_argument('--pre_trained_size', type=int, default=50000, help='Pre-trained size')

    return parser.parse_args()

def get_standard_error(std, sample_size):
    return std/np.sqrt(sample_size)


def load_partial_state_dict(model, pretrained_state_dict, exclude_layer='fc'):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if exclude_layer not in k}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
def get_confidence_interval(std, sample_size, confidence=95):
    if confidence == 90:
        multiplier = 1.645
    elif confidence == 95:
        multiplier = 1.96
    elif confidence == 99:
        multiplier = 2.576
    else:
        return 0
    return get_standard_error(std, sample_size) * multiplier

def to_int_if_int(value):
    try:
        # Try to convert the value to an integer
        int_value = int(value)
        return int_value
    except ValueError:
        # If it's not possible to convert to an integer, return the original value
        return value

def print_elapsed_time(text, start_time):
    if text != "":
        time_elapsed = time.time() - start_time
        print(f'{text} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    return(time.time())
