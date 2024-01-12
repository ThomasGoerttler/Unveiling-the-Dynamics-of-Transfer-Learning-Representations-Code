import subprocess
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import sys

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.communicate()

def generate_commands(command_configs, plot = False):
    # Generate commands using a list comprehension
    commands = []

    for command_config in commands_configs:
        command_parts = []

        for key, value in command_config.items():
            if value is True:
                command_parts.append(f'--{key}')
            elif value is False:
                pass
            elif isinstance(value, list):
                command_parts.append(f'--{key} {" ".join(map(str, value))}')
            else:
                command_parts.append(f'--{key} {value}')

        if plot:
            command = f"python plot.py {' '.join(command_parts)}"
        else:
            command = f"python run.py {' '.join(command_parts)}"
        commands.append(command)

    return commands

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file_name>")
        sys.exit(1)

    config_file_name = sys.argv[1]

    # Dynamically import the config module
    config_module_path = f"{config_file_name}"  # Adjust the path based on your project structure
    spec = importlib.util.spec_from_file_location('configs', 'configs/' + config_module_path + '.py')
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Access the loaded configuration
    commands_configs = config_module.config_list

    # Generate commands using a list comprehension
    commands = generate_commands(commands_configs)

    # Use ThreadPoolExecutor to run commands in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(run_command, commands)
