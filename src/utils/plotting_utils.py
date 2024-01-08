import pandas as pd
import matplotlib.pyplot as plt
import wandb
import numpy as np

from src.utils.utils import get_standard_error, get_confidence_interval, to_int_if_int

def check_conditions(filters, run):
    for key, value in filters.items():

        do_not = False
        if key.startswith("!"):
            do_not = True
            key = key[1:]

        if key not in run.config:
            return False

        config_value = run.config[key]

        if isinstance(value, list):
            if config_value not in value and not do_not:
                return False
        elif isinstance(value, str):
            if value.endswith("..."):
                if not config_value.startswith(value[:-3]):
                    return False
            elif do_not:
                if config_value == value:
                    return False
            else:
                if config_value != value:
                    return False
        else:
            if config_value != value:
                return False
    return True

def load_data(filters={}, history = True):

    wandb.login()
    # Initialize the W&B API
    api = wandb.Api()
    # Get all runs for the specified project
    runs = api.runs(f"thomasgo/cka_analysis")  # Replace "thomasgo" with your W&B username
    # Create an empty list to store individual DataFrames
    data_frames = []
    for run in runs:
        if check_conditions(filters, run):
            full_values = run.history()

            if history:
                # Create a DataFrame with config repeated for each row in full_values
                config_df = pd.DataFrame([run.config] * len(full_values), columns=run.config.keys())
                # Concatenate full_values and config_df along the columns
                combined_df = pd.concat([full_values, config_df], axis=1)
                data_frames.append(combined_df)
            else:
                last_row = full_values.tail(1)

                # Append wandb.config to the last row
                last_row_with_config = last_row.assign(**run.config)

                data_frames.append(last_row_with_config)


    result_df = pd.concat(data_frames, ignore_index=True)

    result_df = result_df.rename(columns={'_step': 'step'})
    return result_df

def aggregate(data, groupby, columns_of_interest, new_column_name = []):

    grouped_df = data.groupby(groupby)[columns_of_interest].agg(['mean', 'std', 'count'], as_index=False)
    grouped_df = grouped_df.reset_index()

    # Rename columns
    if new_column_name != []:
        if len(columns_of_interest) == len(new_column_name):
            new_column_names = {f'{col} mean': new_col for col, new_col in zip(columns_of_interest, new_column_name)}
            new_column_names.update({f'{col} std': f'{new_col} std' for col, new_col in zip(columns_of_interest, new_column_name)})
            new_column_names.update({f'{col} count': f'{new_col} count' for col, new_col in zip(columns_of_interest, new_column_name)})

            grouped_df.columns = [f'{col} {stat}' for col, stat in grouped_df.columns]
            grouped_df = grouped_df.rename(columns=new_column_names)
            for element in groupby:
                grouped_df[element] = grouped_df[element+" "]
        else:
            print("Columns cannot be renamed since length is not the same")
    return(grouped_df)


def plot_error_bar(data, attributes, config):
    fig = plt.figure(figsize=config['figsize'])
    colors = plt.cm.jet(np.linspace(0.7, 0.95, len(data)))

    for i, row in data.iterrows():

        means = [row[attribute] for attribute in attributes]
        errors = [row[attribute + " std"] / np.sqrt(row[attribute + " count"]) for attribute in attributes]
        label = to_int_if_int(row[config['label_attribute']])
        plt.errorbar(attributes, means, yerr=errors, label=label, color=colors[i])

        plt.legend(loc='lower left')

    if config['xlabel'] != "":
        plt.xlabel(config['xlabel'])
    if config['ylabel'] != "":
        plt.ylabel(config['ylabel'])

    plt.title(config['title'])
    plt.savefig(f"../../img/{config['model']}_{config['title']}.pdf")
    plt.show()


def plot_error_bar_old(data, attributes, label_attribute, title, xlabel = "", ylabel = "", model="", figsize = (4, 4)):

    fig = plt.figure(figsize=figsize)
    colors = plt.cm.jet(np.linspace(0.7, 0.95, len(data)))

    for i, row in data.iterrows():

        means = [row[attribute] for attribute in attributes]
        errors = [row[attribute + " std"] / np.sqrt(row[attribute + " count"]) for attribute in attributes]
        label = to_int_if_int(row[label_attribute])
        plt.errorbar(attributes, means, yerr=errors, label=label, color=colors[i])

        plt.legend(loc='lower left')

    if xlabel != "":
        plt.xlabel(xlabel)
    if ylabel != "":
        plt.ylabel(ylabel)

    plt.title(title)
    plt.savefig(f"../../img/{model}_{title}.pdf")
    plt.show()


def plot_line(data, attributes, config):

    if config['same_plot']:
        plt.figure(figsize=config['figsize'])
        plt.xlabel(config['xlabel'])
        plt.ylabel(config['ylabel'])
        plt.title(config['title'])

    for attribute in attributes:

        if not config['same_plot']:
            plt.figure(figsize=config['figsize'])
            plt.xlabel(config['xlabel'])
            plt.ylabel(config['ylabel'])
            titel = attribute
            plt.title(titel)

        if config['label'] != "":
            label_values = data[config['label']].unique()
            datasets = [data[data[config['label']] == value] for value in label_values]
        else:
            datasets = [data]
            label_values = [attribute]

        for dataset, label_value in zip(datasets, label_values):
            plt.plot(dataset[attribute], label = label_value)

            lower_bound = None
            upper_bound = None

            if config['error'] == "std":
                lower_bound = dataset[attribute] - dataset[f'{attribute} std']
                upper_bound = dataset[attribute] + dataset[f'{attribute} std']
            elif config['error'] == "ste":
                se = get_standard_error(dataset[f'{attribute} std'], dataset[f'{attribute} count'])
                lower_bound = dataset[attribute] - se
                upper_bound = dataset[attribute] + se
            elif config['error'] == "confidence_interval":
                ci = get_confidence_interval(dataset[f'{attribute} std'], dataset[f'{attribute} count'])
                lower_bound = dataset[attribute] - ci
                upper_bound = dataset[attribute] + ci

            if lower_bound is not None and upper_bound is not None:
                plt.fill_between(dataset.index, lower_bound, upper_bound, alpha=0.3)#, label=f'{attribute} {error}')

        plt.ylim(bottom=0)
        plt.legend()

        if not config['same_plot']:
            plt.savefig(f"../../img/{config['model']}_{attribute}.pdf")
            plt.show()

    if config['same_plot']:
        plt.savefig(f"../../img/{config['model']}_{config['title']}.pdf")
        plt.show()

def plot_line_old(data, attributes, error =  "confidence_interval", same_plot = False, label = "", figsize = (8, 3.8), title = "tba", xlabel = "tba", ylabel = "tba", model=""):

    if same_plot:
        plt.figure(figsize=figsize)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

    for attribute in attributes:

        if not same_plot:
            plt.figure(figsize=figsize)
            plt.xlabel(xlabel)
            plt.ylabel(attribute)
            titel = attribute
            plt.title(titel)

        if label != "":
            label_values = data[label].unique()
            datasets = [data[data[label] == value] for value in label_values]
        else:
            datasets = [data]
            label_values = [attribute]

        for dataset, label_value in zip(datasets, label_values):
            plt.plot(dataset[attribute], label = label_value)

            lower_bound = None
            upper_bound = None

            if error == "std":
                lower_bound = dataset[attribute] - dataset[f'{attribute} std']
                upper_bound = dataset[attribute] + dataset[f'{attribute} std']
            elif error == "ste":
                se = get_standard_error(dataset[f'{attribute} std'], dataset[f'{attribute} count'])
                lower_bound = dataset[attribute] - se
                upper_bound = dataset[attribute] + se
            elif error == "confidence_interval":
                ci = get_confidence_interval(dataset[f'{attribute} std'], dataset[f'{attribute} count'])
                lower_bound = dataset[attribute] - ci
                upper_bound = dataset[attribute] + ci

            if lower_bound is not None and upper_bound is not None:
                plt.fill_between(dataset.index, lower_bound, upper_bound, alpha=0.3)#, label=f'{attribute} {error}')

        plt.ylim(bottom=0)
        plt.legend()

        if not same_plot:
            plt.savefig(f"../../img/{model}_{title}.pdf")
            plt.show()

    if same_plot:
        plt.savefig(f"../../img/{model}_{title}.pdf")
        plt.show()
