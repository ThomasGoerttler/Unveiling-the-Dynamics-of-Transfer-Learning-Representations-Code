from src.utils.plotting_utils import *
from configs.plot_config import conv4_configs_list, resnet18_configs_list

old_filters = dict()
old_history = ""

for config in resnet18_configs_list:#conv4_configs_list+resnet18_configs_list:
    # do not  load data if same as before
    if old_filters != config['filters'] or old_history != config['history']:
        data = load_data(filters=config['filters'], history=config['history'])
    grouped_df = aggregate(data, groupby=config['groupby'], columns_of_interest=config['columns_of_interest'], new_column_name=config['attributes'])

    if 'columns_of_interest_2' in config.keys():
        grouped_df_pre = aggregate(data, groupby=config["groupby"], columns_of_interest=config['columns_of_interest_2'],
                                   new_column_name=config['attributes'])

        grouped_df = pd.concat([grouped_df, grouped_df_pre])
        grouped_df[config["vertical_label"]] = config["vertical_attributes"]
        grouped_df.reset_index(drop=True, inplace=True)

    if config['type'] == "line":
        plot_line(grouped_df, config['attributes'], config['information'] )
    elif config['type'] == "errorbar":
        plot_error_bar(grouped_df, config['attributes'], config['information'] )

    old_filters = config['filters']
    old_history = config['history']
