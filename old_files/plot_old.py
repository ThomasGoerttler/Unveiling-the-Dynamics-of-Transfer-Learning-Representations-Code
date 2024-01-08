from src.utils.plotting_utils import *

for model in ['resnet18']:
    
    data = load_data(filters={'dataset': 'cifar10...', 'model': model, "finetuning": False, "pre_trained_size": 50000}, history=True)
    columns_of_interest = ['CKAS/pool1', 'CKAS/pool2', 'CKAS/pool3', 'CKAS/pool4', 'CKAS/logits']
    attributes = ['pool1', 'pool2', 'pool3', 'pool4', 'logits']
    grouped_df = aggregate(data, groupby = ['step'], columns_of_interest = columns_of_interest, new_column_name = attributes)
    plot_line_old(grouped_df, attributes, same_plot=True, title = "Similarity of different layers in standard training", xlabel="Epoch", ylabel="CKA", model=model)


    columns_of_interest = ['TRAIN/accuracy', 'TRAIN/loss', 'TEST/accuracy']
    attributes = ['Train accuracy', 'Train loss', 'Test accuracy']
    grouped_df = aggregate(data, groupby = ['step', 'degree_of_randomness'], columns_of_interest = columns_of_interest, new_column_name = attributes)
    plot_line_old(grouped_df, attributes, same_plot=False, label = "degree_of_randomness", xlabel="Epoch", model=model)
    
    data = load_data(filters={ "dataset": "cifar10", 'model': model, "finetuning": False}, history=False)
    columns_of_interest = ['CKAS/pool1', 'CKAS/pool2', 'CKAS/pool3', 'CKAS/pool4', 'CKAS/logits']
    attributes = ['pool1', 'pool2', 'pool3', 'pool4', 'logits']
    grouped_df = aggregate(data, groupby = ["pre_trained_size"], columns_of_interest = columns_of_interest, new_column_name = attributes)
    plot_error_bar_old(grouped_df, attributes, "pre_trained_size", title ="Similarity with changing size of training data", xlabel = "Layer", ylabel = "CKA", model = model)




    data = load_data(filters={ "dataset": "cifar10...", 'model': model, "finetuning": False, "pre_trained_size": 50000}, history=False)
    columns_of_interest = ['CKAS/pool1', 'CKAS/pool2', 'CKAS/pool3', 'CKAS/pool4', 'CKAS/logits']
    attributes = ['pool1', 'pool2', 'pool3', 'pool4', 'logits']
    grouped_df = aggregate(data, groupby = ["degree_of_randomness"], columns_of_interest = columns_of_interest, new_column_name = attributes)
    plot_error_bar(grouped_df, attributes, "degree_of_randomness", title ="Similarity of cifar10 (partially random)", xlabel = "Layer", ylabel = "CKA", model = model)



    data = load_data(filters={'pre_trained_dataset': 'cifar10', 'model': model, "finetuning": True, "dataset": "cifar10...", "!dataset": "cifar10_shifted"}, history=False)
    columns_of_interest = ['CKAS/pool1', 'CKAS/pool2', 'CKAS/pool3', 'CKAS/pool4', 'CKAS/logits']
    attributes = ['pool1', 'pool2', 'pool3', 'pool4', 'logits']
    grouped_df = aggregate(data, groupby = ["degree_of_randomness"], columns_of_interest = columns_of_interest, new_column_name = attributes)
    plot_error_bar(grouped_df, attributes, "degree_of_randomness", title ="Similarity of (partially) random cifar10", xlabel = "Layer", ylabel = "CKA", model=model)

    data = load_data(filters={'pre_trained_dataset': 'cifar10', 'model': model, "finetuning": True, "dataset": "SVHN..."}, history=False)
    columns_of_interest = ['CKAS/pool1', 'CKAS/pool2', 'CKAS/pool3', 'CKAS/pool4', 'CKAS/logits']
    attributes = ['pool1', 'pool2', 'pool3', 'pool4', 'logits']
    grouped_df = aggregate(data, groupby = ["degree_of_randomness"], columns_of_interest = columns_of_interest, new_column_name = attributes)
    plot_error_bar(grouped_df, attributes, "degree_of_randomness", title ="Similarity of (partially) random SVHN", xlabel = "Layer", ylabel = "CKA", model=model)

    data = load_data(filters={'pre_trained_dataset': 'cifar10', 'model': model, "finetuning": True, "dataset": ["cifar10", "cifar10 shuffle_degree: 9", "cifar10_shifted", "SVHN"]}, history=False)
    columns_of_interest = ['CKAS/pool1', 'CKAS/pool2', 'CKAS/pool3', 'CKAS/pool4', 'CKAS/logits']
    attributes = ['pool1', 'pool2', 'pool3', 'pool4', 'logits']
    grouped_df = aggregate(data, groupby = ["dataset"], columns_of_interest = columns_of_interest, new_column_name = attributes)
    plot_error_bar(grouped_df, attributes, "dataset", title ="Similarity of (cross-)domain task", xlabel = "Layer", ylabel = "CKA", model=model)
    
    for dataset, title in zip(["cifar10", "SVHN", "cifar10_shifted", "cifar10 shuffle_degree: 9"], ["cifar10", "SVHN", "cifar10_shifted", "cifar10 shuffle_degree: 9"]):
        data = load_data(filters={'pre_trained_dataset': 'cifar10', 'model': model, "finetuning": True, "dataset": dataset}, history=False)
        columns_of_interest = ['CKAS/pool1', 'CKAS/pool2', 'CKAS/pool3', 'CKAS/pool4', 'CKAS/logits']
        attributes = ['pool1', 'pool2', 'pool3', 'pool4', 'logits']
        grouped_df = aggregate(data, groupby=["finetuning"], columns_of_interest=columns_of_interest, new_column_name=attributes)
        columns_of_interest = ['CKAS/pre_initialized_pool1', 'CKAS/pre_initialized_pool2', 'CKAS/pre_initialized_pool3', 'CKAS/pre_initialized_pool4', 'CKAS/pre_initialized_logits']
        grouped_df_pre = aggregate(data, groupby=["finetuning"], columns_of_interest=columns_of_interest, new_column_name=attributes)

        # Concatenate them vertically
        concatenated_df = pd.concat([grouped_df, grouped_df_pre])
        concatenated_df["initialized"] = ["pre_trained", "pre_initialized"]
        concatenated_df.reset_index(drop=True, inplace=True)
        plot_error_bar(concatenated_df, attributes, "initialized", title=title, xlabel = "Layer", ylabel = "CKA", model=model)
