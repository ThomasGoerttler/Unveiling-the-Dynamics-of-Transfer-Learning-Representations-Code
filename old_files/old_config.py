for dataset, title in zip(["cifar10", "SVHN", "cifar10_shifted", "cifar10 shuffle_degree: 9"],
                          ["cifar10", "SVHN", "cifar10_shifted", "cifar10 shuffle_degree: 9"]):
    config = {
        'filters': {
            'pre_trained_dataset': pre_trained_dataset,
            'dataset': dataset,
            'model': model,
            'finetuning': True
        },
        'history': False,
        'columns_of_interest': columns_of_interest,
        'columns_of_interest_2': columns_of_interest_2,
        'attributes': attributes,
        'groupby': ['finetuning'],
        'vertical_label': 'initialized',
        'vertical_attributes': ["pre_trained", "pre_initialized"],
        'type': 'errorbar',
        'information': {
            'label_attribute': 'initialized',
            'title': title,
            'figsize': (4, 4),
            'xlabel': 'Layer',
            'ylabel': 'CKA',
            'model': model
        }
    }
    configs_list.append(config)
