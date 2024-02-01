def create_plot_list(model, columns_of_interest, columns_of_interest_2, attributes, fine_tune_only = False, pre_trained_dataset = "cifar10"):

    if fine_tune_only:
        configs_list = []
    else:
        configs_list = [{
                'filters': {
                    'dataset': 'cifar10...',
                    'model': model,
                    'finetuning': False,
                    'pre_trained_size': 50000
                },
                'history': True,
                'columns_of_interest': columns_of_interest,
                'attributes': attributes,
                'groupby': ['step'],
                'type': 'line',
                'information': {
                    'error': 'confidence_interval',
                    'same_plot': True,
                    'label': '',
                    'figsize': (4, 4),
                    'title': 'Similarity of different layers in standard training',
                    'subtitle': f'{model} trained on cifar10',
                    'xlabel': 'Epoch',
                    'ylabel': 'CKA',
                    'model': model
                }
            },
            {
                'filters': {
                    'dataset': 'cifar10...',
                    'model': model,
                    'finetuning': False,
                    'pre_trained_size': 50000
                },
                'history': True,
                'columns_of_interest': ['TRAIN/accuracy', 'TRAIN/loss', 'TEST/accuracy'],
                'attributes': ['Train accuracy', 'Train loss', 'Test accuracy'],
                'groupby': ['step', 'degree_of_randomness'],
                'type': 'line',
                'information': {
                    'error': 'confidence_interval',
                    'same_plot': False,
                    'label': 'degree_of_randomness',
                    'figsize': (2.5, 4),
                    'title': '',
                    'subtitle': f'{model} trained on cifar10',
                    'xlabel': 'Epoch',
                    'ylabel': '',
                    'model': model
                }
            },
            {
                'filters': {
                    'dataset': 'cifar10',
                    'model': model,
                    'finetuning': False
                },
                'history': False,
                'columns_of_interest': columns_of_interest,
                'attributes': attributes,
                'groupby': ['pre_trained_size'],
                'type': 'errorbar',
                'information': {
                    'label_attribute': 'pre_trained_size',
                    'title': 'Similarity with changing size of training data',
                    'subtitle': f'{model} trained on cifar10',
                    'figsize': (4, 4),
                    'xlabel': 'Layer',
                    'ylabel': 'CKA',
                    'model': model
                }
            },
            {
                'filters': {
                    'dataset': 'cifar10...',
                    'model': model,
                    'finetuning': False,
                    'pre_trained_size': 50000
                },
                'history': False,
                'columns_of_interest': columns_of_interest,
                'attributes': attributes,
                'groupby': ['degree_of_randomness'],
                'type': 'errorbar',
                'information': {
                    'label_attribute': 'degree_of_randomness',
                    'title': 'Similarity of (partially) random cifar10',
                    'subtitle': f'{model} trained on cifar10',
                    'figsize': (4, 4),
                    'xlabel': 'Layer',
                    'ylabel': 'CKA',
                    'model': model
                }
            }
        ]
    configs_list += [
        {
            'filters': {
                'pre_trained_dataset': pre_trained_dataset,
                'dataset': 'cifar10...',
                '!dataset': 'cifar10_shifted',
                'model': model,
                'finetuning': True
            },
            'history': False,
            'columns_of_interest': columns_of_interest,
            'attributes': attributes,
            'groupby': ['degree_of_randomness'],
            'type': 'errorbar',
            'information': {
                'label_attribute': 'degree_of_randomness',
                'title': f'Similarity of (partially) random cifar10',
                'subtitle': f'{model} pretrained on {pre_trained_dataset}',
                'figsize': (4, 4),
                'xlabel': 'Layer',
                'ylabel': 'CKA',
                'model': model
            }
        },
        {
            'filters': {
                'pre_trained_dataset': pre_trained_dataset,
                'dataset': 'SVHN...',
                'model': model,
                'finetuning': True
            },
            'history': False,
            'columns_of_interest': columns_of_interest,
            'attributes': attributes,
            'groupby': ['degree_of_randomness'],
            'type': 'errorbar',
            'information': {
                'label_attribute': 'degree_of_randomness',
                'title': f'Similarity of (partially) random SVHN',
                'subtitle': f'{model} pretrained on {pre_trained_dataset}',
                'figsize': (4, 4),
                'xlabel': 'Layer',
                'ylabel': 'CKA',
                'model': model
            }
        },
        {
            'filters': {
                'pre_trained_dataset': pre_trained_dataset,
                'dataset': ["cifar10", "cifar10 shuffle_degree: 9", "cifar10_shifted", "SVHN"],
                'model': model,
                'finetuning': True
            },
            'history': False,
            'columns_of_interest': columns_of_interest,
            'attributes': attributes,
            'groupby': ['dataset'],
            'type': 'errorbar',
            'information': {
                'label_attribute': 'dataset',
                'title': f'Similarity of (cross-)domain task',
                'subtitle': f'{model} pretrained on {pre_trained_dataset}',
                'figsize': (4, 4),
                'xlabel': 'Layer',
                'ylabel': 'CKA',
                'model': model
            }
        }
    ]

    if model in ["resnet18", "vgg16"]:
        configs_list += [{
            'filters': {
                'pre_trained_dataset': pre_trained_dataset,
                'dataset': 'imagenet...',
                'model': model,
                'finetuning': True
            },
            'history': False,
            'columns_of_interest': columns_of_interest,
            'attributes': attributes,
            'groupby': ['degree_of_randomness'],
            'type': 'errorbar',
            'information': {
                'label_attribute': 'degree_of_randomness',
                'title': f'Similarity of (partially) random imagenet',
                'subtitle': f'{model} pretrained on {pre_trained_dataset}',
                'figsize': (4, 4),
                'xlabel': 'Layer',
                'ylabel': 'CKA',
                'model': model
            }
        }]

    if pre_trained_dataset == "imagenet":
        datasets = ["cifar10", "cifar10_shifted", "cifar10 shuffle_degree: 9", "SVHN", "SVHN shuffle_degree: 9"]
        names = ["cifar10", "cifar10 shifted", "cifar10 random", "SVHN", "SVHN random"]

    elif pre_trained_dataset == "cifar10":
        if model in ["resnet18", "vgg16"]:
            datasets = ["cifar10", "cifar10_shifted", "cifar10 shuffle_degree: 9", "SVHN", "SVHN shuffle_degree: 9"]
            names = ["cifar10", "cifar10 shifted", "cifar10 random", "SVHN", "SVHN random"]
        else:
            datasets = ["cifar10", "cifar10_shifted", "cifar10 shuffle_degree: 9", "SVHN", "SVHN shuffle_degree: 9"]
            names = ["cifar10", "cifar10 shifted", "cifar10 random", "SVHN", "SVHN random"]

    config = {
        'filters': [{
            'pre_trained_dataset': pre_trained_dataset,
            'dataset': dataset,
            'model': model,
            'finetuning': True
        } for dataset in datasets],
        'history': False,
        'columns_of_interest': columns_of_interest,
        'columns_of_interest_2': columns_of_interest_2,
        'attributes': attributes,
        'groupby': ['finetuning'],
        'vertical_label': 'initialized',
        'vertical_attributes': ["pre-trained", "pre-initialized"],
        'type': 'multiple_errorbar',
        'information': {
            'label_attribute': 'initialized',
            'title': 'Comparision of fine-tuned activations to pre-trained and pre-initialized',
            'figsize': (8, 3.8),
            'xlabel': 'Finetuned dataset',
            'subtitle': f'{model} pretrained on {pre_trained_dataset}',
            'ylabel': 'CKA',
            'xticks': names,
            'model': model
        }
    }
    configs_list = []
    configs_list.append(config)
    if model in ["resnet18", "vgg16"]:
        datasets = ["cifar10", "cifar10 shuffle_degree: 9"]
        names_datasets = ["cifar10", "cifar10 random"]
        pre_trained_datasets = ["cifar10", "imagenet"]
        names = [f"f. on {dataset}\np.-tr. on {pre_trained_dataset}" for dataset in names_datasets for pre_trained_dataset in pre_trained_datasets]
        config = {
            'filters': [{
                'pre_trained_dataset': pre_trained_dataset,
                'dataset': dataset,
                'model': model,
                'finetuning': True
            }for pre_trained_dataset in pre_trained_datasets for dataset in datasets ],
            'history': False,
            'columns_of_interest': columns_of_interest,
            'columns_of_interest_2': columns_of_interest_2,
            'attributes': attributes,
            'groupby': ['finetuning'],
            'vertical_label': 'initialized',
            'vertical_attributes': ["pre-trained", "pre-initialized"],
            'type': 'multiple_errorbar',
            'information': {
                'label_attribute': 'initialized',
                'title': 'Comparision of fine-tuned activations to pre-trained and pre-initialized',
                'figsize': (8, 3.8),
                'xlabel': 'Finetuned dataset',
                'subtitle': f'{model} pretrained on {pre_trained_datasets[0]} and {pre_trained_datasets[1]}',
                'ylabel': 'CKA',
                'xticks': names,
                'model': model
            }
        }
        configs_list.append(config)

    return(configs_list)

conv4_configs_list = create_plot_list(
    'conv4',
    [f"CKAS/layer{i + 1}" for i in range(5)],
    [f"CKAS/pre_initialized_layer{i + 1}" for i in range(5)],
    ['pool1', 'pool2', 'pool3', 'pool4', 'logits'],
    fine_tune_only = False,
    pre_trained_dataset="cifar10"
)

resnet18_configs_list = create_plot_list(
    'resnet18',
    [f"CKAS/layer{i + 1}" for i in range(6)],
    [f"CKAS/pre_initialized_layer{i + 1}" for i in range(6)],
    ['pool', 'block1', 'block2', 'block3', 'block4', 'logits'],
    fine_tune_only = False,
    pre_trained_dataset="cifar10"
)


vgg16_configs_list = create_plot_list(
    'vgg16',
    [f"CKAS/layer{i+1}" for i in list(range(5)) + [7]],
    [f"CKAS/pre_initialized_layer{i+1}" for i in list(range(5)) + [7]],
    ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'logits'],
    fine_tune_only = False,
    pre_trained_dataset="cifar10"
)

resnet18_configs_list_imagenet = create_plot_list(
    'resnet18',
    [f"CKAS/layer{i + 1}" for i in range(6)],
    [f"CKAS/pre_initialized_layer{i + 1}" for i in range(6)],
    ['pool', 'block1', 'block2', 'block3', 'block4', 'logits'],
    fine_tune_only = True,
    pre_trained_dataset="imagenet"
)

vgg16_configs_list_imagenet = create_plot_list(
    'vgg16',
    [f"CKAS/layer{i+1}" for i in list(range(5)) + [7]],
    [f"CKAS/pre_initialized_layer{i+1}" for i in list(range(5)) + [7]],
    ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'logits'],
    fine_tune_only = True,
    pre_trained_dataset="imagenet"
)
