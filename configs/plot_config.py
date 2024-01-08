def create_plot_list(model, columns_of_interest, columns_of_interest_2, attributes):

    configs_list = [
        {
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
                'figsize': (8, 3.8),
                'title': 'Similarity of different layers in standard training',
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
                'figsize': (8, 3.8),
                'title': '',
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
                'title': 'Similarity of cifar10 (partially random)',
                'figsize': (4, 4),
                'xlabel': 'Layer',
                'ylabel': 'CKA',
                'model': model
            }
        },
        {
            'filters': {
                'pre_trained_dataset': 'cifar10',
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
                'title': 'Similarity of (partially) random cifar10',
                'figsize': (4, 4),
                'xlabel': 'Layer',
                'ylabel': 'CKA',
                'model': model
            }
        },
        {
            'filters': {
                'pre_trained_dataset': 'cifar10',
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
                'title': 'Similarity of (partially) random SVHN',
                'figsize': (4, 4),
                'xlabel': 'Layer',
                'ylabel': 'CKA',
                'model': model
            }
        },
        {
            'filters': {
                'pre_trained_dataset': 'cifar10',
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
                'title': 'Similarity of (cross-)domain task',
                'figsize': (4, 4),
                'xlabel': 'Layer',
                'ylabel': 'CKA',
                'model': model
            }
        }
    ]

    for dataset, title in zip(["cifar10", "SVHN", "cifar10_shifted", "cifar10 shuffle_degree: 9"],
                              ["cifar10", "SVHN", "cifar10_shifted", "cifar10 shuffle_degree: 9"]):

        config = {
            'filters': {
                'pre_trained_dataset': 'cifar10',
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

    return(configs_list)

conv4_configs_list = create_plot_list(
    'conv4',
    ['CKAS/pool1', 'CKAS/pool2', 'CKAS/pool3', 'CKAS/pool4', 'CKAS/logits'],
    ['CKAS/pre_initialized_pool1', 'CKAS/pre_initialized_pool2', 'CKAS/pre_initialized_pool3', 'CKAS/pre_initialized_pool4', 'CKAS/pre_initialized_logits'],
    ['pool1', 'pool2', 'pool3', 'pool4', 'logits']
)

resnet18_configs_list = create_plot_list(
    'resnet18',
    ['CKAS/pool1', 'CKAS/pool2', 'CKAS/pool3', 'CKAS/pool4', 'CKAS/pool5', 'CKAS/logits'],
    ['CKAS/pre_initialized_pool1', 'CKAS/pre_initialized_pool2', 'CKAS/pre_initialized_pool3', 'CKAS/pre_initialized_pool4', 'CKAS/pre_initialized_pool5', 'CKAS/pre_initialized_logits'],
    ['pool', 'block1', 'block2', 'block3', 'block4', 'logits']
)
