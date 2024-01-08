config_list = []

for dataset in ["cifar10", "SVHN", "cifar10_shifted"]:
    for degree in [0, 1, 2, 4, 9]:
        # for shifted we do not iterated the degree
        if dataset == "cifar10_shifted" and degree > 0:
            continue
        config = {
            "batch_size": 32,
            "keep_prob": 0.0,
            "learning_rate": 0.001,
            "momentum": 0.9,
            "epochs": 25,
            "dataset": dataset,
            "model": "conv4",
            "seeds": [1, 2, 3, 4, 5],
            "finetuning_size": 5000,
            "finetuning": True,
            "pre_trained_dataset": "cifar10",
            "degree_of_randomness": degree
        }
        config_list.append(config)

# Print or use config_list as needed
print(config_list)
