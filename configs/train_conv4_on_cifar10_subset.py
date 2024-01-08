config_list = []

for size in [64, 256, 1024, 4096]:
    config = {
        "batch_size": 32,
        "keep_prob": 0.0,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "epochs": 150,
        "dataset": "cifar10",
        "model": "conv4",
        "seeds": [1, 2, 3, 4, 5],
        "degree_of_randomness": 0,
        "pre_trained_size": size
    }
    config_list.append(config)

# Print or use config_list as needed
print(config_list)
