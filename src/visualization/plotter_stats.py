import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb


data = pd.read_csv('../../csvs/stats.csv')
#data = pd.read_csv('../../csvs/first_exp.csv')

wandb.login()

# Specify your WandB project and run name
project_name = "cka_analysis"
# Specify your project name

# Initialize the W&B API
api = wandb.Api()

# Get all runs for the specified project
runs = api.runs(f"thomasgo/{project_name}")  # Replace "your-username" with your W&B username

# Iterate through the runs and do something with each run
for run in runs:
    if run.config["dataset"] == "cifar10" and  run.config["model"] == "conv4":
        print(f"Run ID: {run.id}, Run Name: {run.name}, Run State: {run.state}")

    # Add more details or actions as needed



figsize = (8,3.8)
title = "Loss"
title = "Test performance"
#title = "Similarity of different layers in standard training"

print(data)
print(data.head)

fig = plt.figure(figsize=figsize)

plt.ylabel("Loss")
plt.ylabel("Accuracy (%)")
#plt.ylabel("CKA")
plt.xlabel("Epoch")



#
plt.plot(data["degree_of_randomness: 1 - TEST/accuracy"], label = "d=0, normal")
plt.plot(data["degree_of_randomness: 2 - TEST/accuracy"], label = "d=1, ")
plt.plot(data["degree_of_randomness: 10 - TEST/accuracy"], label = "d=9, completely random")
#plt.plot(data["dataset: cifar10 shuffle_degree: cifar10 - TEST/x1"], label = "pool1")
#plt.plot(data["dataset: cifar10 shuffle_degree: cifar10 - TEST/x2"], label = "pool2")
#plt.plot(data["dataset: cifar10 shuffle_degree: cifar10 - TEST/x3"], label = "pool3")
#plt.plot(data["dataset: cifar10 shuffle_degree: cifar10 - TEST/x4"], label = "pool4")
#plt.plot(data["dataset: cifar10 shuffle_degree: cifar10 - TEST/logits"], label = "logits")

plt.title(title)
#plt.legend(loc="upper right")
plt.legend(loc="lower right")
plt.ylim(-0.1,100.1)

plt.savefig(f"../../img/{title}.pdf")
plt.show()
