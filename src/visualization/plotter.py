import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_values(df, colnames, type = "mean", pre = ""):
    re = []
    for colname in colnames:
        re.append(df[pre+colname][type])
    return re

colnames = ["x1", "x2", "x3", "x4", "logits"]
colnames = ["pool1", "pool2", "pool3", "pool4", "logits"]

#data = pd.read_csv('csvs/new_exp.csv')
data = pd.read_csv('../../csvs/transfer_all.csv')
data = pd.read_csv('../../csvs/comp_init_pretrained.csv')
data = pd.read_csv('../../csvs/pretraining_random.csv')
data = pd.read_csv('../../csvs/training_size.csv')
#data = pd.read_csv('csvs/comp_init_pretrained.csv')


group = "dataset"
group = "d"
group = "pretrain_size"
pre = False
figsize = (4,4)
title = "Similarity of (cross-)domain task"
title = "Similarity of (partially) random CIFAR-10"
title = "Similarity of CIFAR-10 (partially random)"
title = "Similarity with changing size of training data"
datasets = "all"
#datasets = ["SVHN", "SVHN shuffle_degree: 10", "SVHN shuffle_degree: 5", "SVHN shuffle_degree: 2"]
#datasets = ["CIFAR-10 (d=0)", "CIFAR-10 (d=9)", "CIFAR-10 (d=1)", "CIFAR-10 (d=4)"]
#datasets = ["SVHN (d=0)", "SVHN (d=9)", "SVHN (d=1)", "SVHN (d=4)"]
#datasets = ["CIFAR-10 (d=0)", "CIFAR-10 (d=9)", "SVHN (d=0)", "CIFAR-10 (shifted)"]
#datasets = ["CIFAR-10 (random)"]
test = "TEST"
#test = "TRANSFER_TEST"


X_s = []
for colname in colnames:
    if pre:
        x = data.groupby([group]).agg(
            {f"TRANSFER_TEST/{colname}": ['mean', 'sem'], f"TRANSFER_TEST/pre_{colname}": ['mean', 'sem']}).rename(
            columns={f"TRANSFER_TEST/{colname}": colname}).rename(
            columns={f"TRANSFER_TEST/pre_{colname}": f"pre_{colname}"})
    else:
        x = data.groupby([group]).agg({f"{test}/{colname}": ['mean', 'sem']}).rename(columns={f"{test}/{colname}": colname})

    X_s.append(x)


together = pd.concat(X_s, axis=1, join="inner")
together.sort_values(by = group)

fig = plt.figure(figsize=figsize)
if pre:
    plt.title(datasets[0])
else:
    plt.title(title)
plt.ylabel("CKA")
plt.xlabel("Layer")

colors = plt.cm.jet(np.linspace(0.7, 0.95, len(together)))

j = 0
for i, row in together.iterrows():

    if datasets != "all":
        if i not in datasets:
            continue
    means = get_values(row, colnames, "mean")
    errors = get_values(row, colnames, "sem")
    label = str(i)

    if pre:
        plt.errorbar(colnames, means, yerr=errors, label = "Similarity to pretrained weights", color = colors[j])
        means = get_values(row, colnames, "mean", pre = "pre_")
        errors = get_values(row, colnames, "sem", pre = "pre_")
        plt.errorbar(colnames, means, yerr=errors, label = "Similarity to initialized weights", color = colors[j+3])
    else:
        plt.errorbar(colnames, means, yerr=errors, label=label, color=colors[j])
    j= j + 1
    plt.legend(loc = 'lower left')

plt.savefig(f"img/{title}.pdf")
plt.show()
