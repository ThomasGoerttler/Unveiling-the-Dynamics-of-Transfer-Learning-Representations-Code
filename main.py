# -*- coding: utf-8 -*-
"""
based on https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
"""
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.optim as optim
from src.utils.cka import linear_CKA

import wandb

from src.models.cnn import Cnn
from src.utils.utils import resize


def test(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = []
        activations = []

        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs and activations by running images through the network
            outs, acts = net(inputs)

            outputs.append(torch.flatten(outs, 1).cpu().detach().numpy())
            activations.append([torch.flatten(activation, 1).cpu().detach().numpy() for activation in acts])

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        outputs = resize(outputs)
        activations = [resize(activation) for activation in zip(*activations)]

    return 100 * correct / total, outputs, activations


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def analyze_activation(base_logits, logits, base_activations, activations, maxi=200):
    cka_activations = []
    for base_activation, activation in zip(base_activations, activations):
        cka = linear_CKA(base_activation[:maxi], activation[:maxi])
        cka_activations.append(cka)
    cka_logits = linear_CKA(base_logits[:maxi], logits[:maxi])
    return cka_logits, cka_activations


def store_ckas_to_wandb(wandb, ckas, base_name='TEST/pool', step=0):
    for i, cka in enumerate(ckas):
        wandb.log({
            base_name + str(i + 1): cka,
        }, step=step)

if __name__=='__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32
    keep_prob = 0.0
    learning_rate = 0.001
    momentum = 0.9
    epochs = 25#150
    dataset = "cifar10"
    seed = 4
    finetuning = True#False
    degree_of_randomness = 10
    pre_train_size = 500000

    torch.manual_seed(seed)
    np.random.seed(seed)

    if degree_of_randomness > 1:
        dataset = f"{dataset} shuffle_degree: {degree_of_randomness}"

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    if not finetuning and degree_of_randomness > 1: # meaning dataset
        trainset.targets = list((np.array(trainset.targets) + np.random.randint(0, degree_of_randomness, len(trainset))) % 10)
        testset.targets = list((np.array(testset.targets) + np.random.randint(0, degree_of_randomness, len(testset))) % 10)

    if pre_train_size < 50000:
        trainset = torch.utils.data.Subset(trainset, range(0, pre_train_size))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    wandb.init(project="CIFAR10", config= {
        "batch_size": batch_size,
        "keep_prob": keep_prob,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "epochs": epochs,
        "dataset": dataset,
        "finetuning": finetuning,
        "pretrain_size": pre_train_size,
        "degree_of_randomness": degree_of_randomness,
        "seed": seed
    })

    # init net with inital parameter
    if not finetuning:

        net = Cnn(keep_prob=keep_prob)
        net.to(device)

        test_accuracy, base_logits, base_activations = test(net, testloader)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

        train_accuracy, _, _ = test(net, trainloader)
        wandb.log({
            'TEST/accuracy': test_accuracy,
            'TRAIN/accuracy': train_accuracy,
            'TEST/logits': 1
        }, step = 0)
        store_ckas_to_wandb(wandb, [1] * len(base_activations), base_name='TEST/pool', step=0) # 1 since similarity of same matrices

        for epoch in range(1, epochs+1):  # loop over the dataset multiple times

            running_loss = []
            for i, data in enumerate(trainloader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs, _ = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss.append(loss.item())
                if i % 2000 == 0:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {np.mean(running_loss):.3f}')


            train_accuracy, _, _ = test(net, trainloader)
            test_accuracy, logits, activations = test(net, testloader)
            cka_logits, cka_activations = analyze_activation(base_logits, logits, base_activations, activations)

            wandb.log({
                'TRAIN/loss': np.mean(running_loss),
                'TRAIN/accuracy': train_accuracy,
                'TEST/accuracy': test_accuracy,
                'TEST/logits': cka_logits
            }, step = epoch)

            store_ckas_to_wandb(wandb, cka_activations, base_name = 'TEST/pool', step = epoch)


        PATH = f'./cifar_net_{seed}.pth'
        torch.save(net.state_dict(), PATH)

    else:

        PATH = f'./cifar_net_{seed}.pth'

        if dataset.startswith("SVHN"):
            testset = torchvision.datasets.SVHN(root='./data', split="train", download=True, transform=transform)
            if degree_of_randomness > 1:
                testset.labels = list(
                    (np.array(testset.labels) + np.random.randint(0, degree_of_randomness, len(testset))) % 10)

        elif dataset.startswith("cifar10 shuffle_degree"):

            if degree_of_randomness > 1:
                testset.targets = list(
                    (np.array(testset.targets) + np.random.randint(0, degree_of_randomness, len(testset))) % 10)

        elif dataset == "cifar10_shifted":
            testset.targets = list((np.array(testset.targets)+seed)%10)



        trainset_new = torch.utils.data.Subset(testset, range(0, 5000))
        testset_new = torch.utils.data.Subset(testset, range(5000, 10000))

        trainloader = torch.utils.data.DataLoader(trainset_new, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset_new, batch_size=batch_size,  shuffle=False, num_workers=2)

        # we need first of all the pre_init_network
        net = Cnn(keep_prob=keep_prob)
        net.to(device)
        _, pre_base_logits, pre_base_activations = test(net, testloader)

        # now we load the learned pretrained network
        net = Cnn(keep_prob=keep_prob)
        net.to(device)
        net.load_state_dict(torch.load(PATH))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        test_accuracy, base_logits, base_activations = test(net, testloader)
        test_accuracy, logits, activations = test(net, testloader)
        train_accuracy, _, _ = test(net, trainloader)

        pre_cka_logits, pre_cka_activations = analyze_activation(pre_base_logits, logits, pre_base_activations, activations)

        wandb.log({
            'TEST/accuracy': test_accuracy,
            'TRAIN/accuracy': train_accuracy,
            'TEST/logits': 1,
            'TRANSFER_TEST/pre_logits': pre_cka_logits
        }, step=0)
        store_ckas_to_wandb(wandb, [1] * len(base_activations), base_name='TEST/pool', step=0) # 1 since similarity of same matrices
        store_ckas_to_wandb(wandb, pre_cka_activations, base_name='TRANSFER_TEST/pre_pool', step=0)

        for epoch in range(epochs):

            running_loss = []
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs, _ = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss.append(loss.item())

                if i % 50 == 0:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {np.mean(running_loss):.3f}')

            train_accuracy, _, _ = test(net, trainloader)
            test_accuracy, logits, activations = test(net, testloader)

            cka_logits, cka_activations = analyze_activation(base_logits, logits, base_activations, activations)
            pre_cka_logits, cka_activations = analyze_activation(pre_base_logits, logits, pre_base_activations, activations)

            wandb.log({
                'TRAIN/loss': np.mean(running_loss),
                'TRAIN/accuracy': train_accuracy,
                'TEST/accuracy': test_accuracy,
                'TEST/logits': cka_logits,
                'TRANSFER_TEST/pre_logits': pre_cka_logits
            }, step=epoch)

            store_ckas_to_wandb(wandb, cka_activations, base_name='TEST/pool', step=epoch)
            store_ckas_to_wandb(wandb, pre_cka_activations, base_name='TRANSFER_TEST/pre_pool', step=epoch)

        print('Finished Training')
