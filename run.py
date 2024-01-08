# -*- coding: utf-8 -*-
"""
based on https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
"""
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

from torchsummary import summary

import numpy as np
import wandb

from src.utils.cka import linear_CKA
from src.models.cnn import Cnn
from src.models.resnet import ResNet18
from src.models.vgg import Vgg16_bn
from src.utils.utils import resize, store_array_to_wandb, parse_args, print_elapsed_time

def test(net, dataloader, n_samples=0):
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = []
        activations = []

        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs and activations by running images through the network
            outs, acts = net(inputs)

            if n_samples > 0:
                outputs.append(torch.flatten(outs[:, :n_samples], 1).cpu().detach().numpy())
                activations.append(
                    [torch.flatten(activation[:, :n_samples], 1).cpu().detach().numpy() for activation in acts])

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if n_samples > 0:
            outputs = resize(outputs)
            activations = [resize(activation) for activation in zip(*activations)]

    return 100 * correct / total, outputs, activations

def analyze_activation(base_logits, logits, base_activations, activations, n_samples=500):
    cka_activations = []
    for base_activation, activation in zip(base_activations, activations):
        cka = linear_CKA(base_activation[:n_samples], activation[:n_samples])
        cka_activations.append(cka)
    cka_logits = linear_CKA(base_logits[:n_samples], logits[:n_samples])
    return cka_logits, cka_activations

if __name__=='__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    args = parse_args()

    # Access configuration parameters using args
    batch_size = args.batch_size
    keep_prob = args.keep_prob
    learning_rate = args.learning_rate
    momentum = args.momentum
    epochs = args.epochs
    dataset = args.dataset
    model = args.model
    seeds = args.seeds
    finetuning = args.finetuning
    finetuning_size = args.finetuning_size
    pre_trained_dataset = args.pre_trained_dataset
    degree_of_randomness = args.degree_of_randomness
    pre_trained_size = args.pre_trained_size

    if degree_of_randomness > 0:
        dataset = f"{dataset} shuffle_degree: {degree_of_randomness}"

    if finetuning:
        pre_trained_size = -1
    else:
        pre_trained_dataset = ""
        finetuning_size = -1
        
    for seed in seeds:

        since = print_elapsed_time("", None) # only sets start time
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Check if CUDA is available and set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Additional steps to set seed for CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # If degree of randomness is larger than 0 targets are randomized
        # finetuned is randomized later
        if degree_of_randomness > 0:  # meaning dataset
            trainset.targets = list(
                (np.array(trainset.targets) + np.random.randint(0, degree_of_randomness+1, len(trainset))) % 10)
            testset.targets = list(
                (np.array(testset.targets) + np.random.randint(0, degree_of_randomness+1, len(testset))) % 10)

        # only select subset
        if not finetuning and pre_trained_size < 50000:
            trainset = torch.utils.data.Subset(trainset, range(0, pre_trained_size))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        if finetuning:
            name = f"{model} {dataset} {seed} fintuning with {finetuning_size} samples"
        else:
            name = f"{model} {dataset} {seed} training size {pre_trained_size}"

        wandb.init(project="cka_analysis", config={
            "batch_size": batch_size,
            "keep_prob": keep_prob,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "epochs": epochs,
            "dataset": dataset,
            "model": model,
            "finetuning": finetuning,
            "pre_trained_dataset": pre_trained_dataset,
            "pre_trained_size": pre_trained_size,
            "degree_of_randomness": degree_of_randomness,
            "seed": seed
        }, name = name)

        # init net with inital parameter
        if model == "conv4":
            net = Cnn(keep_prob=keep_prob)
        elif model == "resnet18":
            net = ResNet18()
        elif model == "vgg16":
            net = Vgg16_bn()
        net.to(device)

        # Print the summary
        summary(net, (3, 32, 32))  # Assuming input size is (3, 32, 32)
        n_samples = 500
        if finetuning:

            # we load the learned pre-trained network
            PATH = f'./models/{pre_trained_dataset}_{model}_{seed}.pth'
            net.load_state_dict(torch.load(PATH))

            # overwrite dataset
            if dataset.startswith("SVHN"):
                testset = torchvision.datasets.SVHN(root='./data', split="train", download=True, transform=transform)
                if degree_of_randomness > 0:
                    testset.labels = list(
                        (np.array(testset.labels) + np.random.randint(0, degree_of_randomness + 1, len(testset))) % 10)

            elif dataset == "cifar10_shifted":
                testset.targets = list((np.array(testset.targets) + seed) % 10)

            # use testset to prevent overlap with pretraining
            trainset_new = torch.utils.data.Subset(testset, range(0, finetuning_size))
            testset_new = torch.utils.data.Subset(testset, range(finetuning_size, finetuning_size*2))

            trainloader = torch.utils.data.DataLoader(trainset_new, batch_size=batch_size, shuffle=True, num_workers=2)
            testloader = torch.utils.data.DataLoader(testset_new, batch_size=batch_size, shuffle=False, num_workers=2)

            # we need the pre_initialized_init_network
            if model == "conv4":
                pre_initialized_net = Cnn(keep_prob=keep_prob)
            elif model == "resnet18":
                pre_initialized_net = ResNet18()
            elif model == "vgg16":
                net = Vgg16_bn()
            pre_initialized_net.to(device)
            _, pre_initialized_base_logits, pre_initialized_base_activations = test(pre_initialized_net, testloader, n_samples)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        since = print_elapsed_time("Setup", since)
        train_accuracy, _, _ = test(net, trainloader, )
        since = print_elapsed_time("Train accuracy", since)
        test_accuracy, base_logits, base_activations = test(net, testloader, n_samples)
        since = print_elapsed_time("Test accuracy", since)

        wandb.log({
            'TEST/accuracy': test_accuracy,
            'TRAIN/accuracy': train_accuracy,
            'CKAS/logits': 1
        }, step=0)
        store_array_to_wandb(wandb, [1] * len(base_activations), base_name='CKAS/pool',
                             step=0)  # 1 since similarity of same matrices

        if finetuning:
            # todo the next line could also be replaced with base
            _, logits, activations = test(net, testloader, n_samples)

            since = print_elapsed_time("Test activations", since)
            pre_initialized_cka_logits, pre_initialized_cka_activations = analyze_activation(
                pre_initialized_base_logits,
                logits,
                pre_initialized_base_activations,
                activations,
                n_samples)
            since = print_elapsed_time("CKA analysis (pre)", since)
            store_array_to_wandb(wandb, pre_initialized_cka_activations, base_name='CKAS/pre_initialized_pool', step=0)
            wandb.log({
                'CKAS/pre_initialized_logits': pre_initialized_cka_logits
            }, step=0)

        for epoch in range(1, epochs + 1):  # loop over the dataset multiple times

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
                if finetuning:
                    print_frequence = int(finetuning_size / batch_size / 8)
                else:
                    print_frequence = int(pre_trained_size / batch_size / 8)
                if print_frequence <= 0:
                    print_frequence = 1
                if (i+1) % print_frequence == 0:
                    print(f'[{epoch}, {(i+1):5d}] loss: {np.mean(running_loss):.3f}')

            since = print_elapsed_time(f"Train {epoch} epoch", since)
            train_accuracy, _, _ = test(net, trainloader, )
            since = print_elapsed_time("Train accuracy", since)
            test_accuracy, logits, activations = test(net, testloader, n_samples)
            since = print_elapsed_time("Test accuracy", since)
            cka_logits, cka_activations = analyze_activation(base_logits, logits, base_activations, activations, n_samples)
            since = print_elapsed_time("CKA analysis", since)

            wandb.log({
                'TRAIN/loss': np.mean(running_loss),
                'TRAIN/accuracy': train_accuracy,
                'TEST/accuracy': test_accuracy,
                'CKAS/logits': cka_logits
            }, step=epoch)

            store_array_to_wandb(wandb, cka_activations, base_name='CKAS/pool', step=epoch)

            if finetuning:
                pre_initialized_cka_logits, pre_initialized_cka_activations = analyze_activation(pre_initialized_base_logits, logits,
                                                                                 pre_initialized_base_activations,
                                                                                 activations, n_samples)
                since = print_elapsed_time("CKA analysis (pre)", since)
                wandb.log({
                    'CKAS/pre_initialized_logits': pre_initialized_cka_logits
                }, step=epoch)

                store_array_to_wandb(wandb, pre_initialized_cka_activations, base_name='CKAS/pre_initialized_pool',
                                     step=epoch)

        wandb.finish()
        # only store if cifar10 pretrained
        if not finetuning:# and dataset == "cifar10":
            # Store model
            PATH = f'./models/{dataset}_{model}_{seed}.pth'
            torch.save(net.state_dict(), PATH)

        print(f'Finished training seed {seed}')
