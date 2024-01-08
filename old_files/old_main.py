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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = self.convLayer(3, 64, keep_prob)
        self.conv2 = self.convLayer(64, 64, keep_prob)
        self.conv3 = self.convLayer(64, 64, keep_prob)
        self.conv4 = self.convLayer(64, 64, keep_prob)
        self.linear = nn.Linear(256, 10)

    def convLayer(self, in_channels, out_channels, keep_prob=0.2):
        """3*3 convolution with padding,ever time call it the output size become half"""
        cnn_seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(keep_prob)
        )
        return cnn_seq

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.flatten(x4, 1) # flatten all dimensions except batch
        logits = self.linear(x)
        return logits, x1, x2, x3, x4

def resize(array):
    array = np.concatenate(array)
    return array

def test(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        outs = []
        x1s = []
        x2s = []
        x3s = []
        x4s = []
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs, x1, x2, x3, x4 = net(inputs)

            outs.append(torch.flatten(outputs, 1).cpu().detach().numpy())
            x1s.append(torch.flatten(x1, 1).cpu().detach().numpy())
            x2s.append(torch.flatten(x2, 1).cpu().detach().numpy())
            x3s.append(torch.flatten(x3, 1).cpu().detach().numpy())
            x4s.append(torch.flatten(x4, 1).cpu().detach().numpy())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        x1s = resize(x1s)
        x2s = resize(x2s)
        x3s = resize(x3s)
        x4s = resize(x4s)
        outs = resize(outs)
    return 100 * correct / total, outs, x1s, x2s, x3s, x4s

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__=='__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32
    keep_prob = 0.0
    learning_rate = 0.001
    momentum = 0.9
    epochs = 25
    dataset = "cifar10"
    seed = 3
    finetuning = True
    degree_of_randomness = 1
    pre_trained_size = 500000

    torch.manual_seed(seed)
    np.random.seed(seed)
    # Check if CUDA is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Print the device information
    print(f"Using device: {device}")

    # Additional steps to set seed for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if degree_of_randomness > 1:
        dataset = f"{dataset} shuffle_degree: {degree_of_randomness}"

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    if not finetuning and degree_of_randomness > 1: # meaning dataset
        trainset.targets = list((np.array(trainset.targets) + np.random.randint(0, degree_of_randomness, len(trainset))) % 10)
        testset.targets = list((np.array(testset.targets) + np.random.randint(0, degree_of_randomness, len(testset))) % 10)

    if pre_trained_size < 50000:
        trainset = torch.utils.data.Subset(trainset, range(0, pre_trained_size))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    wandb.init(project="CIFAR10", config= {
        "batch_size": batch_size,
        "keep_prob": keep_prob,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "epochs": epochs,
        "dataset": dataset,
        "finetuning": finetuning,
        "pre_trained_size": pre_trained_size,
        "degree_of_randomness": degree_of_randomness,
        "seed": seed
    })

    # init net with inital parameter
    if not finetuning:

        net = Net()
        net.to(device)

        test_accuracy, base_logits, base_x1, base_x2, base_x3, base_x4 = test(net, testloader)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

        train_accuracy, _, _, _, _, _ = test(net, trainloader)
        wandb.log({

            'TEST/accuracy': test_accuracy,
            'TRAIN/accuracy': train_accuracy,
            'TEST/pool1': 1,
            'TEST/pool2': 1,
            'TEST/pool3': 1,
            'TEST/pool4': 1,
            'TEST/logits': 1
        })

        for epoch in range(epochs):  # loop over the dataset multiple times


            running_loss = []
            for i, data in enumerate(trainloader, 0):


                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs, x1, x2, x3, x4 = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss.append(loss.item())
                if i % 200 == 199:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {np.mean(running_loss):.3f}')


            test_accuracy, logits, x1, x2, x3, x4 = test(net, testloader)
            train_accuracy, _, _, _, _, _ = test(net, trainloader)

            maxi = 200
            cka_x1 = linear_CKA(base_x1[:maxi], x1[:maxi])
            cka_x2 = linear_CKA(base_x2[:maxi], x2[:maxi])
            cka_x3 = linear_CKA(base_x3[:maxi], x3[:maxi])
            cka_x4 = linear_CKA(base_x4[:maxi], x4[:maxi])
            cka_logits = linear_CKA(base_logits[:maxi], logits[:maxi])

            wandb.log({
                'TRAIN/loss': np.mean(running_loss),
                'TRAIN/accuracy': train_accuracy,
                'TEST/accuracy': test_accuracy,
                'TEST/pool1': cka_x1,
                'TEST/pool2': cka_x2,
                'TEST/pool3': cka_x3,
                'TEST/pool4': cka_x4,
                'TEST/logits': cka_logits
            })

        PATH = f'./models/cifar_net_{seed}.pth'
        torch.save(net.state_dict(), PATH)

    else:

        PATH = f'./models/cifar_net_{seed}.pth'


        if dataset.startswith("SVHN"):
            testset = torchvision.datasets.SVHN(root='./data', split="train",
                                                download=True, transform=transform)
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
        testloader = torch.utils.data.DataLoader(testset_new, batch_size=batch_size, shuffle=False, num_workers=2)

        # we need first of all the pre_init_network
        net = Net()
        net.to(device)

        test_accuracy, base_logits, base_x1, base_x2, base_x3, base_x4 = test(net, testloader)
        pre_base_logits, pre_base_x1, pre_base_x2, pre_base_x3, pre_base_x4 = base_logits, base_x1, base_x2, base_x3, base_x4

        # now we load the learned pretrained network
        net = Net()
        net.to(device)

        net.load_state_dict(torch.load(PATH))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

        maxi = 100
        test_accuracy, base_logits, base_x1, base_x2, base_x3, base_x4 = test(net, testloader)
        logits, x1, x2, x3, x4 = base_logits, base_x1, base_x2, base_x3, base_x4
        train_accuracy, _, _, _, _, _ = test(net, trainloader)

        pre_cka_x1 = linear_CKA(pre_base_x1[:maxi], x1[:maxi])
        pre_cka_x2 = linear_CKA(pre_base_x2[:maxi], x2[:maxi])
        pre_cka_x3 = linear_CKA(pre_base_x3[:maxi], x3[:maxi])
        pre_cka_x4 = linear_CKA(pre_base_x4[:maxi], x4[:maxi])
        pre_cka_logits = linear_CKA(pre_base_logits[:maxi], logits[:maxi])

        wandb.log({

            'TRANSFER_TRAIN/accuracy': train_accuracy,
            'TRANSFER_TEST/accuracy': test_accuracy,
            'TRANSFER_TEST/pool1': 1,
            'TRANSFER_TEST/pool2': 1,
            'TRANSFER_TEST/pool3': 1,
            'TRANSFER_TEST/pool4': 1,
            'TRANSFER_TEST/logits': 1,
            'TRANSFER_TEST/pre_pool1': pre_cka_x1,
            'TRANSFER_TEST/pre_pool2': pre_cka_x2,
            'TRANSFER_TEST/pre_pool3': pre_cka_x3,
            'TRANSFER_TEST/pre_pool4': pre_cka_x4,
            'TRANSFER_TEST/pre_logits': pre_cka_logits
        })



        for epoch in range(epochs):

            running_loss = []
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                outputs, x1, x2, x3, x4 = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss.append(loss.item())
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {np.mean(running_loss):.3f}')


            test_accuracy, logits, x1, x2, x3, x4 = test(net, testloader)
            train_accuracy, _, _, _, _, _ = test(net, trainloader)

            cka_x1 = linear_CKA(base_x1[:maxi], x1[:maxi])
            cka_x2 = linear_CKA(base_x2[:maxi], x2[:maxi])
            cka_x3 = linear_CKA(base_x3[:maxi], x3[:maxi])
            cka_x4 = linear_CKA(base_x4[:maxi], x4[:maxi])
            cka_logits = linear_CKA(base_logits[:maxi], logits[:maxi])

            pre_cka_x1 = linear_CKA(pre_base_x1[:maxi], x1[:maxi])
            pre_cka_x2 = linear_CKA(pre_base_x2[:maxi], x2[:maxi])
            pre_cka_x3 = linear_CKA(pre_base_x3[:maxi], x3[:maxi])
            pre_cka_x4 = linear_CKA(pre_base_x4[:maxi], x4[:maxi])
            pre_cka_logits = linear_CKA(pre_base_logits[:maxi], logits[:maxi])

            wandb.log({
                'TRANSFER_TRAIN/loss': np.mean(running_loss),
                'TRANSFER_TRAIN/accuracy': train_accuracy,
                'TRANSFER_TEST/accuracy': test_accuracy,
                'TRANSFER_TEST/pool1': cka_x1,
                'TRANSFER_TEST/pool2': cka_x2,
                'TRANSFER_TEST/pool3': cka_x3,
                'TRANSFER_TEST/pool4': cka_x4,
                'TRANSFER_TEST/logits': cka_logits,
                'TRANSFER_TEST/pre_pool1': pre_cka_x1,
                'TRANSFER_TEST/pre_pool2': pre_cka_x2,
                'TRANSFER_TEST/pre_pool3': pre_cka_x3,
                'TRANSFER_TEST/pre_pool4': pre_cka_x4,
                'TRANSFER_TEST/pre_logits': pre_cka_logits
            })

        print('Finished Training')


    # else:
    #
    #     # load model
    #     PATH = f'./models/cifar_net_{seed}.pth'
    #     if dataset.startswith("SVHN"):
    #         testset = torchvision.datasets.SVHN(root='./data', split="train", download=True, transform=transform)
    #         if degree_of_randomness > 1:
    #             testset.labels = list(
    #                 (np.array(testset.labels) + np.random.randint(0, degree_of_randomness, len(testset))) % 10)
    #
    #     elif dataset.startswith("cifar10 shuffle_degree"):
    #
    #         if degree_of_randomness > 1:
    #             testset.targets = list(
    #                 (np.array(testset.targets) + np.random.randint(0, degree_of_randomness, len(testset))) % 10)
    #
    #     elif dataset == "cifar10_shifted":
    #         testset.targets = list((np.array(testset.targets)+seed)%10)
    #
    #
    #
    #     trainset_new = torch.utils.data.Subset(testset, range(0, 5000))
    #     testset_new = torch.utils.data.Subset(testset, range(5000, 10000))
    #
    #     trainloader = torch.utils.data.DataLoader(trainset_new, batch_size=batch_size, shuffle=True, num_workers=2)
    #     testloader = torch.utils.data.DataLoader(testset_new, batch_size=batch_size,  shuffle=False, num_workers=2)
    #
    #     # we need first of all the pre_initialized_init_network
    #     net = Cnn(keep_prob=keep_prob)
    #     net.to(device)
    #     _, pre_initialized_base_logits, pre_initialized_base_activations = test(net, testloader)
    #
    #     # now we load the learned pretrained network
    #     net = Cnn(keep_prob=keep_prob)
    #     net.to(device)
    #     net.load_state_dict(torch.load(PATH))
    #
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    #     test_accuracy, base_logits, base_activations = test(net, testloader)
    #     test_accuracy, logits, activations = test(net, testloader)
    #     train_accuracy, _, _ = test(net, trainloader)
    #
    #     pre_initialized_cka_logits, pre_initialized_cka_activations = analyze_activation(pre_initialized_base_logits, logits, pre_initialized_base_activations, activations)
    #
    #     wandb.log({
    #         'TRANSFER_TRAIN/accuracy': train_accuracy,
    #         'TRANSFER_TEST/accuracy': test_accuracy,
    #         'TRANSFER_TEST/logits': 1,
    #         'TRANSFER_TEST/pre_initialized_logits': pre_initialized_cka_logits
    #     }, step=0)
    #     store_array_to_wandb(wandb, [1] * len(base_activations), base_name='TRANSFER_TEST/pool', step=0) # 1 since similarity of same matrices
    #     store_array_to_wandb(wandb, pre_initialized_cka_activations, base_name='TRANSFER_TEST/pre_initialized_pool', step=0)
    #
    #     for epoch in range(epochs):
    #
    #         running_loss = []
    #         for i, data in enumerate(trainloader, 0):
    #             inputs, labels = data[0].to(device), data[1].to(device)
    #
    #             # zero the parameter gradients
    #             optimizer.zero_grad()
    #             outputs, _ = net(inputs)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
    #
    #             # print statistics
    #             running_loss.append(loss.item())
    #
    #             if i % 50 == 0:
    #                 print(f'[{epoch + 1}, {i + 1:5d}] loss: {np.mean(running_loss):.3f}')
    #
    #         train_accuracy, _, _ = test(net, trainloader)
    #         test_accuracy, logits, activations = test(net, testloader)
    #
    #         cka_logits, cka_activations = analyze_activation(base_logits, logits, base_activations, activations)
    #         pre_initialized_cka_logits, cka_activations = analyze_activation(pre_initialized_base_logits, logits, pre_initialized_base_activations, activations)
    #
    #         wandb.log({
    #             'TRANSFER_TRAIN/loss': np.mean(running_loss),
    #             'TRANSFER_TRAIN/accuracy': train_accuracy,
    #             'TRANSFER_TEST/accuracy': test_accuracy,
    #             'TRANSFER_TEST/logits': cka_logits,
    #             'TRANSFER_TEST/pre_initialized_logits': pre_initialized_cka_logits
    #         }, step=epoch)
    #
    #         store_array_to_wandb(wandb, cka_activations, base_name='TRANSFER_TEST/pool', step=epoch)
    #         store_array_to_wandb(wandb, pre_initialized_cka_activations, base_name='TRANSFER_TEST/pre_initialized_pool', step=epoch)
