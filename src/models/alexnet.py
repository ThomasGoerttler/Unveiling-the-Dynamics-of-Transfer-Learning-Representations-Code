# adapted from https://blog.paperspace.com/alexnet-pytorch/
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, small = False):
        super(AlexNet, self).__init__()
        if small:
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer3 = nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU())
            self.layer4 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU())
            self.layer5 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(4096, 1024),  # Adjusted input size for CIFAR-10
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(1024, 1024),
                nn.ReLU())
            self.classifier = nn.Sequential(
                nn.Linear(1024, num_classes))
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2))
            self.layer3 = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU())
            self.layer4 = nn.Sequential(
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU())
            self.layer5 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2))
            self.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(9216, 4096),
                nn.ReLU())
            self.fc2 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU())
            self.classifier = nn.Sequential(
                nn.Linear(4096, num_classes))



    def forward(self, x):
        activations = []
        x = self.layer1(x)
        activations.append(x.clone())
        x = self.layer2(x)
        activations.append(x.clone())
        x = self.layer3(x)
        activations.append(x.clone())
        x = self.layer4(x)
        activations.append(x.clone())
        x = self.layer5(x)
        activations.append(x.clone())
        x = x.reshape(x.size(0), -1)
        activations.append(x.clone())
        x = self.fc1(x)
        activations.append(x.clone())
        x = self.fc2(x)
        activations.append(x.clone())
        out = self.classifier(x)
        activations.append(x.clone())
        return out, activations

