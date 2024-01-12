import torch
import torch.nn as nn

class Cnn(nn.Module):
    def __init__(self, keep_prob, num_classes=10):
        super().__init__()
        self.conv1 = self.convLayer(3, 64, keep_prob)
        self.conv2 = self.convLayer(64, 64, keep_prob)
        self.conv3 = self.convLayer(64, 64, keep_prob)
        self.conv4 = self.convLayer(64, 64, keep_prob)
        self.linear = nn.Linear(256, num_classes)

    def convLayer(self, in_channels, out_channels, keep_prob=0.2):
        """3*3 convolution with padding, every time call it the output size becomes half"""
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
        x = torch.flatten(x4, 1)  # flatten all dimensions except batch
        logits = self.linear(x)

        # Return logits and intermediate feature maps as a list
        return logits, [x1, x2, x3, x4, logits]
