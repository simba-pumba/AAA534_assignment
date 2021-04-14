import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, layers, num_classes, drop_prob):
        super(Model, self).__init__()

        self.in_channels = 16
        self.dropout = torch.nn.Dropout(p=drop_prob)
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(Block, 16, layers[0])
        self.layer2 = self.make_layer(Block, 32, layers[1], 2)
        self.layer3 = self.make_layer(Block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def xavier_init(self, m):
        torch.nn.init.xavier_uniform_(m.weight)

    def he_init(self, m):
        torch.nn.init.kaiming_uniform_(m.weight)

    def reset_params(self, method):
        if method == "xavier":
            torch.nn.init.xavier_uniform_(self.conv.weight)
            self.linear1.apply(xavier_init)
            self.layer2.apply(xavier_init)
            self.layer3.apply(xavier_init)
            torch.nn.init.xavier_uniform_(self.fc.weight)
        elif method == "he":
            torch.nn.init.kaiming_uniform_(self.conv.weight)
            self.linear1.apply(xavier_init)
            self.layer2.apply(xavier_init)
            self.layer3.apply(xavier_init)
            torch.nn.init.kaiming_uniform_(self.fc.weight)
        else:
            assert ValueError

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.dropout(out)
        out = self.layer3(out)
        out = self.dropout(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

