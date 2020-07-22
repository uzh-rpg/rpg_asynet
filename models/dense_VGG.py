import torch.nn as nn


class DenseVGG(nn.Module):
    def __init__(self, nr_classes, in_c=2, vgg_12=False):
        super().__init__()
        self.kernel_size = 3
        if vgg_12:
            sparse_out_channels = 128
            self.conv_layers = nn.Sequential(
                self.conv_block(in_c=in_c, out_c=16),
                self.conv_block(in_c=16, out_c=32),
                self.conv_block(in_c=32, out_c=64),
                self.conv_block(in_c=64, out_c=128),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=sparse_out_channels, kernel_size=self.kernel_size,
                          stride=2, bias=False),
                nn.BatchNorm2d(sparse_out_channels),
                nn.ReLU()
            )

        else:
            sparse_out_channels = 256
            self.conv_layers = nn.Sequential(
                self.conv_block(in_c=in_c, out_c=16),
                self.conv_block(in_c=16, out_c=32),
                self.conv_block(in_c=32, out_c=64),
                self.conv_block(in_c=64, out_c=128),
                self.conv_block(in_c=128, out_c=256),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(in_channels=512, out_channels=sparse_out_channels, kernel_size=self.kernel_size,
                          stride=2, bias=False),
                nn.BatchNorm2d(sparse_out_channels),
                nn.ReLU()
            )

        self.linear_input_features = 2 * 3 * sparse_out_channels
        self.linear = nn.Linear(self.linear_input_features, nr_classes)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=self.kernel_size, padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.kernel_size, stride=2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), self.linear_input_features)  # flat
        x = self.linear(x)

        return x
