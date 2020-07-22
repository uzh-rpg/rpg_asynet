import numpy as np
import torch
from PIL import Image


class ReferenceVGGModel:
    def __init__(self, layer_list, device=torch.device('cpu')):
        self.device = device
        self.layers = []
        self.layer_list = layer_list

        self.createReferenceVGG()
        self.setIdentityWeights()

    def forward(self, x_inter, active_sum=None, active_sites_vis=None):
        """Apply asynchronous layers"""
        x_inter = x_inter.permute(2, 0, 1).unsqueeze(0)

        for j, layer in enumerate(self.layers):
            # print('Layer Name: %s' % self.layer_list[j][0])
            # if active_sum is not None and self.layer_list[j][0][:7] != 'Classic':
            if active_sum is not None:
                active_sum[j] = ((x_inter.sum(1)) > 0).sum()

            if active_sites_vis is not None and self.layer_list[j][0][:7] != 'Classic':
                o_height, o_width = active_sites_vis.shape[2:4]

                np_img = ((x_inter.sum(1) > 0).float()[0, :, :, None].cpu().numpy() * np.array([0, 0, 1])[None, None, :])
                np_img = np.uint8(np_img)
                img = Image.fromarray(np_img)
                np_img = np.asarray(img.resize([o_width, o_height], resample=0))

                x_idx = np_img.nonzero()
                active_sites_vis[0, j, x_idx[0], x_idx[1], :] = np.array([0, 1, 0])

            if self.layer_list[j][0] == 'ClassicFC':
                x_inter = self.layers[j](x_inter.view(1, -1))
            elif self.layer_list[j][0] == 'BNRelu' or self.layer_list[j][0] == 'ClassicBNRelu':
                pass
            else:
                x_inter = self.layers[j](x_inter)

    def setIdentityWeights(self):
        """Sets the different weights and biases equal"""
        for j, i_layer in enumerate(self.layer_list):
            layer_name = i_layer[0]
            if layer_name == 'C' or layer_name == 'ClassicC' or layer_name == 'ClassicFC':
                self.layers[j].weight.data.fill_(1)

    def createReferenceVGG(self):
        """Creates a asynchronous VGG"""
        for j, i_layer in enumerate(self.layer_list):
            if i_layer[0] == 'C':
                self.layers.append(torch.nn.Conv2d(in_channels=i_layer[1], out_channels=i_layer[2], kernel_size=3,
                                                   padding=1, bias=False, padding_mode='zeros'))
            elif i_layer[0] == 'ClassicC':
                self.layers.append(torch.nn.Conv2d(in_channels=i_layer[1], out_channels=i_layer[2],
                                                   kernel_size=i_layer[3], stride=i_layer[4], bias=False,))
            elif i_layer[0] == 'MP':
                self.layers.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
            elif i_layer[0] == 'ClassicFC':
                self.layers.append(torch.nn.Linear(in_features=i_layer[1], out_features=i_layer[2], bias=False))
            elif self.layer_list[j][0] == 'BNRelu' or self.layer_list[j][0] == 'ClassicBNRelu':
                self.layers.append(None)

        return self.layers