import torch
import torch.nn as nn
import sparseconvnet as scn


class FBSparseObjectDet(nn.Module):
    def __init__(self, nr_classes, nr_box=2, nr_input_channels=2, small_out_map=True):
        super(FBSparseObjectDet, self).__init__()
        self.nr_classes = nr_classes
        self.nr_box = nr_box

        sparse_out_channels = 256
        self.sparseModel = scn.SparseVggNet(2, nInputPlanes=nr_input_channels, layers=[
            ['C', 16], ['C', 16], 'MP',
            ['C', 32], ['C', 32], 'MP',
            ['C', 64], ['C', 64], 'MP',
            ['C', 128], ['C', 128], 'MP',
            ['C', 256], ['C', 256]]
            ).add(scn.Convolution(2, 256, sparse_out_channels, 3, filter_stride=2, bias=False)
            ).add(scn.BatchNormReLU(sparse_out_channels)
            ).add(scn.SparseToDense(2, sparse_out_channels))

        if small_out_map:
            self.cnn_spatial_output_size = [5, 7]
        else:
            self.cnn_spatial_output_size = [6, 8]

        spatial_size_product = self.cnn_spatial_output_size[0] * self.cnn_spatial_output_size[1]
        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor(self.cnn_spatial_output_size))
        self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.spatial_size, mode=2)
        self.linear_input_features = spatial_size_product * 256
        self.linear_1 = nn.Linear(self.linear_input_features, 1024)
        self.linear_2 = nn.Linear(1024, spatial_size_product*(nr_classes + 5*self.nr_box))

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, self.linear_input_features)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.linear_2(x)
        x = x.view([-1] + self.cnn_spatial_output_size + [(self.nr_classes + 5*self.nr_box)])

        return x
