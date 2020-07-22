import torch
import torch.nn as nn
import sparseconvnet as scn


class FBSparseVGG(nn.Module):
    def __init__(self, nr_classes, input_channels=2, vgg_12=False):
        super(FBSparseVGG, self).__init__()
        if vgg_12:
            sparse_out_channels = 128
            self.sparseModel = scn.SparseVggNet(2, nInputPlanes=input_channels, layers=[
                ['C', 16], ['C', 16], 'MP',
                ['C', 32], ['C', 32], 'MP',
                ['C', 64], ['C', 64], 'MP',
                ['C', 128], ['C', 128], 'MP',
                ['C', 256]]
            ).add(scn.Convolution(2, 256, sparse_out_channels, 3, filter_stride=2, bias=False)
            ).add(scn.BatchNormReLU(sparse_out_channels)
            ).add(scn.SparseToDense(2, sparse_out_channels))

        else:
            sparse_out_channels = 256
            self.sparseModel = scn.SparseVggNet(2, nInputPlanes=input_channels, layers=[
                ['C', 16], ['C', 16], 'MP',
                ['C', 32], ['C', 32], 'MP',
                ['C', 64], ['C', 64], 'MP',
                ['C', 128], ['C', 128], 'MP',
                ['C', 256], ['C', 256], 'MP',
                ['C', 512]]
            ).add(scn.Convolution(2, 512, sparse_out_channels, 3, filter_stride=2, bias=False)
            ).add(scn.BatchNormReLU(sparse_out_channels)
            ).add(scn.SparseToDense(2, sparse_out_channels))

        cnn_spatial_output_size = [2, 3]
        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor(cnn_spatial_output_size))
        self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.spatial_size, mode=2)
        self.linear_input_features = cnn_spatial_output_size[0] * cnn_spatial_output_size[1] * sparse_out_channels
        self.linear = nn.Linear(self.linear_input_features, nr_classes)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, self.linear_input_features)
        x = self.linear(x)

        return x


class FBSparseSparsityInvVGG(FBSparseVGG):
    def __init__(self, nr_classes, input_channels=2):
        FBSparseVGG.__init__(self, nr_classes, input_channels)

        # generate dense convolutions with only 1s and sparse to dense transforms
        self.ones_transforms = []

        for name, mod in self.sparseModel._modules.items():
            if type(mod) is scn.SubmanifoldConvolution:
                self.ones_transforms += [nn.Conv2d(1,1, kernel_size=mod.filter_size[0].item(), padding=mod.filter_size[0].item()//2, bias=False)]
                self.ones_transforms[-1].weight.use_grad = False

        self.ones_transforms = nn.ModuleList(self.ones_transforms)

    def normalize_input(self, x, ones_conv):
        # implements sparsity invariant convolutions: http://www.cvlibs.net/publications/Uhrig2017THREEDV.pdf
        # unfortunately, now works by bringing activation to dense representation, normalizing and then back to sparse
        eps = 1e-5
        dense_to_sparse = scn.DenseToSparse(2)
        sparse_to_dense = scn.SparseToDense(2, x.features.shape[-1])

        dense_x = sparse_to_dense(x)

        # compute active sites and normalization factors
        active_sites = (dense_x.abs().sum(1, keepdim=True)>0).float()
        dense_x_normalized = dense_x/(ones_conv(active_sites)+eps)

        # back to sparse
        x_normalized = dense_to_sparse(dense_x_normalized)

        return x_normalized

    def forward(self, x):
        x = self.inputLayer(x)
        counter = -1
        for idx, (name, mod) in enumerate(self.sparseModel._modules.items()):
            # if there is a submanifold convolution add sparsity invariant conv
            if type(mod) is scn.SubmanifoldConvolution:
                counter += 1
                x = self.normalize_input(x, self.ones_transforms[counter])
            x = mod(x)

        x = x.view(-1, self.linear_input_features)
        x = self.linear(x)

        return x


class FBSparseVGGTest(nn.Module):
    def __init__(self, nr_classes):
        super(FBSparseVGGTest, self).__init__()
        self.sparseModel = scn.SparseVggNet(2, nInputPlanes=2, layers=[
            ['C', 16], ['C', 16], 'MP',
            ['C', 32], ['C', 32], 'MP',
            ['C', 64], ['C', 64], 'MP',
            ['C', 128], ['C', 128], 'MP',
            ['C', 256], ['C', 256], 'MP',
            ['C', 512]]
        ).add(scn.Convolution(2, 512, 256, 3, filter_stride=2, bias=False)
        ).add(scn.BatchNormReLU(256)
        ).add(scn.SparseToDense(2, 256))

        cnn_spatial_output_size = [2, 3]
        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor(cnn_spatial_output_size))
        self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.spatial_size, mode=2)
        self.linear_input_features = cnn_spatial_output_size[0] * cnn_spatial_output_size[1] * 256
        self.linear = nn.Linear(self.linear_input_features, nr_classes)

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.sparseModel(x)
        x = x.view(-1, self.linear_input_features)
        x = self.linear(x)

        return x
