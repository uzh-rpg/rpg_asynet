import torch
from time import perf_counter

class asynMaxPool:
    def __init__(self, dimension, filter_size, filter_stride, padding_mode='same', first_layer=False,
                 device=torch.device('cpu')):
        """
        Constructs a max pooling layer.

        :param dimension: spatial dimension of the max pooling layer [1, 2, ...]
        :param filter_size: size of the pooling kernel
        :param filter_stride: stride of the kernel
        :param padding: 'same' or 'valid'. Identical behaviour to tensorflow max pooling
        """
        self.dimension = dimension
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.padding_mode = padding_mode
        self.first_layer = first_layer
        self.device = device

        self.padding = [filter_size // 2] * dimension * 2
        self.filter_size_tensor = torch.LongTensor(dimension).fill_(self.filter_size).to(device)
        # Construct lookup table for 1d kernel position to position in nd
        kernel_indices = torch.stack(torch.meshgrid([torch.arange(filter_size) for _ in range(dimension)]), dim=-1)
        self.filter_volume = self.filter_size_tensor.prod().item()
        self.kernel_indices = kernel_indices.reshape([self.filter_volume, dimension]).to(device)
        self.border_dist = filter_size // 2 if padding_mode == 'valid' else 0

        self.output_feature_map = None
        self.old_max_indices_map = None
        self.output_arguments_map = None

    def forward(self, update_location, feature_map):
        """
        Computes the a asynchronous sparse convolution layer based on the update_location

        :param update_location: tensor with shape [N_active, dimension]
        :param feature_map: tensor with shape [N, nIn]
        :return:
        """
        self.checkInputArguments(update_location, feature_map)
        update_location_indices = update_location.split(1, dim=-1)
        output_spatial_shape = (torch.tensor(feature_map.shape[:-1], device=self.device).float() -
                                self.border_dist*2) / self.filter_stride
        output_spatial_shape = torch.ceil(output_spatial_shape).long()
        if self.output_arguments_map is None:
            out_arg_shape = list(output_spatial_shape) + [self.filter_volume] + [feature_map.shape[-1]]
            self.output_arguments_map = torch.zeros(torch.Size(out_arg_shape), device=self.device)
        if self.output_feature_map is None:
            self.output_feature_map = torch.zeros(list(output_spatial_shape) + [feature_map.shape[-1]],
                                                  device=self.device)
        if self.old_max_indices_map is None:
            self.old_max_indices_map = torch.ones(list(output_spatial_shape) + [feature_map.shape[-1], self.dimension],
                                                  device=self.device).long() * -1

        # Find output locations
        out = self.computeOutputLocations(update_location, output_spatial_shape)
        output_locations, valid_locations, nr_update_locations = out

        # Kernel indices relative to output position (reversed kernel index)
        dummy_kernel_indices = torch.arange(self.filter_volume - 1, -1, step=-1, dtype=torch.float, device=self.device)
        dummy_kernel_indices = dummy_kernel_indices.repeat(nr_update_locations)
        dummy_kernel_indices = dummy_kernel_indices[valid_locations]

        # Update arguments in output argument map
        argument_indices = torch.cat((output_locations, dummy_kernel_indices[:, None]), dim=-1).long().split(1, dim=-1)
        input_locations = update_location[:, None, :].repeat([1, self.filter_volume, 1])
        input_locations = input_locations.view(-1, self.dimension)[valid_locations, :]
        self.output_arguments_map[argument_indices] = feature_map[input_locations.split(1, dim=-1)].float()

        n_Channels = self.output_arguments_map.shape[-1]
        zero_added_output_arguments_map = torch.cat((torch.zeros(list(output_spatial_shape) + [1, n_Channels],
                                                                 device=self.device),
                                                     self.output_arguments_map), dim=-2)

        output_indices = argument_indices[:1]
        out = torch.max(zero_added_output_arguments_map[output_indices], dim=-2)
        self.output_feature_map[output_indices], max_indices = out
        self.old_max_indices_map[output_indices] = self.computeMaxIndices(max_indices, output_spatial_shape,
                                                                          output_indices)

        new_updates_map = torch.zeros(torch.Size(output_spatial_shape))
        new_updates_map[output_locations.long().split(1, dim=-1)] = 1
        new_update_events = new_updates_map.nonzero()

        return new_update_events, self.output_feature_map.clone(), self.old_max_indices_map.clone()

    def computeOutputLocations(self, update_location, output_spatial_shape):
        """Computes the outpout location based on the input shape, stride and padding mode"""
        output_locations = update_location[:, None, :] + self.kernel_indices[None, :, :] - self.filter_size // 2
        # Adjust for padding mode and stride
        output_locations = (output_locations - self.border_dist).float() / self.filter_stride
        nr_update_locations = output_locations.shape[0]
        output_locations = output_locations.view(-1, self.dimension)

        # Check the conditions for a valid index: even and inside frame
        valid_locations_even = torch.eq(output_locations.int().float(), output_locations).sum(-1) == self.dimension
        spatial_dimension = (output_spatial_shape - 1).unsqueeze(0).float()

        valid_locations_frame = ~((output_locations < 0).sum(-1) + (output_locations > spatial_dimension).sum(-1)).bool()
        valid_locations = valid_locations_even & valid_locations_frame

        return output_locations[valid_locations, :], valid_locations, nr_update_locations

    def computeMaxIndices(self, max_indices, output_spatial_shape, output_indices):
        """Compute the maximum indices"""
        # Shift index by -1, current 0 index corresponds to zero feature value, which will be changed to -1
        max_indices = max_indices - 1
        kernel_position = self.kernel_indices[max_indices, :]

        grid_map = torch.stack(torch.meshgrid([torch.arange(output_spatial_shape[i], device=self.device) for i in
                                               range(self.dimension)]), dim=-1)[output_indices]

        max_position = grid_map.unsqueeze(dim=-2) * self.filter_stride + self.border_dist - \
                       self.filter_size // 2 + kernel_position
        max_indices = max_position
        max_indices[max_indices == -1] = -1

        return max_indices

    def checkInputArguments(self, update_location, feature_map):
        """Checks if the input arguments have the correct shape"""
        if update_location.ndim != 2 or update_location.shape[-1] != self.dimension:
            raise ValueError('Expected update_location to have shape [N, %s]. Got size %s' %
                             (self.dimension, list(update_location.shape)))
        if feature_map.ndim != self.dimension + 1:
            raise ValueError('Expected feature_map to have shape [Spatial_1, Spatial_2, ..., C]. Got size %s' %
                             list(feature_map.shape))
