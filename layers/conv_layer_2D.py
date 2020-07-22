import torch

from layers.site_enum import Sites


class asynSparseConvolution2D:
    def __init__(self, dimension, nIn, nOut, filter_size, first_layer=False, use_bias=False,
                 device=torch.device('cpu')):
        """
        Constructs a convolution layer.

        :param dimension: spatial dimension of the convolution e.g. dimension=2 leads to kxk kernel
        :param nIn: number of channels in the input features
        :param nOut: number of output channels
        :param filter_size: kernel size
        :param first_layer: bool indicating if it is the first layer. Used for computing new inactive sites.
        """
        self.dimension = dimension
        self.nIn = nIn
        self.nOut = nOut
        self.filter_size = filter_size
        self.device = device

        self.filter_size_tensor = torch.LongTensor(dimension).fill_(self.filter_size).to(device)
        self.filter_volume = self.filter_size_tensor.prod().item()
        std = (2.0 / nIn / self.filter_volume)**0.5
        self.weight = torch.nn.Parameter(torch.Tensor(self.filter_volume, nIn, nOut).normal_(0, std)).to(device)
        self.first_layer = first_layer
        self.use_bias = use_bias
        if use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(nOut).normal_(0, std)).to(device)

        self.padding = [filter_size // 2] * dimension * 2
        # Construct lookup table for 1d kernel position to position in nd
        kernel_indices = torch.stack(torch.meshgrid([torch.arange(filter_size) for _ in range(dimension)]), dim=-1)
        self.kernel_indices = kernel_indices.reshape([self.filter_volume, dimension]).to(device)

        self.output_feature_map = None
        self.old_input_feature_map = None

    def forward(self, update_location, feature_map, active_sites_map=None, rule_book_input=None, rule_book_output=None):
        """
        Computes the a asynchronous sparse convolution layer based on the update_location

        :param update_location: tensor with shape [N_active, dimension]
        :param feature_map: tensor with shape [N, nIn]
        :param active_sites_map: tensor with shape [N]. Includes 1 - active side and 2 - location stored in Rulebooks,
                                  3 - new active site in input. 4 - new inactive site
        :param rule_book_input: list containing #(kernel_size) lists with locations in the input
        :param rule_book_output: list containing #(kernel_size) lists with locations in the output
        :return:
        """
        self.checkInputArguments(update_location, feature_map, active_sites_map)
        spatial_dimension = list(feature_map.shape[:-1])
        update_location_indices = update_location.split(1, dim=-1)

        if rule_book_input is None:
            rule_book_input = [[] for _ in range(self.filter_size ** self.dimension)]
            rule_book_output = [[] for _ in range(self.filter_size ** self.dimension)]
        if self.output_feature_map is None:
            self.output_feature_map = torch.zeros(spatial_dimension + [self.nOut]).to(self.device)
        if self.old_input_feature_map is None:
            self.old_input_feature_map = torch.zeros(spatial_dimension + [self.nIn]).to(self.device)
        if active_sites_map is None:
            active_sites_map = torch.squeeze((torch.sum(feature_map**2, dim=-1) != 0).float()).to(self.device)
            # Catch case if input feature is reduced to zero
            active_sites_map[update_location_indices] = Sites.ACTIVE_SITE.value

        if self.first_layer:
            # Set new deactivate sites to Sites.NEW_INACTIVE_SITE
            zero_input_update = torch.squeeze(torch.sum(feature_map[update_location_indices] ** 2, dim=-1) == 0, dim=-1)
            bool_new_active_site = (self.old_input_feature_map[update_location_indices] ** 2).sum(-1).squeeze(-1) == 0
        else:
            zero_input_update = None
            bool_new_active_site = torch.zeros([update_location.shape[0]]).bool()

        if update_location.nelement() != 0:
            out = self.updateRuleBooks(active_sites_map, update_location, bool_new_active_site, zero_input_update,
                                       rule_book_input, rule_book_output, update_location_indices)
            rule_book_input, rule_book_output, new_update_events, active_sites_map = out

            # New update sites for next layer
            if len(new_update_events) == 0:
                new_update_events = torch.empty([0, self.dimension])
            else:
                new_update_events = torch.stack(new_update_events, dim=0)

        else:
            new_update_events = torch.empty([0, self.dimension])

        # Compute update step with the rule book
        # Change to float64 for numerical stability
        feature_map = feature_map.double()
        old_input_feature_map = self.old_input_feature_map.double()
        output_feature_map = self.output_feature_map.double()

        # Create vector for ravel the 2D indices to 1D flattened indices
        # Only valid for 2D
        flattened_indices_dim = torch.tensor(feature_map.shape[:-1], device=self.device)
        flattened_indices_dim = torch.cat((flattened_indices_dim[1:],
                                           torch.ones([1], dtype=torch.long, device=self.device)))

        for i_kernel in range(self.filter_volume):
            if len(rule_book_input[i_kernel]) == 0:
                continue

            input_indices = torch.stack(rule_book_input[i_kernel], dim=0).long().split(1, dim=-1)
            output_indices = torch.stack(rule_book_output[i_kernel], dim=0).long().split(1, dim=-1)

            bool_not_new_sites = (active_sites_map[output_indices] != Sites.NEW_ACTIVE_SITE.value).float()

            delta_feature = torch.squeeze(feature_map[input_indices], 1) - \
                            torch.squeeze(old_input_feature_map[input_indices], 1) * bool_not_new_sites
            update_term = torch.matmul(delta_feature[:, None, :],
                                       self.weight[None, i_kernel, :, :].double()).squeeze(dim=1)

            flattend_indices = torch.cat(output_indices, dim=-1)
            flattend_indices = flattend_indices * torch.unsqueeze(flattened_indices_dim, dim=0)
            flattend_indices = flattend_indices.sum(dim=-1)

            output_feature_map = output_feature_map.view([-1, self.nOut])
            # .index_add_ might not work if gradients are needed
            output_feature_map.index_add_(dim=0, index=flattend_indices, source=update_term)
            output_feature_map = output_feature_map.view(spatial_dimension + [self.nOut])

        # Set deactivated update sites in the output to zero, but keep it in the rulebook for the next layers
        output_feature_map = output_feature_map * \
                             torch.unsqueeze((active_sites_map != Sites.NEW_INACTIVE_SITE.value).float(), -1)
        if self.use_bias:
            output_feature_map[active_sites_map == Sites.NEW_ACTIVE_SITE.value] += self.bias

        del self.old_input_feature_map
        del self.output_feature_map
        self.old_input_feature_map = feature_map.clone()
        self.output_feature_map = output_feature_map.clone()

        return new_update_events, output_feature_map, active_sites_map, rule_book_input, rule_book_output

    def updateRuleBooks(self, active_sites_map, update_location, bool_new_active_site, zero_input_update,
                        rule_book_input, rule_book_output, update_location_indices):
        """Updates the rule books used for the weight multiplication"""

        # Pad input to index with kernel
        padded_active_sites_map = torch.nn.functional.pad(active_sites_map, self.padding, mode='constant', value=0)
        shifted_update_location = update_location + (self.filter_size_tensor // 2)[None, :]

        # Compute indices corresponding to the receptive fields of the update location
        kernel_update_location = shifted_update_location[:, None, :] + self.kernel_indices[None, :, :] \
                                    - self.filter_size // 2

        active_sites_to_update = torch.squeeze(padded_active_sites_map[kernel_update_location.split(1, dim=-1)], dim=-1)
        active_sites_to_update = active_sites_to_update.nonzero()

        # Set updated sites to 2
        active_sites_map[update_location_indices] = Sites.UPDATED_SITE.value

        if self.first_layer:
            # Set new deactivate sites to Sites.NEW_INACTIVE_SITE
            active_sites_map[update_location[zero_input_update].split(1, dim=-1)] = Sites.NEW_INACTIVE_SITE.value

        new_update_events = []

        position_kernels = self.filter_volume - 1 - active_sites_to_update[:, 1]
        input_locations = update_location[active_sites_to_update[:, 0], :].clone()
        nd_kernel_positions = self.kernel_indices[position_kernels]
        output_locations = input_locations + self.filter_size // 2 - nd_kernel_positions
        output_location_indices_i, output_location_indices_j = output_locations.split(1, dim=-1)
        output_active_sites = active_sites_map[output_locations[..., 0], output_locations[..., 1]]

        # Compute Rule Book
        i_active_sites = torch.arange(active_sites_to_update.shape[0], dtype=torch.long)
        i_active_sites = i_active_sites[(output_active_sites != Sites.NEW_ACTIVE_SITE.value) & \
                                        (output_active_sites != Sites.NEW_INACTIVE_SITE.value)]
        for i_active_site in i_active_sites:
            output_active_site = active_sites_map[output_location_indices_i[i_active_site],
                                                  output_location_indices_j[i_active_site]]

            rule_book_output[position_kernels[i_active_site]].append(output_locations[i_active_site])
            rule_book_input[position_kernels[i_active_site]].append(input_locations[i_active_site])

            if output_active_site == Sites.ACTIVE_SITE.value:
                new_update_events.append(output_locations[i_active_site])
                active_sites_map[output_location_indices_i[i_active_site],
                                 output_location_indices_j[i_active_site]] = Sites.UPDATED_SITE.value
                active_sites_map[output_locations[i_active_site, ..., 0],
                                output_locations[i_active_site, ..., 1]] = Sites.UPDATED_SITE.value

        # Set newly initialised sites to 3 equal to Sites.NEW_ACTIVE_SITE
        active_sites_map[update_location[bool_new_active_site].split(1, dim=-1)] = Sites.NEW_ACTIVE_SITE.value

        # Update neuron if it is first time active. Exclude points, which influence is propagated at same time step
        padded_active_sites_map[shifted_update_location.split(1, dim=-1)] = 0

        # Update the influence from the active sites in the receptive field, if the site is newly active
        if bool_new_active_site.nelement() != 0:
            # Return if no new activate site are given as input
            if bool_new_active_site.sum() == 0:
                return rule_book_input, rule_book_output, new_update_events, active_sites_map

            new_active_site_influence_indices = kernel_update_location[bool_new_active_site, :].split(1, dim=-1)
            new_active_sites_influence = padded_active_sites_map[new_active_site_influence_indices]
            new_active_sites_influence = new_active_sites_influence.nonzero()

            for i_new_active_site in range(new_active_sites_influence.shape[0]):
                position_kernel = new_active_sites_influence[i_new_active_site, 1]
                output_location = update_location[bool_new_active_site][new_active_sites_influence[i_new_active_site, 0], :]
                input_location = output_location - self.filter_size // 2 + self.kernel_indices[position_kernel]

                rule_book_output[position_kernel].append(output_location.clone())
                rule_book_input[position_kernel].append(input_location)

        return rule_book_input, rule_book_output, new_update_events, active_sites_map

    def checkInputArguments(self, update_location, feature_map, active_sites_map):
        """Checks if the input arguments have the correct shape"""
        if update_location.ndim != 2 or update_location.shape[-1] != self.dimension:
            raise ValueError('Expected update_location to have shape [N, %s]. Got size %s' %
                             (self.dimension, list(update_location.shape)))
        if feature_map.ndim != self.dimension + 1 or feature_map.shape[-1] != self.nIn:
            raise ValueError('Expected feature_map to have shape [Spatial_1, Spatial_2, ..., %s]. Got size %s' %
                             (self.nIn, list(feature_map.shape)))
        if active_sites_map is None:
            return
        if active_sites_map.ndim != self.dimension:
            raise ValueError('Expected active_sites_map to have %s dimensions. Got size %s' %
                             (self.dimension, list(active_sites_map.shape)))

