import numpy as np
from async_sparse import RuleBook, AsynSparseConvolution2D
from time import perf_counter


class asynSparseConvolution2Dcpp:
    def __init__(self, dimension, nIn, nOut, filter_size, first_layer=False, use_bias=False, debug=False):
        self.conv = AsynSparseConvolution2D(dimension, nIn, nOut, filter_size, first_layer, use_bias, debug)
        
        self.nIn = nIn
        self.nOut = nOut
        self.filter_size = filter_size
        self.dimension = dimension
        self.first_layer = first_layer
    
    def setParameters(self, weights, bias):
        weights = weights.reshape(self.filter_size ** self.dimension, self.nIn * self.nOut).astype("float32")
        bias = bias.reshape((-1, 1)).astype("float32")
        self.conv.setParameters(bias, weights)
        
    def forward(self, update_location, feature_map, active_sites_map=None, rule_book=None):
        spatial_dimension = tuple(feature_map.shape[:2])
        H, W = spatial_dimension

        no_updates = False
        if update_location.shape[0] == 0:
            no_updates = True
            update_location = np.zeros([1, 2])
        # wrap update locations in fortran array
        update_location = np.array(update_location, dtype="int32")
        update_location = np.asfortranarray(update_location)

        # prepare input feature map
        feature_map = np.asfortranarray(feature_map)
        feature_map = feature_map.reshape((-1, self.nIn)).astype("float32")

        if self.first_layer:
            self.conv.initMaps(H, W)
            active_sites_map = self.conv.initActiveMap(feature_map, update_location)
            rule_book = RuleBook(H, W, self.filter_size, self.dimension)
        else:
            self.conv.initMaps(H, W)
            # active_sites_map = self.conv.initActiveMap(feature_map, update_location)
            active_sites_map = active_sites_map.flatten()

        # t1 = perf_counter()
        new_update_locations, output_map, active_sites_map = self.conv.forward(update_location,
                                                                               feature_map,
                                                                               active_sites_map,
                                                                               rule_book,
                                                                               no_updates)
        # dt = perf_counter() - t1
        # print("CPP implementation: ", dt)

        output_map = output_map.reshape((H, W, self.nOut))
        active_sites_map = active_sites_map.reshape(spatial_dimension)

        return new_update_locations, output_map, active_sites_map, rule_book