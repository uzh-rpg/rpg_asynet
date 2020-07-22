from async_sparse import RuleBook, AsynSparseConvolution2D

H, W = 180,240
k = 3
dimension = 2
nIn = 2
nOut = 16
first_layer = True
use_bias = True

rulebook = RuleBook(H, W, 3, 2)
conv = AsynSparseConvolution2D(dimension, nIn, nOut, k, first_layer, use_bias)