rm -rf async_sparse.egg-info/ build/ dist/ tmp/; pip uninstall async_sparse -y; python setup.py install
python ../unittests/sparse_conv2D_cpp_test.py
