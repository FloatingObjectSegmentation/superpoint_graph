import h5py
import numpy as np

def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for path, _ in h5py_dataset_iterator(f):
            yield path



filename = '/media/km/ad02048a-21c3-4454-b1b4-58c5a99df3c5/workspace/results/predictions_val.h5'
confmatfile = '/media/km/ad02048a-21c3-4454-b1b4-58c5a99df3c5/workspace/results/pointwise_cm.npy'

with h5py.File(filename, 'r') as f:
    for dset in traverse_datasets(filename):
        print('Path: ', dset)
        print('Shape: ', f[dset].shape)
        print('Data type: ', f[dset].dtype)

A = np.load(confmatfile)
print(A)