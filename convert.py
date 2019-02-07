import numpy as np
import zarr


from netCDF4 import Dataset
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor


# shared memory path
SHARED = '/dev/shm/'


# Convert NetCDF files to Zarr store
# N.B. To use threads or processes
# a thread safe and process safe version of hdf5 (underlying of netcdf4) is required.
# this is because even if only concurrent reads on different data are issued,
# the HDF5 library modifies global data structures that are independent
# of a particular HDF5 dataset or HDF5 file.
# HDF5 relies on a semaphore around the library API calls in the thread-safe version of the library
# to protect the data structure from corruption by simultaneous manipulation from different threads.


def netcdf_to_zarr(src, dst, axis=None, mode='serial', nested=False):

    if nested:
        local_store = zarr.NestedDirectoryStore(dst)
    else:
        local_store = zarr.DirectoryStore(dst)

    root = zarr.group(store=local_store, overwrite=True)

    for i, dname in enumerate(src):
        # cycling over groups, the first one is the root.
        for j, gname in enumerate(__get_groups(dname)):
            if j == 0:
                group = root
                ds = ''
            else:
                group = __set_group(gname, root)
                ds = dname
            if i == 0:
                __set_meta(ds + gname, group)
                __set_vars(ds + gname, group, mode)
            else:
                __append_vars(gname, group, axis, mode)


# Convert non-json-encodable types to built-in types


def __json_encode(val):

    if isinstance(val, np.integer):
        return int(val)
    elif isinstance(val, np.floating):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    else:
        return val


# Open netcdf files and groups with the same interface


def __nc_open(ds, *args, **kwargs):

    tok = ds.split('.nc')

    if not tok[1]:
        return Dataset(tok[0] + '.nc', *args, **kwargs)
    else:
        return Dataset(tok[0] + '.nc', *args, **kwargs)[tok[1]]


# Return serielizable attributes as dicts


def __dsattrs(dataset):

    # JSON encode attributes so they can be serialized
    return {key: __json_encode(getattr(dataset, key)) for key in dataset.ncattrs()}


# Return chunking informations about the given variable


def __get_var_chunks(var):

    if var.chunking() != 'contiguous':
        return tuple(var.chunking())

    return None


# Returns an iterator with every group of the file (root included)


def __get_groups(ds):

    # recursive version with generator
    def walktree(top):
        values = top.groups.values()
        if values:
            for v in values:
                yield v.path

        for value in values:
            for children in walktree(value):
                yield children

    dataset = Dataset(ds)

    grps = (g for g in walktree(dataset))

    return chain((ds,), grps)


# Set file group


def __set_group(ds, store):

    print('creating group: ' + ds)
    return store.create_group(ds)


# Set file metadata


def __set_meta(ds, store):

    print("Setting meta for: " + ds)
    store.attrs.put(__dsattrs(__nc_open(ds)))


# Set variable data, including dimensions and metadata


# serial access
def __set_var_s(ds, store, name):

    print("Setting variable " + name)
    dataset = __nc_open(ds)
    var = dataset.variables[name]

    store.create_dataset(name,
                         data=var,
                         shape=var.shape,
                         chunks=(__get_var_chunks(var)),
                         dtype=var.dtype
                         )
    attrs = __dsattrs(var)
    attrs['dimensions'] = list(var.dimensions)
    store[name].attrs.put(attrs)


# process access
def __set_var_p(ds, store, name):

    print("Setting variable " + name)

    dataset = __nc_open(ds)
    var = dataset.variables[name]
    syncro = zarr.ProcessSynchronizer(SHARED + 'ntz.sync')

    store.create_dataset(name,
                         data=var,
                         shape=var.shape,
                         chunks=(__get_var_chunks(var)),
                         dtype=var.dtype,
                         synchronizer=syncro
                         )
    attrs = __dsattrs(var)
    attrs['dimensions'] = list(var.dimensions)
    store[name].attrs.put(attrs)


# thread access
def __set_var_t(ds, store, name):

    print("Setting variable " + name)

    dataset = __nc_open(ds)
    var = dataset.variables[name]
    syncro = zarr.ThreadSynchronizer()

    store.create_dataset(name,
                         data=var,
                         shape=var.shape,
                         chunks=(__get_var_chunks(var)),
                         dtype=var.dtype,
                         synchronizer=syncro
                         )
    attrs = __dsattrs(var)
    attrs['dimensions'] = list(var.dimensions)
    store[name].attrs.put(attrs)


# Append data to existing variable

# serial access
def __append_var_s(ds, store, name, dim):
    print("Appending " + name + " from " + ds)
    dataset = __nc_open(ds)
    var = dataset.variables[name]

    if dim in var.dimensions:
        axis = store[name].attrs['dimensions'].index(dim)
        store[name].append(var, axis)


# process access
def __append_var_p(ds, store, name, dim):
    print("Appending " + name + " from " + ds)
    dataset = __nc_open(ds)
    var = dataset.variables[name]
    syncro = zarr.ProcessSynchronizer(SHARED + 'ntz.sync')

    if dim in var.dimensions:
        axis = store[name].attrs['dimensions'].index(dim)
        array = zarr.open_array(store=store[name],
                                mode='r+',
                                synchronizer=syncro)
        array.append(var, axis)


# thread access
def __append_var_t(ds, store, name, dim):
    print("Appending " + name + " from " + ds)
    dataset = __nc_open(ds)
    var = dataset.variables[name]
    syncro = zarr.ThreadSynchronizer()

    if dim in var.dimensions:
        axis = store[name].attrs['dimensions'].index(dim)
        array = zarr.open_array(store=store[name],
                                mode='r+',
                                synchronizer=syncro)
        array.append(var, axis)


# setting executor
def __set_vars(ds, store, mode='serial'):

    print("Setting variables for: " + ds)
    dataset = __nc_open(ds)

    if mode == 'serial':
        for name in dataset.variables.keys():
            __set_var_s(ds, store, name)
    elif mode == 'threads':
        with ProcessPoolExecutor(max_workers=8) as executor:
            for name in dataset.variables.keys():
                executor.submit(__set_var_t, ds, store, name)
    elif mode == 'processes':
        with ThreadPoolExecutor(max_workers=8) as executor:
            for name in dataset.variables.keys():
                executor.submit(__set_var_p, ds, store, name)
    else:
        raise ValueError('the mode %s is not valid.' % mode)


# appending executor
def __append_vars(ds, store, dim, mode='serial'):

    print("Append vars")
    dataset = __nc_open(ds)

    store[dim].append(dataset[dim])

    if mode == 'serial':
        for name in dataset.variables.keys():
            __append_var_s(ds, store, name, dim)
    elif mode == 'threads':
        with ProcessPoolExecutor(max_workers=8) as executor:
            for name in dataset.variables.keys():
                executor.submit(__append_var_t, ds, store, name, dim)
    elif mode == 'processes':
        with ThreadPoolExecutor(max_workers=8) as executor:
            for name in dataset.variables.keys():
                executor.submit(__append_var_p, ds, store, name, dim)
    else:
        raise ValueError('the mode %s is not valid.' % mode)


def __configure(args):

    import argparse

    class toList(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs not allowed")
            super(toList, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values.split(' '))

    parser = argparse.ArgumentParser(description=('This script converts netCDF4 into zarr files\n'
                                                  'It is possible to specify a list of netCDF4 files\n'
                                                  'to merge them into a single zarr file.\n'
                                                  'to do so, the variables in the netCD4 file\n'
                                                  'must have a common shape and run along the same dimension.\n'
                                                  '\nExamples:\n'
                                                  '  -convert.py src.nc dst.nc [-options]\n'
                                                  '  -convert.py "src1.nc src2.nc src3.nc" dst.zarr [-options]\n'),
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('src',
                        help='source netCDF4 file/files',
                        action=toList,
                        default=[])

    parser.add_argument('dst',
                        help='target zarr file',
                        default='')

    parser.add_argument('-a',
                        '--axis',
                        metavar='axis',
                        help='axis name to append along',
                        default=None)

    parser.add_argument('-n',
                        '--nested',
                        help='chunks are located into a nested path',
                        action='store_true',
                        default=False)

    parser.add_argument('-m',
                        '--mode',
                        metavar='mode',
                        help='transfer size for multipart',
                        choices=('threads', 'processes'),
                        default='serial')

    return parser.parse_args(args)


def main(args):
    args = __configure(args)
    netcdf_to_zarr(**vars(args))


if __name__ == '__main__':

    '''
    Usage:
    python convert.py 'src1.nc src2.nc...srcN.nc' dst.zarr [-options]
    '''

    import sys
    main(sys.argv[1:])
