import zarr
import re
from utils.encoder import json_encode
from netCDF4 import Dataset
from itertools import chain
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor


# shared memory path
SHARED = '/dev/shm/'


"""
Convert NetCDF files to Zarr store
N.B. To use threads or processes a thread/process safe version
of hdf5 (underlying netcdf4) is required.
this is because even if only concurrent reads on different data are issued,
 the HDF5 library modifies global data structures that are independent
 of a particular HDF5 dataset or HDF5 file.
 HDF5 relies on a semaphore around the library API calls in the thread-safe
 version of the library to protect the data structure from corruption
 by simultaneous manipulation from different threads.
"""


def netcdf_to_zarr(src, dst, axis=None, mode='serial', nested=False):
    """Summary

    Args:
        src (TYPE): Description
        dst (TYPE): Description
        axis (None, optional): Description
        mode (str, optional): Description
        nested (bool, optional): Description
    """
    if isinstance(dst, str):
        if nested:
            local_store = zarr.NestedDirectoryStore(dst)
        else:
            local_store = zarr.DirectoryStore(dst)
    else:
        local_store = dst

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


# Open netcdf files and groups with the same interface


def __nc_open(ds, *args, **kwargs):
    base, ext, path = re.split(r'(\.nc|\.hdf5)', ds, maxsplit=1, flags=re.IGNORECASE)

    filename = base + ext
    if path == '':
        return Dataset(filename, *args, **kwargs)
    else:
        return Dataset(filename, *args, **kwargs)[path]


# Return serielizable attributes as dicts


def __get_meta(dataset):

    # JSON encode attributes so they can be serialized
    return {key: json_encode(getattr(dataset, key)) for key in dataset.ncattrs()}


# Return chunking informations about the given variable


def __get_chunks(var):

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
    store.attrs.put(__get_meta(__nc_open(ds)))


# Set variable data, including dimensions and metadata


def __set_var(ds, store, name, syncro=None):

    print("Setting variable " + name)

    dataset = __nc_open(ds)
    var = dataset.variables[name]

    store.create_dataset(name,
                         data=var,
                         shape=var.shape,
                         chunks=(__get_chunks(var)),
                         dtype=var.dtype,
                         synchronizer=syncro
                         )
    attrs = __get_meta(dataset)
    attrs['dimensions'] = list(var.dimensions)
    store[name].attrs.put(attrs)


# Append data to existing variable


def __append_var(ds, store, name, dim, syncro=None):

    print("Appending " + name + " from " + ds)

    dataset = __nc_open(ds)
    var = dataset.variables[name]

    if dim in var.dimensions:
        axis = store[name].attrs['dimensions'].index(dim)
        array = zarr.open_array(store=store[name],
                                mode='r+',
                                synchronizer=syncro
                                )
        array.append(var, axis)


# setting executor


def __set_vars(ds, store, mode='serial'):

    print("Setting variables for: " + ds)
    dataset = __nc_open(ds)

    if mode == 'serial':
        for name in dataset.variables.keys():
            __set_var(ds, store, name)

    elif mode == 'processes':
        with ProcessPoolExecutor(max_workers=8) as executor:
            syncro = zarr.ProcessSynchronizer(SHARED + 'ntz.sync')
            for name in dataset.variables.keys():
                executor.submit(__set_var, ds, store, name, syncro)

    elif mode == 'threads':
        with ThreadPoolExecutor(max_workers=8) as executor:
            syncro = zarr.ThreadSynchronizer()
            for name in dataset.variables.keys():
                executor.submit(__set_var, ds, store, name, syncro)

    else:
        raise ValueError('the mode %s is not valid.' % mode)


# appending executor


def __append_vars(ds, store, dim, mode='serial'):

    print("Append vars")
    dataset = __nc_open(ds)

    store[dim].append(dataset[dim])

    if mode == 'serial':
        for name in dataset.variables.keys():
            __append_var(ds, store, name, dim)

    elif mode == 'processes':
        with ProcessPoolExecutor(max_workers=8) as executor:
            syncro = zarr.ProcessSynchronizer(SHARED + 'ntz.sync')
            for name in dataset.variables.keys():
                executor.submit(__append_var, ds, store, name, dim, syncro)

    elif mode == 'threads':
        with ThreadPoolExecutor(max_workers=8) as executor:
            syncro = zarr.ThreadSynchronizer()
            for name in dataset.variables.keys():
                executor.submit(__append_var, ds, store, name, dim, syncro)

    else:
        raise ValueError('the mode %s is not valid.' % mode)


# main


def main(args):
    import utils.parser as p
    args = p.configure(args)
    netcdf_to_zarr(**vars(args))


if __name__ == '__main__':

    '''
    Usage:
    python convert.py 'src1.nc src2.nc...srcN.nc' dst.zarr [-options]
    '''

    import sys
    main(sys.argv[1:])
