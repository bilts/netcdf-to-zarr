import numpy as np
import zarr
from netCDF4 import Dataset
from pathlib import Path
from itertools import chain
from concurrent.futures import ProcessPoolExecutor

# Convert NetCDF files to Zarr store


def netcdf_to_zarr(datasets, store, append_axis=None):

    root = zarr.group(store=store, overwrite=True)

    for i, dname in enumerate(datasets):
        for j, gname in enumerate(__get_groups(dname)):
            if j == 0:
                group = root
                ds = ''
            else:
                group = __set_group(gname, root)
                ds = dname
            if i == 0:
                __set_meta(ds + gname, group)
                __set_vars_serial(ds + gname, group)
            else:
                __append_vars_serial(gname, group, append_axis)


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

    tok = Path(ds).parts

    if len(tok) < 2:
        return Dataset(tok[0], *args, **kwargs)
    else:
        return Dataset(tok[0], *args, **kwargs)['/'.join(tok[1:])]


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


def __set_var(ds, store, name):

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


# Append data to existing variable


def __append_var(ds, store, name, dim):
    print("Appending " + name + " from " + ds)
    dataset = __nc_open(ds)
    var = dataset.variables[name]
    print('dim: ' + dim, 'vdim: ' + var.dimensions)
    if dim in var.dimensions:
        axis = group[name].attrs['dimensions'].index(dim)
        store[name].append(var, axis)


# NOTE: zarr parallel writes fails  on High Sierra 10.13.6


# serial execution


def __set_vars_serial(ds, store):

    print("Setting variables for: " + ds)
    dataset = __nc_open(ds)

    for name in dataset.variables.keys():
        __set_var(ds, store, name)


def __append_vars_serial(ds, store, dim):

    print("Append vars")
    dataset = __nc_open(ds)

    store[dim].append(dataset[dim])

    for name in dataset.variables.keys():
        __append_var(ds, store, name, dim)


# parallel execution


def __set_vars_parallel(ds, store):

    with ProcessPoolExecutor(max_workers=8) as executor:
        for name in dataset.variables.keys():
            executor.submit(__set_var, ds, store, name)


def __append_vars_parallel(ds, store, dim):

    print("Append vars")
    dataset = __nc_open(ds)

    store[dim].append(dataset[dim])

    with ProcessPoolExecutor(max_workers=8) as executor:
        for name in dataset.variables.keys():
            executor.submit(__append_var, ds, store, name, dim)
