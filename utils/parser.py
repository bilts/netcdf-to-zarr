# parser for netcdf_to_zarr

import argparse


def configure(args):

    class toList(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs not allowed")
            super(toList, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values.split(' '))

    desc = ('This script converts netCDF4 into zarr files\n'
            'It is possible to specify a list of netCDF4 files\n'
            'to merge them into a single zarr file.\n'
            'to do so, the variables in the netCD4 file\n'
            'must have a common shape and run along the same dimension.\n'
            '\nExamples:\n'
            '  -convert.py src.nc dst.nc [-options]\n'
            '  -convert.py "src1.nc src2.nc src3.nc" dst.zarr [-options]\n')

    parser = argparse.ArgumentParser(description=desc,
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
                        help='use processes or threads',
                        choices=('threads', 'processes'),
                        default='serial')

    return parser.parse_args(args)
