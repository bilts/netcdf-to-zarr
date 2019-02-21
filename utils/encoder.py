import numpy as np


# Convert non-json-encodable types to built-in types


def json_encode(val):

    if isinstance(val, np.integer):
        return int(val)
    elif isinstance(val, np.floating):
        return float(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    else:
        return val
