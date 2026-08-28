"""
Serialization utilities for numpy types.
"""

from typing import Any

import numpy as np


def np_encoder(obj: Any) -> Any:
    """JSON encoder for numpy types.

    Use with ``json.dumps(data, default=np_encoder)``.

    Parameters
    ----------
    obj : Any
        Object to encode.

    Returns
    -------
    Any
        JSON-serializable version of the object.

    Raises
    ------
    TypeError
        If object type is not supported.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.random.Generator):
        return obj.bit_generator.state
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
