"""
Reproducibility utilities for ML experiments.

Provides seed management, library version tracking, and hardware
information collection for experiment reproducibility.
"""

import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """Set all random seeds for full reproducibility.

    Sets seeds for Python's random module, NumPy, PyTorch (if available),
    and environment variables. This ensures reproducible experiments across
    runs when combined with deterministic algorithms.

    Parameters
    ----------
    seed : int
        The random seed to use for all random number generators.
    deterministic : bool, default=True
        If True, enables PyTorch deterministic mode (may impact performance).
        Recommended for reproducibility but may slow down training.

    Returns
    -------
    dict
        Dictionary containing information about what was seeded:
        - 'seed': The seed value used
        - 'python_random': True if Python random was seeded
        - 'numpy': True if NumPy was seeded
        - 'torch': True if PyTorch was seeded (False if not installed)
        - 'torch_deterministic': True if PyTorch deterministic mode enabled
        - 'pythonhashseed': True if PYTHONHASHSEED was set

    Notes
    -----
    For full reproducibility, this function should be called at the very
    beginning of any experiment, before any random operations.

    Setting ``deterministic=True`` enables ``torch.use_deterministic_algorithms(True)``,
    which ensures that PyTorch operations are deterministic but may raise errors
    for operations that don't have deterministic implementations.

    Examples
    --------
    >>> from vartrustml.utils.reproducibility import set_all_seeds
    >>> info = set_all_seeds(42)
    >>> print(f"Seeds set: {info}")
    Seeds set: {'seed': 42, 'python_random': True, 'numpy': True, ...}

    References
    ----------
    - PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    """
    result = {
        "seed": seed,
        "python_random": False,
        "numpy": False,
        "torch": False,
        "torch_deterministic": False,
        "pythonhashseed": False,
    }

    # Set PYTHONHASHSEED (affects hash randomization)
    os.environ["PYTHONHASHSEED"] = str(seed)
    result["pythonhashseed"] = True

    # Python's built-in random
    random.seed(seed)
    result["python_random"] = True

    # NumPy legacy global RNG - intentional: sklearn internals still use it
    np.random.seed(seed)
    result["numpy"] = True
    result["numpy_rng"] = np.random.default_rng(seed)

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            # Ensure deterministic behavior for CUDA operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        result["torch"] = True

        if deterministic:
            # This will raise an error if a non-deterministic operation is used
            # Set to warn mode instead of error to avoid crashes
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                result["torch_deterministic"] = True
            except TypeError:
                # Older PyTorch versions don't have warn_only parameter
                torch.use_deterministic_algorithms(True)
                result["torch_deterministic"] = True

        logger.info(f"All random seeds set to {seed} (PyTorch available)")
    except ImportError:
        logger.info(f"Random seeds set to {seed} (PyTorch not available)")

    return result


def get_library_versions() -> Dict[str, str]:
    """Get versions of key libraries for reproducibility logging.

    Returns a dictionary of library names to version strings for all
    relevant scientific Python packages used in the ML pipeline.

    Returns
    -------
    dict of {str: str}
        Dictionary mapping library names to version strings.
        Libraries that are not installed will have value "not installed".

    Examples
    --------
    >>> from vartrustml.utils.reproducibility import get_library_versions
    >>> versions = get_library_versions()
    >>> print(versions['numpy'])
    1.24.3
    """
    versions = {}

    # Core scientific libraries
    libraries = [
        ("python", None),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "sklearn"),
        ("xgboost", "xgboost"),
        ("catboost", "catboost"),
        ("torch", "torch"),
        ("shap", "shap"),
        ("optuna", "optuna"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]

    # Get Python version
    import sys

    versions["python"] = sys.version.split()[0]

    for name, module_name in libraries:
        if module_name is None:
            continue
        try:
            module = __import__(module_name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not installed"

    return versions


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information for reproducibility tracking.

    Collects CPU, GPU, and system information to help diagnose
    reproducibility issues across different environments.

    Returns
    -------
    dict
        Dictionary containing:
        - 'cpu': CPU identifier
        - 'cpu_count': Number of CPU cores
        - 'platform': OS platform
        - 'gpu': GPU name (if CUDA available)
        - 'cuda_version': CUDA version (if available)
        - 'gpu_count': Number of GPUs (if available)

    Examples
    --------
    >>> from vartrustml.utils.reproducibility import get_hardware_info
    >>> info = get_hardware_info()
    >>> print(info['cpu'])
    """
    import platform

    hardware = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
    }

    # Try to get GPU info via torch
    try:
        import torch

        if torch.cuda.is_available():
            hardware["gpu"] = torch.cuda.get_device_name(0)
            hardware["cuda_version"] = torch.version.cuda
            hardware["gpu_count"] = torch.cuda.device_count()
        else:
            hardware["gpu"] = None
            hardware["cuda_version"] = None
            hardware["gpu_count"] = 0
    except ImportError:
        hardware["gpu"] = None
        hardware["cuda_version"] = None
        hardware["gpu_count"] = 0

    return hardware


def log_reproducibility_info(
    seed: int, output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Log reproducibility information including seeds and library versions.

    Convenience function that sets all seeds and logs library versions.
    Optionally saves this information to a file for experiment tracking.

    Parameters
    ----------
    seed : int
        The random seed to use.
    output_path : Path, optional
        If provided, saves reproducibility info to this file (JSON format).

    Returns
    -------
    dict
        Dictionary containing:
        - 'seed_info': Result from set_all_seeds()
        - 'library_versions': Result from get_library_versions()
        - 'hardware_info': Result from get_hardware_info()
        - 'timestamp': ISO format timestamp

    Examples
    --------
    >>> from vartrustml.utils.reproducibility import log_reproducibility_info
    >>> info = log_reproducibility_info(42, Path("./reproducibility.json"))
    """
    import json

    from vartrustml.utils.serialization import np_encoder

    # Set all seeds
    seed_info = set_all_seeds(seed)

    # Get library versions
    library_versions = get_library_versions()

    # Get hardware info
    hardware_info = get_hardware_info()

    # Assemble the reproducibility record
    repro_info = {
        "seed_info": seed_info,
        "library_versions": library_versions,
        "hardware_info": hardware_info,
        "timestamp": datetime.now().isoformat(),
    }

    # Log summary
    logger.info(
        f"Reproducibility: seed={seed}, numpy={library_versions.get('numpy')}, "
        f"sklearn={library_versions.get('sklearn')}, torch={library_versions.get('torch')}"
    )

    # Save to file if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(repro_info, f, indent=2, default=np_encoder)
        logger.info(f"Reproducibility info saved to: {output_path}")

    return repro_info
