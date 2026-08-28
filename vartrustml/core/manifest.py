"""Manifest generation for output artifact tracking.

:class:`ManifestGenerator` writes a manifest.json listing every output
artifact with its size and SHA256 checksum, for reproducibility and for
Nextflow integration.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from vartrustml._version import __version__
from vartrustml.config import ExperimentConfig

logger = logging.getLogger(__name__)


class ManifestGenerator:
    """Generate manifest.json files for output artifact tracking.

    Creates manifest files listing all output files with their sizes
    and SHA256 checksums for reproducibility and Nextflow integration.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration (used for metadata in manifest).
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def generate(self, output_dir: Path, dataset_name: str) -> Path:
        """Generate manifest.json with all output artifacts.

        Parameters
        ----------
        output_dir : Path
            Directory containing output files.
        dataset_name : str
            Name of the dataset for metadata.

        Returns
        -------
        Path
            Path to the generated manifest.json file.
        """
        manifest: Dict[str, Any] = {
            "vartrustml_version": __version__,
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "files": [],
        }

        # Collect all output files
        for file_path in output_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "manifest.json":
                rel_path = file_path.relative_to(output_dir)

                # Calculate SHA256 checksum
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha256_hash.update(chunk)

                manifest["files"].append(
                    {
                        "path": str(rel_path),
                        "size_bytes": file_path.stat().st_size,
                        "sha256": sha256_hash.hexdigest(),
                    }
                )

        # Sort files by path for consistent ordering
        manifest["files"].sort(key=lambda x: x["path"])
        manifest["total_files"] = len(manifest["files"])
        manifest["total_size_bytes"] = sum(f["size_bytes"] for f in manifest["files"])

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Manifest generated: {len(manifest['files'])} files, "
            f"{manifest['total_size_bytes'] / 1024 / 1024:.2f} MB total"
        )

        return manifest_path
