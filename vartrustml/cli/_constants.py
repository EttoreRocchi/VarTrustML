"""
Shared constants for the VarTrustML CLI.

Exit codes follow nf-core conventions for Nextflow integration.
"""

# Exit code constants for nf-core compatibility
EXIT_ERROR = 1
EXIT_VALIDATION_ERROR = 2
EXIT_INTERRUPTED = 130  # 128 + SIGINT(2)
EXIT_TERMINATED = 143  # 128 + SIGTERM(15)
