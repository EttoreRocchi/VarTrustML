"""
VartrustML Command-Line Interface (CLI).

Exit Codes
----------
0 : Success
1 : Runtime error (general failure)
2 : Validation error (configuration or input validation failed)
130 : Interrupted (SIGINT/Ctrl+C)
143 : Terminated (SIGTERM)
"""

import signal

import typer

from vartrustml.cli._constants import (
    EXIT_INTERRUPTED,
    EXIT_TERMINATED,
)


def _signal_handler(signum: int, frame) -> None:
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    typer.echo(f"\n{signal_name} received. Shutting down gracefully...", err=True)
    # Exit with appropriate code
    if signum == signal.SIGTERM:
        raise SystemExit(EXIT_TERMINATED)
    else:
        raise SystemExit(EXIT_INTERRUPTED)


_app_ctx = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    add_completion=False,
    help="VartrustML: reliable variant calling through machine learning predictions. "
    + "Compare, train and test ML models.",
    context_settings=_app_ctx,
)


@app.callback(invoke_without_command=True)
def _app_callback(ctx: typer.Context) -> None:
    """Register signal handlers at runtime (not import time)."""
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# Import sub-command modules to register commands on the app
import vartrustml.cli.ablation_cmd  # noqa: F401, E402
import vartrustml.cli.compare_models_cmd  # noqa: F401, E402
import vartrustml.cli.cross_dataset_cmd  # noqa: F401, E402
import vartrustml.cli.evaluate_cmd  # noqa: F401, E402
import vartrustml.cli.predict_cmd  # noqa: F401, E402
import vartrustml.cli.train_cmd  # noqa: F401, E402
import vartrustml.cli.utils_cmd  # noqa: F401, E402


if __name__ == "__main__":
    app()
