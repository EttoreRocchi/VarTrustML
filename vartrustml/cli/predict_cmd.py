"""
Prediction CLI command for VarTrustML.

Contains:
- predict: Make predictions with a saved model
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from vartrustml import DataLoader
from vartrustml.core.train_model import ModelTrainer

from vartrustml.cli.main import app


@app.command("predict", rich_help_panel="Model Application")
def predict(
    model_path: Path = typer.Argument(..., help="Path to saved model (.joblib)."),
    data: Path = typer.Argument(..., help="Path to data file for predictions."),
    output: Path = typer.Option(
        Path("predictions.csv"),
        "--output",
        "-o",
        help="Where to save predictions.",
        rich_help_panel="Output",
    ),
    proba: bool = typer.Option(
        False,
        "--proba",
        help="Output probabilities along with predicted classes.",
        rich_help_panel="Prediction Options",
    ),
    default_threshold: bool = typer.Option(
        False,
        "--default-threshold",
        help="Use default 0.5 threshold instead of model's optimized threshold.",
        rich_help_panel="Prediction Options",
    ),
    id_column: Optional[str] = typer.Option(
        None,
        "--id-column",
        help="Column to use as row identifier in output (preserved as index).",
        rich_help_panel="Data Options",
    ),
):
    """
    Make predictions with a saved model.

    If the model was trained with --optimize-threshold, the optimized threshold
    is used by default. Use --default-threshold to force the standard 0.5 threshold.
    """
    typer.echo(f"Loading model from {model_path}")
    model_data = ModelTrainer.load_model(str(model_path))

    # Extract model and threshold metadata
    model = model_data["model"]
    optimal_threshold = model_data.get("optimal_threshold", 0.5)
    threshold_metadata = model_data.get("threshold_metadata")

    typer.echo(f"Loading data from {data}")
    loader = DataLoader(str(data.parent))
    df = loader.load_dataset(data.name, drop_duplicates=False, id_column=id_column)
    X = df.select_dtypes(include=[np.number])
    typer.echo(f"Making predictions on {len(X)} samples")

    if proba:
        preds = model.predict_proba(X)
        prob_class_1 = preds[:, 1]

        # Determine threshold to use for predicted class
        if default_threshold or threshold_metadata is None:
            threshold_to_use = 0.5
        else:
            threshold_to_use = optimal_threshold

        out_df = pd.DataFrame(
            {
                "prob_class_0": preds[:, 0],
                "prob_class_1": prob_class_1,
                "predicted_class": (prob_class_1 >= threshold_to_use).astype(int),
            }
        )

        # Add threshold info and report which threshold is used
        if threshold_metadata is not None and not default_threshold:
            out_df["optimal_threshold"] = optimal_threshold
            out_df["prob_diff_to_threshold"] = prob_class_1 - optimal_threshold
            typer.echo(f"Using model's optimized threshold: {optimal_threshold:.4f}")
        else:
            typer.echo("Using default threshold: 0.5")
            if threshold_metadata is not None:
                typer.echo(f"(Model has optimized threshold: {optimal_threshold:.4f})")
    else:
        # Get probabilities first
        y_prob = model.predict_proba(X)

        # Apply threshold: use optimized by default, --default-threshold overrides to 0.5
        if default_threshold or threshold_metadata is None:
            # Use default 0.5 threshold
            preds = model.predict(X)
            if threshold_metadata is not None:
                typer.echo(
                    f"Using default threshold (0.5). Model has optimized threshold "
                    f"{optimal_threshold:.4f}."
                )
        else:
            # Use optimized threshold (default when model has one)
            preds = (y_prob[:, 1] >= optimal_threshold).astype(int)
            typer.echo(f"Using model's optimized threshold: {optimal_threshold:.4f}")

        out_df = pd.DataFrame({"prediction": preds})

    out_df.index = df.index
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output, index=True)
    typer.echo(f"Predictions saved to {output}")
