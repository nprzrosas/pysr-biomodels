"""Plot held-out generalization results for frozen Selector v1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "selector_v1_held_out_test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "selector_v1_held_out_figures"
)

TRAINING_SIZES = (
    4,
    8,
    16,
    32,
)

DPI = 300


def load_results():
    """Load and validate the aggregated held-out test results."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing held-out aggregate: {INPUT_PATH}"
        )

    frame = pd.read_csv(
        INPUT_PATH
    )

    required_columns = {
        "model_id",
        "n_states",
        "train_trajectories",
        "validation_macro_nrmse",
        "test_macro_nrmse",
        "test_status",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Held-out aggregate is missing columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    return frame


def short_model_id(
    model_id,
):
    """Return the compact numeric portion of a BioModels identifier."""
    return str(model_id).replace(
        "BIOMD000000",
        "",
    )


def save_figure(
    *,
    figure,
    stem,
):
    """Save one figure as PDF and high-resolution PNG."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_path = (
        OUTPUT_DIR
        / f"{stem}.pdf"
    )

    png_path = (
        OUTPUT_DIR
        / f"{stem}.png"
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    print(
        "Saved:",
        pdf_path,
    )

    print(
        "Saved:",
        png_path,
    )


def plot_validation_vs_test(
    frame,
):
    """Plot held-out test NRMSE against validation NRMSE."""
    successful = frame[
        (
            frame["test_status"]
            == "success"
        )
        & np.isfinite(
            frame[
                "validation_macro_nrmse"
            ]
        )
        & np.isfinite(
            frame[
                "test_macro_nrmse"
            ]
        )
        & (
            frame[
                "validation_macro_nrmse"
            ]
            > 0.0
        )
        & (
            frame[
                "test_macro_nrmse"
            ]
            > 0.0
        )
    ].copy()

    if successful.empty:
        raise ValueError(
            "No finite successful held-out results are available."
        )

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            6.5,
        )
    )

    for train_size in TRAINING_SIZES:
        subset = successful[
            successful[
                "train_trajectories"
            ]
            == train_size
        ]

        axis.scatter(
            subset[
                "validation_macro_nrmse"
            ],
            subset[
                "test_macro_nrmse"
            ],
            label=(
                f"{train_size} trajectories"
            ),
            alpha=0.8,
        )

    minimum_value = min(
        successful[
            "validation_macro_nrmse"
        ].min(),
        successful[
            "test_macro_nrmse"
        ].min(),
    )

    maximum_value = max(
        successful[
            "validation_macro_nrmse"
        ].max(),
        successful[
            "test_macro_nrmse"
        ].max(),
    )

    diagonal = np.logspace(
        np.log10(
            minimum_value
        ),
        np.log10(
            maximum_value
        ),
        200,
    )

    axis.plot(
        diagonal,
        diagonal,
        linestyle="--",
        linewidth=1.2,
        label="Validation = test",
    )

    annotations = [
        (
            "BIOMD0000000476",
            32,
        ),
        (
            "BIOMD0000000174",
            16,
        ),
    ]

    for model_id, train_size in annotations:
        point = successful[
            (
                successful[
                    "model_id"
                ]
                == model_id
            )
            & (
                successful[
                    "train_trajectories"
                ]
                == train_size
            )
        ]

        if len(point) != 1:
            continue

        row = point.iloc[0]

        axis.annotate(
            (
                f"{short_model_id(model_id)}, "
                f"train={train_size}"
            ),
            (
                row[
                    "validation_macro_nrmse"
                ],
                row[
                    "test_macro_nrmse"
                ],
            ),
            xytext=(
                8,
                8,
            ),
            textcoords="offset points",
            fontsize=9,
        )

    axis.set_xscale(
        "log"
    )

    axis.set_yscale(
        "log"
    )

    axis.set_xlabel(
        "Validation macro NRMSE"
    )

    axis.set_ylabel(
        "Held-out test macro NRMSE"
    )

    axis.set_title(
        "Selector v1: validation vs. held-out test error"
    )

    axis.legend()

    axis.grid(
        True,
        which="both",
        alpha=0.2,
    )

    save_figure(
        figure=figure,
        stem="validation_vs_test",
    )

    plt.close(
        figure
    )


def plot_test_success_rate(
    frame,
):
    """Plot held-out integration success rate by training size."""
    rows = []

    for train_size in TRAINING_SIZES:
        subset = frame[
            frame[
                "train_trajectories"
            ]
            == train_size
        ]

        if subset.empty:
            continue

        successful = (
            subset[
                "test_status"
            ]
            == "success"
        ).sum()

        total = len(
            subset
        )

        rows.append(
            {
                "train_trajectories": train_size,
                "success_rate": (
                    successful
                    / total
                ),
                "successful": successful,
                "total": total,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            5.5,
        )
    )

    axis.plot(
        summary[
            "train_trajectories"
        ],
        summary[
            "success_rate"
        ],
        marker="o",
    )

    for _, row in summary.iterrows():
        axis.annotate(
            (
                f"{int(row['successful'])}"
                f"/{int(row['total'])}"
            ),
            (
                row[
                    "train_trajectories"
                ],
                row[
                    "success_rate"
                ],
            ),
            xytext=(
                0,
                8,
            ),
            textcoords="offset points",
            ha="center",
        )

    axis.set_xticks(
        list(
            TRAINING_SIZES
        )
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.set_xlabel(
        "Number of training trajectories"
    )

    axis.set_ylabel(
        "Held-out success rate"
    )

    axis.set_title(
        "Selector v1: held-out dynamic robustness"
    )

    axis.grid(
        True,
        alpha=0.2,
    )

    save_figure(
        figure=figure,
        stem="test_success_rate",
    )

    plt.close(
        figure
    )


def plot_generalization_ratio(
    frame,
):
    """Plot test-to-validation NRMSE ratio for successful systems."""
    successful = frame[
        (
            frame[
                "test_status"
            ]
            == "success"
        )
        & np.isfinite(
            frame[
                "validation_macro_nrmse"
            ]
        )
        & np.isfinite(
            frame[
                "test_macro_nrmse"
            ]
        )
        & (
            frame[
                "validation_macro_nrmse"
            ]
            > 0.0
        )
        & (
            frame[
                "test_macro_nrmse"
            ]
            > 0.0
        )
    ].copy()

    successful[
        "generalization_ratio"
    ] = (
        successful[
            "test_macro_nrmse"
        ]
        / successful[
            "validation_macro_nrmse"
        ]
    )

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            6.0,
        )
    )

    for train_size in TRAINING_SIZES:
        subset = successful[
            successful[
                "train_trajectories"
            ]
            == train_size
        ]

        axis.scatter(
            np.full(
                len(subset),
                train_size,
                dtype=float,
            ),
            subset[
                "generalization_ratio"
            ],
            label=(
                f"{train_size} trajectories"
            ),
            alpha=0.8,
        )

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
    )

    axis.set_yscale(
        "log"
    )

    axis.set_xticks(
        list(
            TRAINING_SIZES
        )
    )

    axis.set_xlabel(
        "Number of training trajectories"
    )

    axis.set_ylabel(
        "Test NRMSE / validation NRMSE"
    )

    axis.set_title(
        "Selector v1: held-out generalization ratio"
    )

    axis.grid(
        True,
        which="both",
        alpha=0.2,
    )

    save_figure(
        figure=figure,
        stem="test_validation_ratio",
    )

    plt.close(
        figure
    )


def print_summary(
    frame,
):
    """Print held-out success and generalization statistics."""
    print()
    print("=" * 70)
    print("HELD-OUT GENERALIZATION SUMMARY")
    print("=" * 70)

    print(
        "Models:",
        frame[
            "model_id"
        ].nunique(),
    )

    print(
        "Conditions:",
        len(frame),
    )

    successful = frame[
        frame[
            "test_status"
        ]
        == "success"
    ]

    failed = frame[
        frame[
            "test_status"
        ]
        == "failed"
    ]

    print(
        "Successful conditions:",
        len(successful),
    )

    print(
        "Failed conditions:",
        len(failed),
    )

    print()

    rows = []

    for train_size in TRAINING_SIZES:
        subset = frame[
            frame[
                "train_trajectories"
            ]
            == train_size
        ]

        successful_subset = subset[
            subset[
                "test_status"
            ]
            == "success"
        ].copy()

        finite_subset = successful_subset[
            np.isfinite(
                successful_subset[
                    "validation_macro_nrmse"
                ]
            )
            & np.isfinite(
                successful_subset[
                    "test_macro_nrmse"
                ]
            )
            & (
                successful_subset[
                    "validation_macro_nrmse"
                ]
                > 0.0
            )
        ].copy()

        ratios = (
            finite_subset[
                "test_macro_nrmse"
            ]
            / finite_subset[
                "validation_macro_nrmse"
            ]
        )

        rows.append(
            {
                "train_trajectories": train_size,
                "conditions": len(
                    subset
                ),
                "successful": len(
                    successful_subset
                ),
                "failed": (
                    len(subset)
                    - len(
                        successful_subset
                    )
                ),
                "success_rate": (
                    len(
                        successful_subset
                    )
                    / len(subset)
                ),
                "median_validation_nrmse": (
                    finite_subset[
                        "validation_macro_nrmse"
                    ].median()
                ),
                "median_test_nrmse": (
                    finite_subset[
                        "test_macro_nrmse"
                    ].median()
                ),
                "median_test_validation_ratio": (
                    ratios.median()
                ),
                "max_test_validation_ratio": (
                    ratios.max()
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    print(
        summary.to_string(
            index=False
        )
    )


def main() -> None:
    """Generate held-out generalization figures."""
    frame = load_results()

    print_summary(
        frame
    )

    plot_validation_vs_test(
        frame
    )

    plot_test_success_rate(
        frame
    )

    plot_generalization_ratio(
        frame
    )

    print()
    print(
        "Figure directory:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()