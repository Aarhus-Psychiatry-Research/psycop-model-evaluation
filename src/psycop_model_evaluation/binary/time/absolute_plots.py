from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from psycop_model_evaluation.base_charts import (
    plot_basic_chart,
)
from psycop_model_evaluation.binary.time.absolute_data import (
    create_roc_auc_by_absolute_time_df,
)
from psycop_model_training.training_output.dataclasses import EvalDataset


def plot_metric_by_absolute_time(
    eval_dataset: EvalDataset,
    y_title: str = "AUC",
    bin_period: Literal["H", "D", "W", "M", "Q", "Y"] = "Y",
    pred_type: Optional[str] = "visits",
    save_path: Optional[Union[str, Path]] = None,
    y_limits: Optional[tuple[float, float]] = (0.5, 1.0),
) -> Union[None, Path]:
    """Plot performance by calendar time of prediciton.
    Args:
        eval_dataset (EvalDataset): EvalDataset object
        y_title (str): Title of y-axis. Defaults to "AUC".
        bin_period (str): Which time period to bin on. Takes "M" for month, "Q" for quarter or "Y" for year
        pred_type (str, optional): What description of prediction type to use for plotting, e.g. "number of visits".
        save_path (str, optional): Path to save figure. Defaults to None.
        y_limits (tuple[float, float], optional): Limits of y-axis. Defaults to (0.5, 1.0).
    Returns:
        Union[None, Path]: Path to saved figure or None if not saved.
    """
    df = create_roc_auc_by_absolute_time_df(
        labels=eval_dataset.y,
        y_hat=eval_dataset.y_hat_probs,
        timestamps=eval_dataset.pred_timestamps,
        bin_period=bin_unit,
    )
    sort_order = list(range(len(df)))

    x_titles = {
        "H": "Hour",
        "D": "Day",
        "W": "Week",
        "M": "Month",
        "Q": "Quarter",
        "Y": "Year",
    }

    return plot_basic_chart(
        x_values=df["time_bin"],
        y_values=df["metric"],
        x_title=x_titles[bin_unit],
        y_title=y_title,
        sort_x=sort_order,
        y_limits=y_limits,
        bar_count_values=df["n_in_bin"],
        bar_count_y_axis_title=f"Number of {pred_type}",
        plot_type=["line", "scatter"],
        save_path=save_path,
    )


def plot_prob_over_time(
    timestamp: Iterable[datetime],
    pred_prob: Iterable[float],
    label: Iterable[Union[int, str]],
    outcome_timestamp: Iterable[Union[datetime, None]],
    patient_id: Iterable[Union[int, str]],
    x_axis: str = "Time from outcome",
    y_axis: str = "Model Predictive Probability",
    legend: str = "Highest Predictive Probability",
    look_behind_distance: Optional[int] = None,
    line_opacity: Optional[float] = 0.3,
    fig_size: Optional[tuple] = (10, 10),
    save_path: Optional[Path] = None,
) -> Union[None, Path]:
    """Plot probabilities over time for a given outcome. Each element passed
    (e.g. timestamp, pred_prob etc.) must have the same length, and for each
    iterable, the i'th item must correspond to the same patient.
    Args:
        timestamp (Iterable[datetime]): Timestamps for each prediction time.
        pred_prob (Iterable[float]): The predictive probabilities of the model for each prediction time.
        label (Iterable[Union[int, str]]): True labels for each prediction time.
        outcome_timestamp (Iterable[Union[datetime, None]]): Timestamp of the
            positive outcome.
        patient_id (Iterable[Union[int, str]]): Patient ID for each prediction time. Used for
            connecting timestamp/pred-prob points into one line pr. patient.
        x_axis (str, optional): Label on x-axis. Defaults to "Time from outcome".
        y_axis (str, optional): Label of y-axis. Defaults to "Model Predictive
            Probability".
        legend (str, optional): Label on legend. Defaults to "Highest Predictive
            Probability".
        look_behind_distance (Optional[int], optional): Look-behind window. Used for
            shading the corresponding area. Defaults to None in which case no shaded
            areas is plotted.
        line_opacity (float, optional): Opacity of the line. Defaults to 0.3.
        fig_size (Optional[tuple], optional): figure size. Defaults to None.
        save_path (Optional[Path], optional): path to save figure. Defaults to None.
    Returns:
        Union[None, Path]: None if save_path is None, else path to saved figure
    Examples:
        >>> from pathlib import Path
        >>> repo_path = Path(__file__).parent.parent.parent.parent
        >>> path = repo_path / "tests" / "test_data" / "synth_eval_data.csv"
        >>> df = pd.read_csv(path)
        >>> plot_prob_over_time(
        >>>     patient_id=df["dw_ek_borger"],
        >>>     timestamp=df["timestamp"],
        >>>     pred_prob=df["pred_prob"],
        >>>     outcome_timestamp=df["timestamp_t2d_diag"],
        >>>     label=df["label"],
        >>>     look_behind=500,
        >>> )
    """

    # construct pandas df ensuring types
    plot_df = pd.DataFrame(
        {
            "timestamp": list(timestamp),
            "pred_prob": list(pred_prob),
            "outcome_timestamp": list(outcome_timestamp),
            "patient_id": list(patient_id),
            "label": list(label),
        },
    )
    # remove individuals with no outcome
    plot_df = plot_df.dropna()

    time_cols = ["timestamp", "outcome_timestamp"]
    plot_df.loc[:, time_cols] = plot_df[time_cols].apply(pd.to_datetime)
    plot_df["delta_time"] = plot_df["timestamp"] - plot_df["outcome_timestamp"]
    plot_df["delta_time"] = plot_df["delta_time"].dt.days

    plot_df["patient_id"] = plot_df["patient_id"].astype(str)
    plot_df["label"] = plot_df["label"].astype(str)

    max_pred_prob = plot_df.groupby(["patient_id"])["pred_prob"].max()
    plot_df["color"] = [max_pred_prob[id_] for id_ in plot_df["patient_id"]]

    plt.figure(figsize=fig_size)

    sns.lineplot(
        data=plot_df,
        x="delta_time",
        y="pred_prob",
        alpha=line_opacity,
        hue="color",
        palette="magma",
        legend="auto",
    )

    # Reformat y-axis values to percentage
    plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))  # type: ignore

    plt.xlabel(x_axis, size=14)
    plt.ylabel(y_axis, size=14)

    plt.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.gca().set_facecolor("#f2f2f2")

    plt.legend(title=legend, loc="lower right", fontsize=12)

    # Add shaded area for look-behind window
    if look_behind_distance is not None:
        plt.axvspan(
            -look_behind_distance,
            0,
            color="grey",
            alpha=0.2,
        )
        plt.text(
            -look_behind_distance / 2,
            plot_df["pred_prob"].max(),
            "Predictive window",
            horizontalalignment="center",
            verticalalignment="center",
            rotation=0,
            size=12,
        )

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()
    return save_path
