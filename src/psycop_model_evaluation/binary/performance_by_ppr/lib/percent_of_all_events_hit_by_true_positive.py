import pandas as pd
from psycop_model_training.training_output.dataclasses import EvalDataset


def get_percentage_of_events_captured_from_eval_dataset(
    eval_dataset: EvalDataset,
    positive_rate: float,
) -> float:
    df = pd.DataFrame(
        {
            "y": eval_dataset.y,
            "pred": eval_dataset.get_predictions_for_positive_rate(positive_rate)[0],
            "id": eval_dataset.ids,
        },
    )

    return get_percentage_of_events_captured(df=df)


def get_percentage_of_events_captured(df: pd.DataFrame) -> float:
    # Get all patients with at least one event and at least one positive prediction
    df_patients_with_events = df.groupby("id").filter(lambda x: x["y"].sum() > 0)

    df_events_captured = df_patients_with_events.groupby("id").filter(
        lambda x: x["pred"].sum() > 0,
    )

    return len(df_events_captured) / len(df_patients_with_events)