from psycop_model_evaluation.binary.performance_by_ppr.lib.percent_of_all_events_hit_by_true_positive import (
    get_percentage_of_events_captured,
    get_percentage_of_events_captured_from_eval_dataset,
)
from psycop_model_evaluation.test_meta.test_utils import str_to_df
from psycop_model_training.training_output.dataclasses import EvalDataset


def test_get_percentage_of_events_captured_from_eval_dataset(
    synth_eval_dataset: EvalDataset,
):
    result = get_percentage_of_events_captured_from_eval_dataset(
        eval_dataset=synth_eval_dataset, postive_rate=0.02
    )


def test_get_percentage_of_events_captured():
    input_df = str_to_df(
        """id,y,pred
        1,1,0, # Not captured
        2,1,0, # Not captured
        3,1,1, # Captured
        3,1,0, # Not relevant: ID is 3
        4,0,0, # Not relevant: y is 0
        """
    )

    percentage_of_events_captured = get_percentage_of_events_captured(df=input_df)

    assert percentage_of_events_captured == 1 / 3
