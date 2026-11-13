import numpy as np
from src.prediction_service import fields, scaler

# Kept in the same order as the loaded model.
expected_features = list(fields)


def preprocess_single_row(input_list):
    """
    Scale one row of raw feature values for the loaded model.
    input_list must follow expected_features order.
    """
    values = [float(value) for value in input_list]
    if len(values) != len(expected_features):
        raise ValueError(
            f"Expected {len(expected_features)} features, got {len(values)}."
        )
    scaled_input = scaler.transform(np.array(values, dtype=float).reshape(1, -1))
    print(f"✅ Scaled input: {scaled_input}")
    return scaled_input
