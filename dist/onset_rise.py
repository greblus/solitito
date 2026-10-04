"""A causal spectral-change input, not a gate and not an extra training loss."""
import math

RISE_PAST_FRAMES = 4
RISE_SPEC = {
    "version": "positive-spectral-rise-v1",
    "past_frames": RISE_PAST_FRAMES,
    "input": "existing 770 float32 log-magnitude features, after ordinary gain augmentation",
    "formula": "a=expm1(x*log(1001)); d=relu(a-mean(previous 4 a)); log1p(d)/log(1001)",
    "current_frame_excluded_from_baseline": True,
    "startup": "zero feature history",
    "future_frames": 0,
    "projection": "extra 770->96 linear projection, no bias, initialized to zero; added before first ReLU",
    "shared_weights": "identical initialization of all original parameters",
    "loss": "ordinary BCE, positive weight 4, no paired loss or extra negative weights",
    "raw_feature_cache": "unchanged; rise is computed inside the exported ONNX graph",
}


def positive_spectral_rise(features):
    import torch
    # Exp/Sub and Add/Log also work in the app's ONNX opset 17 exporter.
    amplitude = torch.exp(features * math.log(1001)) - 1
    # Exclude the current frame: a left pad of four produces T+1 averages,
    # and the final average belongs to the following (unavailable) frame.
    previous = torch.nn.functional.avg_pool1d(
        torch.nn.functional.pad(amplitude, (RISE_PAST_FRAMES, 0)),
        kernel_size=RISE_PAST_FRAMES, stride=1)[:, :, :-1]
    return torch.log(1 + torch.relu(amplitude - previous)) / math.log(1001)
