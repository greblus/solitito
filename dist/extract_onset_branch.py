"""Cut the onset branch out of a combined take7 model.

    python dist/extract_onset_branch.py best_model_v2_take7_masking_v2.onnx short_onset_masking_v2.onnx

A take7 file holds the chord trunk (input `features`) and the onset branch
(input `short_features`) side by side, and the onset answer does not depend on
the trunk at all - checked: identical to the last digit with two different
trunk inputs. Run whole, every onset answer pays for the trunk, 38 ms; cut out,
the branch is 1 MB and 0.7 ms a hop, which is what lets the app ask it on every
16 ms hop. `src/strike.rs` loads the result.

The threshold and history the branch was evaluated with travel with it as
metadata, and the file says where it came from. Needs `pip install onnx`.
"""
import sys

import onnx
from onnx import helper, utils


def main(src: str, out: str) -> None:
    meta = {p.key: p.value for p in onnx.load(src).metadata_props}
    utils.extract_model(src, out, input_names=['short_features'], output_names=['onset_logits'])
    model = onnx.load(out)
    keep = {k: meta[k] for k in ('onset_threshold', 'onset_history_frames') if k in meta}
    keep['extracted_from'] = src
    helper.set_model_props(model, keep)
    onnx.checker.check_model(model)
    onnx.save(model, out)
    print(f'{out}: inputs {[i.name for i in model.graph.input]}, '
          f'outputs {[o.name for o in model.graph.output]}, metadata {keep}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
