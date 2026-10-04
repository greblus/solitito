"""Build one readable Python file for Kaggle, without repo imports or embedded exec."""
import ast
from pathlib import Path
import textwrap

from build_train_onset_kaggle import build, node_text


def build_model_trainer(directory):
    header = '''"""Solitito take7: one ONNX with chord, quality, pitch and Rise onset outputs.

Copy/run this WHOLE file in the existing Kaggle notebook. Configuration is at
the top: RUN_TAG, MODE, BASE_RUN, HF_REPO_ID, USE_HF, INITIAL_ONSET, ONSET_EPOCHS.
AUTO reuses take6 and trains only Rise; FULL ignores take6 and trains both models.
Both modes resume their own run. For a completely new run choose a new RUN_TAG.
USE_HF=False works locally without a Hugging Face account. Errors accessing HF
are errors, never evidence of an empty repository. The final graph has two inputs
(CQT features and short spectra) and four outputs. MODE="export_only" combines
an already completed two-file run without training or feature preparation.

Generated from chord_training.py, take7_training.py and the tested onset modules
by build_model_trainer.py. Edit sources and regenerate, not this copy.
"""

if __name__ == "__main__":
    import importlib.util
    import subprocess
    import sys
    export_only = (MODE == "export_only" or "--mode=export_only" in sys.argv or
                   any(a == "--mode" and b == "export_only" for a, b in zip(sys.argv, sys.argv[1:])))
    packages = {"onnx": "onnx", "onnxruntime": "onnxruntime", "numpy": "numpy",
                "huggingface_hub": "huggingface_hub", "soundfile": "soundfile", "scipy": "scipy"}
    if not export_only:
        packages.update(joblib="joblib", librosa="librosa", pandas="pandas", tqdm="tqdm")
    if not USE_HF or "--no-hf" in sys.argv:
        packages.pop("huggingface_hub")
    missing = [package for module, package in packages.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    if not export_only and importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is required; select a Kaggle GPU image or install it first")

'''
    onset = build(directory)
    nodes = []
    for node in ast.parse(onset).body:
        if isinstance(node, ast.If) or isinstance(node, ast.Expr):
            continue
        nodes.append(node_text(onset, node))
    exports = ["run_pipeline", "onset_sources", "cache_onset_features", "train_onset_experiment",
               "sha256", "write_json", "FEATURE_SPEC"]
    factory = ('def _onset_runtime():\n' + textwrap.indent('\n\n'.join(nodes), '    ')
               + '\n    from types import SimpleNamespace\n    return SimpleNamespace('
               + ', '.join(f'{name}={name}' for name in exports) + ')\n\n')
    chords = (directory / 'chord_training.py').read_text()
    entry = (directory / 'take7_training.py').read_text()
    code = []
    configuration = []
    for node in ast.parse(entry).body:
        if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
            configuration.append(node_text(entry, node))
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == 'chord_training':
            continue
        if isinstance(node, ast.ImportFrom) and node.module == 'train_short_onset':
            code.append('_onset = _onset_runtime()\n' + '\n'.join(f'{n} = _onset.{n}' for n in exports))
            continue
        code.append(node_text(entry, node))
    header = header.replace('if __name__ == "__main__":', '\n'.join(configuration) + '\n\nif __name__ == "__main__":', 1)
    return header + chords + '\n\n' + factory + '\n\n'.join(code) + '\n'


if __name__ == '__main__':
    directory = Path(__file__).parent
    (directory / 'model_trainer.py').write_text(build_model_trainer(directory))
