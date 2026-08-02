# ==========================================
# HF CLEANUP - clear the checkpoint repo before a fresh start
# ==========================================
# Removes old checkpoints/ONNX files/logs from the model repo on Hugging Face.
# Touches ONLY the model repo (HF_REPO_ID); the dataset repo is left alone.
#
# TWO STEPS, deliberately:
#   1. run as-is       -> only PRINTS what it would delete (deletes nothing)
#   2. set CONFIRM=True -> actually deletes
#
# KAGGLE: paste into a cell and call main(). Token from the HF_TOKEN secret.
# ==========================================

import sys

CONFIRM = False          # <- set True only after checking the list

HF_REPO_ID     = "greblus/chord-model-snapshots"
HF_SECRET_NAME = "HF_TOKEN"

# Keep files matching these substrings, e.g. the current run if cleaning mid-flight.
KEEP_SUBSTRINGS = []     # np. ["v2_take1"]


def main():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        import subprocess
        subprocess.call([sys.executable, "-m", "pip", "install", "huggingface_hub", "--quiet"])
        from huggingface_hub import HfApi

    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret(HF_SECRET_NAME)
    except Exception:
        import os
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("❌ No HF token (Kaggle secret HF_TOKEN or env var HF_TOKEN).")

    api = HfApi(token=token)
    files = sorted(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="model"))

    protected = {".gitattributes", "README.md"}
    to_delete, to_keep = [], []
    for f in files:
        if f in protected or any(k in f for k in KEEP_SUBSTRINGS):
            to_keep.append(f)
        else:
            to_delete.append(f)

    print(f"📦 Repo: {HF_REPO_ID}   ({len(files)} files)\n")
    if to_keep:
        print("✅ KEEPING:")
        for f in to_keep: print(f"    {f}")
        print()
    print(f"🗑  TO DELETE ({len(to_delete)}):")
    for f in to_delete: print(f"    {f}")

    if not to_delete:
        print("\nNothing to delete."); return

    if not CONFIRM:
        print("\n" + "=" * 62)
        print("PREVIEW MODE - nothing was deleted.")
        print("Check the list above, then set CONFIRM = True and run again.")
        print("=" * 62)
        return

    print("\n🔥 Deleting...")
    ok = err = 0
    for f in to_delete:
        try:
            api.delete_file(path_in_repo=f, repo_id=HF_REPO_ID, repo_type="model")
            ok += 1; print(f"    ✓ {f}")
        except Exception as e:
            err += 1; print(f"    ✗ {f}  ({e})")
    print(f"\nDeleted {ok}, errors {err}.")
    print("Repo ready for a clean run - the trainer writes names from RUN_TAG.")


if __name__ == "__main__":
    main()
