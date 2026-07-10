r"""merge_qwen_encoder.py - merge the Qwen3-VL text encoder shards into ONE
model.safetensors, streaming tensor-by-tensor (never holds the 8GB model in RAM).

transformers loads from a FOLDER, so the merged file stays inside the encoder
folder next to config.json; the shards and the index are retired to a backup
subfolder. The single file is also HF-upload-ready.

Usage:
  cd /d D:\AI_Tools\ComfyUI_windows_portable
  python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI_Rebels_LingBot\merge_qwen_encoder.py --src D:\LINGBOT\text_encoder
"""
import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="text_encoder folder with shards + index")
    a = ap.parse_args()

    idx_path = os.path.join(a.src, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        single = os.path.join(a.src, "model.safetensors")
        if os.path.isfile(single):
            print("already a single model.safetensors - nothing to do")
            return
        raise SystemExit("no index json in {} - is this the text_encoder folder?".format(a.src))

    with open(idx_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    print("{} tensors across {} shards".format(len(weight_map), len(shards)))

    # stream: open each shard once, pull its tensors to CPU, accumulate references.
    # safetensors mmaps, so 'accumulating' costs address space, not RAM; save_file
    # then streams them out. Peak RAM stays around one tensor.
    tensors = {}
    for shard in shards:
        p = os.path.join(a.src, shard)
        print("reading", shard)
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

    out_tmp = os.path.join(a.src, "model.safetensors._tmp")
    out = os.path.join(a.src, "model.safetensors")
    print("writing merged file ({} tensors)...".format(len(tensors)))
    save_file(tensors, out_tmp, metadata={"format": "pt"})
    del tensors
    os.replace(out_tmp, out)
    sz = os.path.getsize(out) / 1e9
    print("wrote {} ({:.2f} GB)".format(out, sz))

    backup = os.path.join(a.src, "_shards_backup")
    os.makedirs(backup, exist_ok=True)
    for shard in shards:
        shutil.move(os.path.join(a.src, shard), os.path.join(backup, shard))
    shutil.move(idx_path, os.path.join(backup, "model.safetensors.index.json"))
    print("shards + index moved to {} (delete after a successful test load)".format(backup))


if __name__ == "__main__":
    main()
