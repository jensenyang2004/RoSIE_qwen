"""
Baseline: the official Qwen-Image-2.1-Turbo edit (vendored pipeline, no instance control) on MICE-Bench / LoMOE-Bench,
one instruction per image (edit_inst_single). Same conventions as mice: {id}_result.png resized back to the source
size, args.json + env_<host>_shard<k>.json in the run folder, finished samples skipped.

    python infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --exp_name turbo_mice
    python infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --sample_ids 06 09 --exp_name smoke
    ./run_2gpu.sh infer_baseline.py --dataset_root ... --exp_name turbo_mice     (two shards, one per GPU)
"""
import argparse
import json
import platform
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from bench import Bench
from qwen21 import MODEL_ID, load_pipeline

RUN_CONTROL = {"sample_ids", "num_shards", "shard_id", "device", "cpu_offload", "overwrite"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="../3D-bench/image/mice_bench")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--model", type=str, default=MODEL_ID)
    p.add_argument("--prompt_key", type=str, default="edit_inst_single")
    p.add_argument("--resolution", type=int, default=1024, help="output area = resolution^2 (pipeline default 1024)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_kv_cache", action="store_true", help="recompute the text/source prefix every step")
    p.add_argument("--sample_ids", type=str, nargs="+", default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cpu_offload", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def environment(args) -> dict:
    import diffusers
    import transformers
    env = dict(host=platform.node(), python=platform.python_version(), torch=torch.__version__,
               cuda=torch.version.cuda, diffusers=diffusers.__version__, transformers=transformers.__version__,
               gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    try:  # the snapshot folder name is the model revision actually loaded
        from huggingface_hub import snapshot_download
        env["model_snapshot"] = snapshot_download(args.model, local_files_only=True)
    except Exception as e:
        env["model_snapshot"] = f"{args.model} (no hub snapshot: {type(e).__name__})"
    return env


def main():
    args = parse_args()
    save_dir = Path(args.output_dir) / f"qwen21_{args.exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    # skipped samples must have been made with the same settings: refuse to mix settings in one folder
    settings = {k: v for k, v in vars(args).items() if k not in RUN_CONTROL}
    args_path = save_dir / "args.json"
    if args_path.exists() and not args.overwrite:
        old = {k: v for k, v in json.loads(args_path.read_text()).items() if k not in RUN_CONTROL}
        if old != settings:
            diff = {k: (old.get(k), settings.get(k)) for k in old.keys() | settings.keys()
                    if old.get(k) != settings.get(k)}
            raise SystemExit(f"{save_dir} was made with other settings {diff} (old, new): use another --exp_name, "
                             f"or --overwrite")
    args_path.write_text(json.dumps(settings, indent=2))

    bench = Bench(args.dataset_root)
    ids = bench.shard(args.num_shards, args.shard_id)
    if args.sample_ids:
        ids = [i for i in ids if i in set(args.sample_ids)]
    if not args.overwrite:
        ids = [i for i in ids if not (save_dir / f"{i}_result.png").exists()]
    print(f"{len(ids)} samples to run -> {save_dir}")
    if not ids:
        return

    pipe = load_pipeline(args.model, args.device, cpu_offload=args.cpu_offload)
    env = environment(args)
    (save_dir / f"env_{env['host']}_shard{args.shard_id}.json").write_text(json.dumps(env, indent=2))

    for sid in tqdm(ids):
        s = bench[sid]
        result = pipe(
            prompt=s[args.prompt_key],
            image=s["image"],
            output_resolution=args.resolution,
            use_kv_cache=not args.no_kv_cache,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0].convert("RGB")  # the VAE is RGBA; the benches and metrics are RGB
        if result.size != s["image"].size:
            result = result.resize(s["image"].size, Image.LANCZOS)
        result.save(save_dir / f"{sid}_result.png")


if __name__ == "__main__":
    main()
