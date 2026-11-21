"""
Baseline: the official Qwen-Image-2.1-Turbo edit (vendored pipeline, no instance control) on MICE-Bench / LoMOE-Bench,
one instruction per image (edit_inst_single). Run folder conventions in runs.py.

    python infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --exp_name turbo_mice
    python infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --sample_ids 06 09 --exp_name smoke
    ./run_2gpu.sh infer_baseline.py --dataset_root ... --exp_name turbo_mice     (two shards, one per GPU)
"""
import argparse

import torch
from tqdm import tqdm

from bench import Bench
from qwen21 import MODEL_ID, load_pipeline
from runs import add_run_args, run_folder, save_environment, save_result, todo


def parse_args():
    p = argparse.ArgumentParser()
    add_run_args(p)
    p.add_argument("--model", type=str, default=MODEL_ID)
    p.add_argument("--prompt_key", type=str, default="edit_inst_single")
    p.add_argument("--resolution", type=int, default=1024, help="output area = resolution^2 (pipeline default 1024)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_kv_cache", action="store_true", help="recompute the text/source prefix every step")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = run_folder(args, "qwen21")
    bench = Bench(args.dataset_root)
    ids = todo(bench, args, save_dir)
    if not ids:
        return
    pipe = load_pipeline(args.model, args.device, cpu_offload=args.cpu_offload)
    save_environment(save_dir, args, args.model)

    for sid in tqdm(ids):
        s = bench[sid]
        result = pipe(
            prompt=s[args.prompt_key],
            image=s["image"],
            output_resolution=args.resolution,
            use_kv_cache=not args.no_kv_cache,
            generator=torch.Generator("cpu").manual_seed(args.seed),
        ).images[0]
        save_result(result, s["image"], save_dir / f"{sid}_result.png")


if __name__ == "__main__":
    main()
