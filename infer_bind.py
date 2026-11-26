"""
Text binding only (no steering): mice's final binding on Qwen-Image-2.1-Turbo, global prompt " " plus one
"Replace the {src} with {tgt}" prompt per instance (rosie/). Run folder conventions in runs.py; {id}_layout.json
records the token counts of each sample.

    python infer_bind.py --sample_ids 06 09 276 --exp_name bind_smoke
    python infer_bind.py --sample_ids 06 09 276 --exp_name bind_smoke_image --local_mode image
    python infer_bind.py --sample_ids 06 --exp_name nobind --no_bind     (same prompts, the model's own attention)
    ./run_2gpu.sh infer_bind.py --exp_name bind_mice
"""
import argparse
import json

from tqdm import tqdm

from bench import Bench
from qwen21 import MODEL_ID, load_pipeline
from rosie.instance_edit import edit
from rosie.schedule import resample_sigmas
from runs import add_run_args, run_folder, save_environment, save_result, todo


def parse_args():
    p = argparse.ArgumentParser()
    add_run_args(p)
    p.add_argument("--model", type=str, default=MODEL_ID)
    p.add_argument("--resolution", type=int, default=1024, help="output area = resolution^2 (pipeline default 1024)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local_mode", type=str, default="text", choices=["text", "image"],
                   help="encode each instance prompt text-only, or with the source image (keep the text after it)")
    p.add_argument("--no_strict", action="store_true", help="keep overlapping masks (mice: --strict_masks on)")
    p.add_argument("--no_bind", action="store_true", help="same prompts, no binding (the model's own mask)")
    p.add_argument("--no_kv_cache", action="store_true", help="recompute the text/source prefix every step")
    p.add_argument("--prompt_pos", type=str, default="stacked", choices=["stacked", "shared"],
                   help="RoPE positions of the instance prompts (rosie/positions.py): stacked one after another (the "
                        "model's own), or shared = every prompt at the first one's positions")
    p.add_argument("--steps", type=int, default=None, help="denoising steps on the curve of Turbo's schedule "
                   "(rosie/schedule.py); default: the model's own 8")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = run_folder(args, "qwen21bind")
    bench = Bench(args.dataset_root)
    ids = todo(bench, args, save_dir)
    if not ids:
        return
    pipe = load_pipeline(args.model, args.device, cpu_offload=args.cpu_offload)
    save_environment(save_dir, args, args.model)
    sigmas = None if args.steps is None else resample_sigmas(pipe.config.sample_sigmas, args.steps)
    if sigmas is not None:
        print(f"{len(sigmas)} steps, sigma " + " ".join(f"{s:.3f}" for s in sigmas))

    for sid in tqdm(ids):
        s = bench[sid]
        result, prep = edit(pipe, s, resolution=args.resolution, local_mode=args.local_mode,
                            strict=not args.no_strict, bind=not args.no_bind, seed=args.seed,
                            use_kv_cache=not args.no_kv_cache, prompt_pos=args.prompt_pos, sigmas=sigmas)
        (save_dir / f"{sid}_layout.json").write_text(json.dumps(prep["layout"].summary()))
        save_result(result, s["image"], save_dir / f"{sid}_result.png")


if __name__ == "__main__":
    main()
