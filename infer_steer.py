"""
RoSIE on Qwen-Image-2.1-Turbo: the text binding of infer_bind.py plus rotation steering (rosie/steer.py) of every
block's update of the target tokens. Defaults are mice's final run (4B_ROT_mice_2), every block. Run folder
conventions in runs.py; {id}_steer.json has the per-(step, block) turns and cosines, {id}_layout.json the token
counts.

    python infer_steer.py --sample_ids 08 09 276 --exp_name rot_smoke
    python infer_steer.py --sample_ids 08 09 276 --exp_name rot_noins --rot_ins 0     (one term off)
    ./run_2gpu.sh infer_steer.py --exp_name rot_mice
"""
import argparse
import json

from tqdm import tqdm

from bench import Bench
from qwen21 import MODEL_ID, load_pipeline
from rosie.instance_edit import edit
from rosie.steer import TERMS, RotationSteer
from runs import add_run_args, run_folder, save_environment, save_result, todo


def steps(s: str):
    """"all" -> None (every step), "none" -> (), "0,1,2" -> (0, 1, 2)."""
    return None if s == "all" else () if s == "none" else tuple(int(i) for i in s.split(","))


def parse_args():
    p = argparse.ArgumentParser()
    add_run_args(p)
    p.add_argument("--model", type=str, default=MODEL_ID)
    p.add_argument("--resolution", type=int, default=1024, help="output area = resolution^2 (pipeline default 1024)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local_mode", type=str, default="text", choices=["text", "image"])
    p.add_argument("--no_strict", action="store_true", help="keep overlapping masks (mice: --strict_masks on)")
    p.add_argument("--no_bind", action="store_true", help="no text binding (the model's own mask)")
    p.add_argument("--no_kv_cache", action="store_true", help="recompute the text/source prefix every step")
    p.add_argument("--prompt_pos", type=str, default="stacked", choices=["stacked", "shared"],
                   help="RoPE positions of the instance prompts (rosie/positions.py): stacked one after another (the "
                        "model's own), or shared = every prompt at the first one's positions")

    p.add_argument("--rot_ctc", type=float, default=0.9, help="contact: fraction of the way from each neighbour's "
                   "write to orthogonal, times the pair weight")
    p.add_argument("--rot_src", type=float, default=0.25, help="source: fraction of the way from the source write "
                   "to orthogonal (instance tokens)")
    p.add_argument("--rot_bg", type=float, default=0.35, help="background: fraction of the way to the source write")
    p.add_argument("--rot_ins", type=float, default=0.2, help="instruction: fraction of the way to the target-word "
                   "write")
    p.add_argument("--ctc_steps", type=str, default="0,1,2", help="steps of the contact term, \"all\" or \"none\" "
                   "(0,1,2 = the noise range of mice's steps 0,1)")
    p.add_argument("--src_steps", type=str, default="all")
    p.add_argument("--bg_steps", type=str, default="all")
    p.add_argument("--ins_steps", type=str, default="all")
    p.add_argument("--sigma_terms", type=str, default="ctc,src,bg", help="terms scaled by the step's sigma, or none")
    p.add_argument("--local_tau", type=float, default=2.0, help="contact pair-weight falloff (tokens)")
    p.add_argument("--bg_margin", type=float, default=1.0, help="background tokens this close to an instance are free")
    p.add_argument("--bg_tau", type=float, default=2.0, help="background ramp width after the margin (tokens)")
    p.add_argument("--max_turn", type=float, default=90.0, help="cap on the total turn per token (deg)")
    args = p.parse_args()
    terms = () if args.sigma_terms == "none" else tuple(args.sigma_terms.split(","))
    if set(terms) - set(TERMS):
        p.error(f"--sigma_terms: use {','.join(TERMS)} or none")
    return args, terms


def main():
    args, terms = parse_args()
    save_dir = run_folder(args, "qwen21rot")
    bench = Bench(args.dataset_root)
    ids = todo(bench, args, save_dir)
    if not ids:
        return
    pipe = load_pipeline(args.model, args.device, cpu_offload=args.cpu_offload)
    save_environment(save_dir, args, args.model)
    steer = RotationSteer(rot_ctc=args.rot_ctc, rot_src=args.rot_src, rot_bg=args.rot_bg, rot_ins=args.rot_ins,
                          ctc_steps=steps(args.ctc_steps), src_steps=steps(args.src_steps),
                          bg_steps=steps(args.bg_steps), ins_steps=steps(args.ins_steps), sigma_terms=terms,
                          local_tau=args.local_tau, bg_margin=args.bg_margin, bg_tau=args.bg_tau,
                          max_turn_deg=args.max_turn)

    for sid in tqdm(ids):
        s = bench[sid]
        result, prep = edit(pipe, s, resolution=args.resolution, local_mode=args.local_mode,
                            strict=not args.no_strict, bind=not args.no_bind, seed=args.seed,
                            use_kv_cache=not args.no_kv_cache, prompt_pos=args.prompt_pos, steer=steer)
        (save_dir / f"{sid}_layout.json").write_text(json.dumps(prep["layout"].summary()))
        (save_dir / f"{sid}_steer.json").write_text(json.dumps(steer.stats))
        save_result(result, s["image"], save_dir / f"{sid}_result.png")


if __name__ == "__main__":
    main()
