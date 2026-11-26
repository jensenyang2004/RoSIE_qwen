# rosie-qwen

RoSIE (rotation-based residual steering, developed on FLUX.2-klein-4B in `../mice`) ported to
[Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) to test whether it carries over to a second
backbone.

## Setup

Use a separate environment: this repo needs transformers 5.x for Qwen3-VL, which must not go into the mice venv.
The new venv can borrow everything else (torch, accelerate, ...) from the mice venv through a `.pth` file. Its own
packages come first on `sys.path`, so the diffusers/transformers installed here shadow mice's, and the mice venv
is never changed:

```bash
MICE_VENV=../mice/.venv                              # the mice uv venv (adjust the path)
uv venv --python $MICE_VENV/bin/python .venv          # same Python as mice, so its compiled packages load
SP=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
MICE_SP=$($MICE_VENV/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$MICE_SP" > "$SP/_mice_base.pth"
source .venv/bin/activate
uv pip install "diffusers==0.41.0" "transformers>=5.17"
python -c "import torch, diffusers, transformers; print(torch.__version__, diffusers.__version__, transformers.__version__)"
```

A standalone environment works too: `uv venv .venv && uv pip install -r requirements.txt`.

```bash
python check_vendored.py                 # CPU, no weights: vendored == installed diffusers
python check_vendored.py --full          # GPU: whole pipeline, official vs vendored, max pixel diff must be 0
```

Weights download on first use (~31 GB: 7B transformer + Qwen3-VL-8B text encoder + VAE). One 80 GB GPU holds
everything in bf16. On smaller cards, pass `--cpu_offload`.

## Run

```bash
python infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --exp_name turbo_mice --sample_ids 06 09
./run_2gpu.sh infer_baseline.py --dataset_root ../3D-bench/image/mice_bench --exp_name turbo_mice
./run_2gpu.sh infer_baseline.py --dataset_root ../3D-bench/image/lomoe_bench --exp_name turbo_lomoe
python infer_bind.py --dataset_root ../3D-bench/image/mice_bench --exp_name bind_smoke --sample_ids 06 09 276
python infer_steer.py --dataset_root ../3D-bench/image/mice_bench --exp_name rot_smoke --sample_ids 08 09 276
```

Results go to `results/qwen21_<exp_name>/` (baseline), `results/qwen21bind_<exp_name>/` (binding) and
`results/qwen21rot_<exp_name>/` (binding + steering) as `{id}_result.png`, resized to the source size as in mice.
The steering run also writes `{id}_steer.json`: per step and block, the mean turn of each term (deg) and the
cosines to the anchors.

## Layout

| Path | What |
|---|---|
| `qwen21/transformer_qwenimage21.py`, `qwen21/pipeline_qwenimage21.py` | Copied from diffusers v0.41.0 with only the imports changed. **Never edit them**: experiments wrap them at runtime (attention processors, block wrappers) or live in new files. |
| `qwen21/__init__.py` | `load_pipeline()`: the vendored pipeline with the vendored transformer (plain `from_pretrained` would load diffusers' own class). |
| `bench.py` | MICE-Bench / LoMOE-Bench loader, and `output_size()`, the pipeline's generation size. |
| `infer_baseline.py` | The official Turbo edit with one instruction (`edit_inst_single`) and no instance control. |
| `check_vendored.py` | Equality checks for the vendored code. |
| `rosie/tokens.py` | Token assignment: prompt tokenization (exactly the pipeline's), target-word positions, grid masks, the joint-sequence `Layout`. |
| `rosie/binding.py` | The binding mask (mice's final rules AND the model's block-causal rule) and `BindAttnProcessor`. |
| `rosie/instance_edit.py` | One edit: wraps the pipeline's `encode_prompt` (global + instance prompts) and installs the processor. |
| `infer_bind.py` | Binding only, no steering, on the bench. |
| `check_binding.py` | CPU checks of the binding (tiny random model); `--tokens` checks token positions with the real processor. |
| `rosie/positions.py` | RoPE positions of the joint sequence; `--prompt_pos shared` puts every instance prompt at the first one's positions. |
| `rosie/steer.py` | Rotation steering: block hooks that turn each block's update of the target tokens; anchors captured in the step-0 prefill. |
| `infer_steer.py` | Binding + steering on the bench (mice's final strengths by default). |
| `check_steer.py` | CPU checks: the rotation equals mice's `ResidualRotate` (needs `../mice` or `--mice`), and cached anchors give the same output as recomputing them every step. |
| `runs.py` | Run-folder conventions shared by the scripts. |

## What differs from FLUX.2-klein (read from the diffusers code)

- **Blocks**: 32 single-stream blocks, width 4096 (32 heads × 128), no double blocks. Text, source and target share
  one residual stream and the same weights.
- **Sequence**: `[chat text] [source image] [instruction text] [target image]`. The source image enters as VAE latents
  placed in the Qwen3-VL image slots (one slot = 2×2 latent tokens); the VLM's own vision features are dropped. One
  token covers 16 px, flattened row-major. The source is resized to the target's grid, and each image's RoPE grid is
  centred on zero, so source token i and target token i share a position.
- **Attention is block-causal**: `q >= k or same image`. Text is causal, the source sees only the text before it, and
  the target sees everything. Text and source never attend to the target.
- **Text positions share the image axes**: a text token at position p gets (frame, h, w) = (p, p, p) in RoPE; an
  image block sits at frame = where it starts, h/w = its grid centred on zero. Stacking instance prompts (mice's
  layout) therefore shifts each later prompt relative to the image grid; in FLUX.2 text had its own axis, so it did
  not. With the stacked layout only the first instance prompt gets followed (08, 09, 276); `--prompt_pos shared` gives
  every prompt the first one's positions.
- **Prefix cache**: text and source are modulated at t = 0, so they do not depend on the step. Step 0 computes the
  whole sequence and stores the prefix K/V. Steps 1–7 run only the target tokens against the cache. Anything read
  from the text or source (the steering anchors) must be captured at step 0; the cache keeps only K/V.
- **Turbo**: 8 steps with the schedule saved in `model_index.json`: σ = 1.0, .978, .954, .927, .895, .845, .705,
  .415. Six of the eight steps are above σ = 0.84. CFG is 1, and the default output is 1024². The pipeline
  ignores `num_inference_steps` when the model has `sample_sigmas`; `--steps N` (infer_bind.py, infer_steer.py)
  resamples this curve to N steps (`rosie/schedule.py`).

## Plan

1. Baseline (done: `check_vendored.py --full` gives identical images): run the Turbo baseline on both benches.
2. Token assignment (written, checked on CPU): instance masks → latent-grid indices in the source and target blocks; instance prompts encoded
   separately and appended before the target (mice's `outer_local_prompts`); target-word spans from the tokenizer
   offsets.
3. Binding (done, run on GPU; the lower-memory grouped processor is checked on CPU only): one attention processor whose mask is the model's block-causal rule AND the instance binding, used for
   the step-0 prefill and (target rows) for the cached steps. With the binding off it matches the model's own
   processor (to float rounding). Check binding alone before adding steering.
4. Steering (written, checked on CPU, not yet run on GPU): block hooks rotate the target slice of each block's
   update (mice's `residual_rotate.py`, same math). The source write s_i and the target-word write u_k are captured
   in the step-0 prefill. Defaults are mice's final strengths in all 32 blocks; contact runs on steps 0–2, the
   same noise range as mice's steps 0–1. Strengths and the block schedule may need re-tuning on this model.

License: Qwen Research License (model weights).
