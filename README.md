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
```

Results go to `results/qwen21_<exp_name>/{id}_result.png`, resized to the source size as in mice.

## Layout

| Path | What |
|---|---|
| `qwen21/transformer_qwenimage21.py`, `qwen21/pipeline_qwenimage21.py` | Copied from diffusers v0.41.0 with only the imports changed. **Never edit them**: experiments wrap them at runtime (attention processors, block wrappers) or live in new files. |
| `qwen21/__init__.py` | `load_pipeline()`: the vendored pipeline with the vendored transformer (plain `from_pretrained` would load diffusers' own class). |
| `bench.py` | MICE-Bench / LoMOE-Bench loader, and `output_size()`, the pipeline's generation size. |
| `infer_baseline.py` | The official Turbo edit with one instruction (`edit_inst_single`) and no instance control. |
| `check_vendored.py` | Equality checks for the vendored code. |

## What differs from FLUX.2-klein (read from the diffusers code)

- **Blocks**: 32 single-stream blocks, width 4096 (32 heads × 128), no double blocks. Text, source and target share
  one residual stream and the same weights.
- **Sequence**: `[chat text] [source image] [instruction text] [target image]`. The source image enters as VAE latents
  placed in the Qwen3-VL image slots (one slot = 2×2 latent tokens); the VLM's own vision features are dropped. One
  token covers 16 px, flattened row-major. The source is resized to the target's grid, and each image's RoPE grid is
  centred on zero, so source token i and target token i share a position.
- **Attention is block-causal**: `q >= k or same image`. Text is causal, the source sees only the text before it, and
  the target sees everything. Text and source never attend to the target.
- **Prefix cache**: text and source are modulated at t = 0, so they do not depend on the step. Step 0 computes the
  whole sequence and stores the prefix K/V. Steps 1–7 run only the target tokens against the cache. Anything read
  from the text or source (the steering anchors) must be captured at step 0; the cache keeps only K/V.
- **Turbo**: 8 steps with the schedule saved in `model_index.json`: σ = 1.0, .978, .954, .927, .895, .845, .705,
  .415. Six of the eight steps are above σ = 0.84. CFG is 1, and the default output is 1024².

## Plan

1. Baseline: run `check_vendored.py --full`, then the Turbo baseline on MICE-Bench and LoMOE-Bench.
2. Token assignment: instance masks → latent-grid indices in the source and target blocks; instance prompts encoded
   separately and appended before the target (mice's `outer_local_prompts`); target-word spans from the tokenizer
   offsets.
3. Binding: one attention processor whose mask is the model's block-causal rule AND the instance binding, used for
   the step-0 prefill and (target rows) for the cached steps. An all-allowed mask must reproduce the baseline
   exactly. Check binding alone before adding steering.
4. Steering: wrap `block.forward` at runtime and rotate the target slice of each block's update (`residual_rotate.py`
   from mice). Anchors are captured at step 0: the source write s_t per position, and the target-word write u_k per
   instance. The block schedule has to be re-diagnosed on the 32 blocks.

License: Qwen Research License (model weights).
