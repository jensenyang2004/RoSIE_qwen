"""
One instance-controlled edit: tokens (rosie/tokens.py) -> layout -> binding mask (rosie/binding.py) -> the vendored
pipeline, wrapped at runtime (its encode_prompt and the attention processors), never edited.
"""
from contextlib import contextmanager

import torch
from PIL import Image

from bench import output_size
from rosie.binding import allowed_mask, binding
from rosie.tokens import GLOBAL_PROMPT, LOCAL_TEMPLATE, Layout, Prompter, build_layout, grid_masks, tokenize


def prepare(prompter: Prompter, sample: dict, resolution: int = 1024, local_mode: str = "text",
            strict: bool = True):
    """Everything that needs no model: output size, token segments, grid masks, layout."""
    assert local_mode in ("text", "image"), local_mode
    w, h = output_size(*sample["image"].size, resolution)
    H, W = h // 16, w // 16
    img = sample["image"].resize((w, h), Image.LANCZOS)  # only counts the VLM image slots
    glob = tokenize(prompter, GLOBAL_PROMPT, img)
    contents = [LOCAL_TEMPLATE.format(src=s, tgt=t) for s, t in zip(sample["sources"], sample["targets"])]
    image_mode = local_mode == "image"
    locs = [tokenize(prompter, c, img if image_mode else None, s, t, after_image=image_mode)
            for c, s, t in zip(contents, sample["sources"], sample["targets"])]
    layout = build_layout(glob, locs, grid_masks(sample["masks"], H, W, strict), H, W)
    return dict(w=w, h=h, glob=glob, contents=contents, locs=locs, layout=layout, local_mode=local_mode)


@contextmanager
def instance_prompts(pipe, prep: dict):
    """Replace the pipeline's encode_prompt: the global prompt with the source image, then every instance prompt
    encoded on its own (pipe._get_qwen_prompt_embeds, so exactly the pipeline's encoding), as one text sequence."""
    L: Layout = prep["layout"]

    def encode_prompt(prompt=None, image=None, device=None, num_images_per_prompt=1, **_):
        assert num_images_per_prompt == 1 and image is not None
        emb, _, ipm = pipe._get_qwen_prompt_embeds([GLOBAL_PROMPT], image, device)
        assert emb.shape[1] == len(prep["glob"].ids) and torch.equal(ipm[0].cpu(), prep["glob"].slot), \
            "global prompt encoded differently from its tokenization"
        parts = [emb]
        for content, seg in zip(prep["contents"], prep["locs"]):
            e, _, _ = pipe._get_qwen_prompt_embeds([content], image if prep["local_mode"] == "image" else None,
                                                   device)
            assert e.shape[1] == len(seg.ids), f"instance prompt {content!r} encoded differently"
            parts.append(e[:, seg.keep])
        emb = torch.cat(parts, dim=1)
        assert emb.shape[1] == len(L.img_mask)
        return emb, None, L.img_mask[None].to(emb.device)

    pipe.encode_prompt = encode_prompt  # instance attribute shadows the method
    try:
        yield
    finally:
        del pipe.encode_prompt


def edit(pipe, sample: dict, resolution: int = 1024, local_mode: str = "text", strict: bool = True,
         bind: bool = True, seed: int = 0, use_kv_cache: bool = True):
    """Returns (PIL image at the generation size, prep)."""
    prep = prepare(Prompter.from_pipe(pipe), sample, resolution, local_mode, strict)
    L = prep["layout"]
    allowed = allowed_mask(L, bind)
    with instance_prompts(pipe, prep), binding(pipe.transformer, allowed, L.H * L.W):
        out = pipe(prompt=GLOBAL_PROMPT, image=sample["image"], height=prep["h"], width=prep["w"],
                   output_resolution=resolution, use_kv_cache=use_kv_cache,
                   generator=torch.Generator("cpu").manual_seed(seed)).images[0]
    return out, prep
