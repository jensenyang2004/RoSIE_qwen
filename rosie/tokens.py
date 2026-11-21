"""
Token assignment for the joint Qwen-Image-2.1 sequence. Pure bookkeeping: needs the processor (tokenizer), never
the model.

The transformer builds its joint sequence from encoder_hidden_states + img_mask (see transformer_qwenimage21.py):
every VLM image slot becomes 2x2 latent tokens, filled row-major with the VAE latents, and the target image is
appended. We pass the global prompt followed by every instance prompt as one text sequence, so the joint sequence is

    [global text | source image, H*W | global text] [instance prompt 1] ... [instance prompt K] [target image, H*W]

    global prompt     the edit template with the source image and an empty instruction (" "), as mice's global " "
    instance prompt   "Replace the {src} with {tgt}" (mice's outer_local_prompts), encoded on its own:
                      mode "text"  text-only template (the Qwen3-VL encoder never sees the image; closest to mice)
                      mode "image" edit template with the source image; only the tokens after the image are kept

Source token i and target token i share a grid position (the condition image is resized to the output size).
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

LOCAL_TEMPLATE = "Replace the {src} with {tgt}"
GLOBAL_PROMPT = " "  # mice's global prompt
SLOT = 4  # latent tokens per VLM image slot (2x2)


@dataclass
class Prompter:
    """What the pipeline uses to encode a prompt (templates, system-token drop, image token)."""
    processor: object
    drop_idx: int
    t2i: str
    ti2i: str
    img_token_id: int

    @classmethod
    def from_pipe(cls, pipe):
        return cls(pipe.processor, pipe._drop_idx, pipe.prompt_template_t2i, pipe.prompt_template_ti2i,
                   pipe._img_token_id)


@dataclass
class Segment:
    """One encoded prompt, after the pipeline drops the system tokens (positions are within it)."""
    ids: torch.Tensor  # (L,) token ids
    slot: torch.Tensor  # (L,) bool, VLM image slots
    keep: slice  # the part that goes into the joint sequence
    src_words: list = field(default_factory=list)  # positions of the source words (instance prompts)
    tgt_words: list = field(default_factory=list)  # positions of the target words (instance prompts)


def _span(offsets, start: int, end: int) -> list:
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]


def tokenize(p: Prompter, content: str, image: Image.Image | None = None, src: str | None = None,
             tgt: str | None = None, after_image: bool = False) -> Segment:
    """Token ids exactly as the pipeline's _get_qwen_prompt_embeds builds them, with the source/target word
    positions of an instance prompt found by character offsets (target = after "<src> with ", as fixed in mice).
    after_image: keep only the tokens after the image (instance prompts in "image" mode); else keep everything."""
    text = (p.ti2i if image is not None else p.t2i).format(content)
    tok = p.processor.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = list(tok["input_ids"]), list(tok["offset_mapping"])
    if image is not None:  # the processor expands the single <|image_pad|> to one token per slot
        proc = p.processor(text=[text], images=[image], return_tensors="pt")["input_ids"][0].tolist()
        n_slots = proc.count(p.img_token_id)
        j = ids.index(p.img_token_id)
        ids = ids[:j] + [p.img_token_id] * n_slots + ids[j + 1:]
        offsets = offsets[:j] + [offsets[j]] * n_slots + offsets[j + 1:]
    else:
        proc = p.processor(text=[text], return_tensors="pt")["input_ids"][0].tolist()
    assert proc == ids, "tokenizer and processor disagree on the prompt tokens"
    ids, offsets = ids[p.drop_idx:], offsets[p.drop_idx:]
    slot = torch.tensor(ids) == p.img_token_id

    src_words, tgt_words = [], []
    if tgt is not None:
        base = text.rindex(content)  # character offset of the content in the templated text
        prefix, sep = "Replace the ", " with "
        if src is not None and content.startswith(prefix + src + sep):
            s0 = base + len(prefix)
            t0 = s0 + len(src) + len(sep)
        else:
            s0, t0 = base + content.find(src or ""), base + content.rfind(tgt)
        end = base + len(content)  # never into the template after the content
        if src:
            src_words = _span(offsets, s0, min(s0 + len(src), end))
        tgt_words = _span(offsets, t0, min(t0 + len(tgt), end))
    keep = slice(int(slot.nonzero().max()) + 1 if after_image else 0, len(ids))
    return Segment(torch.tensor(ids), slot, keep, src_words, tgt_words)


def grid_masks(masks: list, H: int, W: int, strict: bool = True) -> list:
    """Instance masks on the H x W latent grid (nearest, as mice's resize_mask). strict: each mask loses the tokens
    also claimed by any smaller instance (mice's --strict_masks)."""
    g = [F.interpolate(torch.as_tensor(np.asarray(m), dtype=torch.float32)[None, None], size=(H, W),
                       mode="nearest")[0, 0] > 0.5 for m in masks]
    if strict and len(g) > 1:
        areas = [int(m.sum()) for m in g]
        carved = []
        for i, m in enumerate(g):
            m = m.clone()
            for j, o in enumerate(g):
                if j != i and areas[j] < areas[i]:
                    m &= ~o
            carved.append(m)
        g = carved
    return g


@dataclass
class Layout:
    """Positions in the joint sequence (what the attention processors and the steering index)."""
    n: int  # joint length
    H: int
    W: int
    src: slice  # source image block (H*W, row-major)
    tgt: slice  # target image block
    glob: torch.Tensor  # global prompt text positions
    local: list  # per instance: slice of its prompt
    inst: list  # per instance: flat grid indices (strict-carved), the same for source and target
    src_words: list  # per instance: joint positions of the source words
    tgt_words: list  # per instance: joint positions of the target words
    image_ids: torch.Tensor  # (n,) -1 text, 0 source, 1 target (the model's block ids)
    img_mask: torch.Tensor  # (len global + sum local,) bool VLM image slots: the pipeline's image_pad_mask

    def summary(self) -> dict:
        return dict(n=self.n, grid=[self.H, self.W], glob=len(self.glob),
                    local=[s.stop - s.start for s in self.local], inst=[len(i) for i in self.inst],
                    tgt_words=[len(w) for w in self.tgt_words])


def build_layout(glob: Segment, locals_: list, masks: list, H: int, W: int) -> Layout:
    """glob: the global prompt with the source image; locals_: one Segment per instance; masks: grid_masks()."""
    assert glob.keep == slice(0, len(glob.ids)), "the global prompt is kept whole"
    n_slots = int(glob.slot.sum())
    assert n_slots * SLOT == H * W, f"source has {n_slots * SLOT} latent tokens, target grid {H}x{W}"
    # joint position of every global token: each slot before it adds SLOT - 1
    before = torch.cumsum(glob.slot.long(), 0) - glob.slot.long()
    pos = torch.arange(len(glob.ids)) + (SLOT - 1) * before
    s0 = int(pos[glob.slot][0])
    src = slice(s0, s0 + H * W)
    glob_text = pos[~glob.slot]
    cur = len(glob.ids) + (SLOT - 1) * n_slots
    local, src_words, tgt_words = [], [], []
    for seg in locals_:
        start = cur - seg.keep.start  # joint position of segment index 0
        local.append(slice(cur, cur + seg.keep.stop - seg.keep.start))
        src_words.append(torch.tensor([start + i for i in seg.src_words if i >= seg.keep.start], dtype=torch.long))
        tgt_words.append(torch.tensor([start + i for i in seg.tgt_words if i >= seg.keep.start], dtype=torch.long))
        cur = local[-1].stop
    tgt = slice(cur, cur + H * W)
    n = tgt.stop
    image_ids = torch.full((n,), -1, dtype=torch.long)
    image_ids[src], image_ids[tgt] = 0, 1
    img_mask = torch.cat([glob.slot] + [seg.slot[seg.keep] for seg in locals_])
    assert not img_mask[len(glob.ids):].any(), "instance prompts carry no image slots"
    inst = [m.reshape(-1).nonzero(as_tuple=True)[0] for m in masks]
    return Layout(n, H, W, src, tgt, glob_text, local, inst, src_words, tgt_words, image_ids, img_mask)
