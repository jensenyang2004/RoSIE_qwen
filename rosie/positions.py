"""
RoPE positions of the joint sequence, and the "shared" layout of the instance prompts.

Qwen-Image-2.1 (QwenImage21Rope) gives every text token one position p on all three axes, (frame, h, w) = (p, p, p),
the axes the image grid uses too; an image block gets frame = the position where it starts, and h/w = its own grid
centred on zero, then the position jumps by max(H, W). So a prompt's place in the sequence moves it relative to the
image grid. In FLUX.2 text had its own axis (t = h = w = 0, the index on the 4th axis), so mice's stacked prompts all
sat at the same offset from the image.

    stacked  the model's own positions: instance prompt k starts after prompts 0..k-1 (where mice put them)
    shared   every instance prompt takes prompt 0's positions (right after the global prompt, where an instruction
             sits in an ordinary edit), and the target's frame is placed as if only the longest prompt came before it.
             Each instance then sees its prompt where a single-instruction edit would show it.

Shared positions need the binding: with it no token reads two instance prompts (prompt k reads itself and its
source region; target instance k reads prompt k; the background reads the global prompt), so the shared positions
never meet in one attention row.
"""
from contextlib import contextmanager

import torch

from rosie.tokens import Layout

MODES = ("stacked", "shared")


def positions(L: Layout, mode: str = "stacked") -> tuple:
    """(frame, h, w) RoPE indices [n] of the joint sequence. "stacked" reproduces QwenImage21Rope for this layout
    (check_binding.py compares them)."""
    assert mode in MODES, mode
    H, W, n = L.H, L.W, L.n
    s0, mid = L.src.start, L.tgt.start - L.src.stop
    frame = torch.empty(n, dtype=torch.long)
    frame[:s0] = torch.arange(s0)
    frame[L.src] = s0
    c = s0 + max(H, W)
    frame[L.src.stop:L.tgt.start] = c + torch.arange(mid)
    frame[L.tgt] = c + mid
    if mode == "shared" and L.local:
        p0 = int(frame[L.local[0].start])
        longest = max(sl.stop - sl.start for sl in L.local)
        for sl in L.local:
            frame[sl] = p0 + torch.arange(sl.stop - sl.start)
        assert L.local[-1].stop == L.tgt.start, "instance prompts end where the target starts"
        frame[L.tgt] = p0 + longest
    h, w = frame.clone(), frame.clone()
    gh = torch.arange(-(H - H // 2), H // 2).repeat_interleave(W)
    gw = torch.arange(-(W - W // 2), W // 2).repeat(H)
    for block in (L.src, L.tgt):
        h[block], w[block] = gh, gw
    return frame, h, w


def rope(pos_embed, frame, h, w, device) -> torch.Tensor:
    """The rotary table QwenImage21Rope.forward returns, for these indices."""
    f = [t.to(pos_embed.freqs[0].device) for t in (frame, h, w)]
    return torch.cat([pos_embed.freqs[0][f[0]], pos_embed.freqs[1][f[1]], pos_embed.freqs[2][f[2]]], dim=-1)


@contextmanager
def prompt_positions(transformer, L: Layout, mode: str = "stacked"):
    """mode "shared": wrap transformer.pos_embed for one pipeline call. Every call first checks that positions(L)
    reproduces the model's own table, so the layout and the real sequence agree."""
    assert mode in MODES, mode
    if mode == "stacked":
        yield
        return
    pe = transformer.pos_embed
    original = pe.forward
    stacked, shared = positions(L, "stacked"), positions(L, "shared")

    def forward(img_shapes, image_pad_mask, device):
        table = original(img_shapes, image_pad_mask, device)  # also moves pe.freqs to the device
        if table.shape[0] != L.n or not torch.equal(table, rope(pe, *stacked, device)):
            raise RuntimeError("the model's RoPE positions differ from the layout's")
        return rope(pe, *shared, device)

    pe.forward = forward  # instance attribute shadows the method
    try:
        yield
    finally:
        del pe.forward
