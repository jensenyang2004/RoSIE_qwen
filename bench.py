"""
MICE-Bench / LoMOE-Bench loader (same LoMOE.json format: mice_bench/ and lomoe_bench/ under 3D-bench/image).

Images and masks are returned at their original size: the pipeline resizes the source itself (output_size below),
and the masks are put on the latent grid later, when tokens are assigned to instances.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image


def _quoted(line: str) -> list[str]:
    return re.findall(r'"([^"]*)"', line)


def output_size(width: int, height: int, resolution: int = 1024) -> tuple[int, int]:
    """(width, height) the pipeline generates at for a source of this size: area resolution^2, the source's aspect
    ratio, sides rounded to multiples of 32 (calculate_dimensions in the pipeline). The latent grid is this / 16."""
    w = math.sqrt(resolution * resolution * width / height)
    return round(w / 32) * 32, round(w / (width / height) / 32) * 32


class Bench:
    def __init__(self, root: str):
        self.root = Path(root)
        self.entries = json.loads((self.root / "LoMOE.json").read_text())
        self.ids = sorted(self.entries, key=int)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, sample_id: str) -> dict:
        e = self.entries[sample_id]
        image = Image.open(self.root / e["image_path"]).convert("RGB")
        masks = [np.asarray(Image.open(self.root / p).convert("L")) > 127 for p in _quoted(e["mask_path"])]
        return dict(
            sample_id=sample_id,
            image=image,
            masks=masks,  # list of [H, W] bool, original size
            sources=_quoted(e.get("source_prompt", "")),
            targets=_quoted(e["fg_prompt"]),
            edit_inst_single=e["edit_inst_single"],
        )

    def shard(self, num_shards: int = 1, shard_id: int = 0) -> list[str]:
        return self.ids[shard_id::num_shards]
