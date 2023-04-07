from logging import warning
import copy
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from PIL import Image
from matplotlib.figure import Figure
import io



def remap_df_cols(df, mapping):
    df.columns = [
            mapping[c]
            for c in df.columns
        ]

def fig2img(fig: Figure) -> Image.Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plt_img(xs, filename="a.png"):
    # TODO fix filename
    shape = np.array(xs.shape)
    if len(shape) == 4:
        xs = xs[0]
    shape = np.array(xs.shape)

    if (shape[0] == 3 or shape[0] == 1) and (shape[0] <= shape).all():
        xs = xs.permute(1, 2, 0)
    xs = xs.detach().cpu().numpy()

    plt.figure()
    plt.imshow(xs)
    plt.savefig(filename)


def process_row(row: str, **aux_info) -> Optional[Dict[str, Any]]:
    if not row.startswith("{"):
        row = "{" + row
    if not row.endswith("}") and not row.endswith("}\n"):
        row += "}"
    s: Dict[str, Any] = {}
    try:
        s = json.loads(row)
    except json.decoder.JSONDecodeError:
        return None
    assert (
        len(list(filter(lambda k: k in s, aux_info.keys()))) == 0
    ), f"Overlapping column names between auxilary dictionary and run results. aux_info: {aux_info}\n\nrow:{row} "
    s.update(aux_info)
    return s

