"""Extract individual scatter points by comparing visible and hidden renders."""
import numpy as np
import matplotlib.pyplot as plt
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_scatter(return_value_list, elem_indices_list_list,
                    return_only_union=True, diff_thresh=0):
    """Return scatter masks, restoring point coordinates even if rendering fails."""
    if len(return_value_list) != len(elem_indices_list_list):
        raise ValueError("Artist and index lists must have equal lengths.")
    fig = return_value_list[0].figure if return_value_list else plt.gcf()
    if any(sc.figure is not fig for sc in return_value_list):
        raise ValueError("All scatter collections must belong to the same figure.")
    restore_legends = _freeze_legends(fig)
    states = [(sc, sc.get_offsets().copy()) for sc in return_value_list]
    masks = []
    try:
        full = fig_rgba_u8(fig)
        for (sc, original), indices in zip(states, elem_indices_list_list):
            count = len(original)
            selected = range(count) if indices is None else sorted(set(indices))
            for index in selected:
                if not isinstance(index, (int, np.integer)) or not 0 <= index < count:
                    raise IndexError(f"Scatter index {index!r} is outside [0, {count}).")
                hidden = np.asarray(np.ma.filled(original, np.nan), dtype=float).copy()
                hidden[index] = np.nan
                sc.set_offsets(hidden)
                masks.append(diff_to_mask(full, fig_rgba_u8(fig), diff_thresh))
                sc.set_offsets(original)
        if return_only_union:
            union = np.zeros(full.shape[:2], dtype=bool)
            for mask in masks:
                union |= mask
            return {"scatter": union}
        return {"scatter": masks}
    finally:
        for sc, original in states:
            sc.set_offsets(original)
        restore_legends()
