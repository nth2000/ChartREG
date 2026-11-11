import numpy as np
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_pie_wedge(
    return_value_list,
    elem_indices_list_list,
    ax_list,
    return_union=True,
    diff_thresh=0
):
    """Extract wedges from pie returns, wedge sequences, or individual wedges. None selects all wedges; indices are zero-based. Return a union or one mask per wedge."""

    # ========= basic checks =========
    if len(return_value_list) != len(elem_indices_list_list) or len(return_value_list) != len(ax_list):
        raise ValueError("return_value_list, elem_indices_list_list, and ax_list must have equal lengths.")
    if len(ax_list) == 0:
        z = np.zeros((0, 0), dtype=bool)
        return {"pie_wedge": z} if return_union else {"pie_wedge": []}

    # ========= helpers =========
    def _wedges_from_pie_ret(ret):
        """
        Normalize one pie return to a list of wedge-like artists.
        """
        if ret is None:
            return []
        # Standard: (wedges, texts) or (wedges, texts, autotexts)
        if isinstance(ret, tuple) and len(ret) >= 1:
            wedges = ret[0]
        else:
            wedges = ret

        if wedges is None:
            return []
        if isinstance(wedges, (list, tuple)):
            return [w for w in wedges if w is not None]
        # single wedge
        return [wedges]

    # ========= figure consistency =========
    fig = ax_list[0].figure
    for ax in ax_list:
        if ax.figure is not fig:
            raise ValueError("extract_pie requires all inputs to belong to the same figure.")

    fig.canvas.draw()
    restore_legends = _freeze_legends(fig)

    try:
        img_full = fig_rgba_u8(fig)
        H, W = img_full.shape[:2]

        # collect targets in stable order
        targets = []
        for ret, idxs in zip(return_value_list, elem_indices_list_list):
            wedges = _wedges_from_pie_ret(ret)
            n = len(wedges)
            if n == 0:
                continue

            if idxs is None:
                sel = list(range(n))
            else:
                sel = [int(i) for i in idxs if i is not None and 0 <= int(i) < n]

            for i in sel:
                targets.append(wedges[i])

        if not targets:
            z = np.zeros((H, W), dtype=bool)
            return {"pie_wedge": z} if return_union else {"pie_wedge": []}

        # =========================================================
        # UNION MODE
        # =========================================================
        if return_union:
            saved = []
            for wdg in targets:
                try:
                    saved.append((wdg, wdg.get_visible()))
                except Exception:
                    saved.append((wdg, True))
                try:
                    wdg.set_visible(False)
                except Exception:
                    pass

            img_hide = fig_rgba_u8(fig)

            # restore
            for wdg, vis0 in saved:
                try:
                    wdg.set_visible(vis0)
                except Exception:
                    pass

            union_mask = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)
            return {"pie_wedge": union_mask}

        # =========================================================
        # PER-INSTANCE MODE
        # =========================================================
        res = {"pie_wedge": []}

        for wdg in targets:
            try:
                vis0 = wdg.get_visible()
            except Exception:
                vis0 = True

            try:
                wdg.set_visible(False)
            except Exception:
                # Skip wedges that do not support visibility changes.
                continue

            img_hide = fig_rgba_u8(fig)

            try:
                wdg.set_visible(vis0)
            except Exception:
                pass

            m = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)
            res["pie_wedge"].append(m)

        return res

    finally:
        restore_legends()
