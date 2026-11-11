import numpy as np
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_stackplot(
    return_value_list,
    elem_indices_list_list,
    ax_list,
    return_only_union=True,
    diff_thresh=0
):
    """Extract stacked area masks from PolyCollections. None selects all areas; index lists are zero-based. Inputs must share a figure. Return a union or one mask per area."""

    KEY = "stacked_area"

    if len(return_value_list) != len(elem_indices_list_list) or len(return_value_list) != len(ax_list):
        raise ValueError("return_value_list, elem_indices_list_list, and ax_list must have equal lengths.")

    if len(ax_list) == 0:
        z = np.zeros((0, 0), dtype=bool)
        return {KEY: z} if return_only_union else {KEY: []}

    def _areas_from_stackplot_ret(ret):
        """
        Normalize one stackplot return to list[PolyCollection].
        stackplot returns list of PolyCollection (one per area).:contentReference[oaicite:3]{index=3}
        """
        if ret is None:
            return []
        if isinstance(ret, (list, tuple)):
            return [x for x in ret if x is not None]
        return [ret]

    # ---- figure consistency ----
    fig = ax_list[0].figure
    for ax in ax_list:
        if ax.figure is not fig:
            raise ValueError("extract_stackplot requires all inputs to belong to the same figure.")

    fig.canvas.draw()
    restore_legends = _freeze_legends(fig)

    try:
        img_full = fig_rgba_u8(fig)
        H, W = img_full.shape[:2]

        # Collect PolyCollections in stable invocation and element order.
        targets = []
        for ret, idxs in zip(return_value_list, elem_indices_list_list):
            areas = _areas_from_stackplot_ret(ret)
            n = len(areas)
            if n == 0:
                continue

            if idxs is None:
                sel = list(range(n))
            else:
                sel = [int(i) for i in idxs if i is not None and 0 <= int(i) < n]

            for i in sel:
                targets.append(areas[i])

        if not targets:
            z = np.zeros((H, W), dtype=bool)
            return {KEY: z} if return_only_union else {KEY: []}

        # =========================
        # UNION MODE
        # =========================
        if return_only_union:
            saved = []
            for a in targets:
                try:
                    saved.append((a, a.get_visible()))
                except Exception:
                    saved.append((a, True))
                try:
                    a.set_visible(False)  # Artist.set_visible
                except Exception:
                    pass

            img_hide = fig_rgba_u8(fig)

            # restore
            for a, v in saved:
                try:
                    a.set_visible(v)
                except Exception:
                    pass

            union_mask = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)
            return {KEY: union_mask}

        # =========================
        # PER-INSTANCE MODE
        # =========================
        res = {KEY: []}
        for a in targets:
            try:
                vis0 = a.get_visible()
            except Exception:
                vis0 = True

            try:
                a.set_visible(False)
            except Exception:
                continue

            img_hide = fig_rgba_u8(fig)

            try:
                a.set_visible(vis0)
            except Exception:
                pass

            m = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)
            res[KEY].append(m)

        return res

    finally:
        restore_legends()
