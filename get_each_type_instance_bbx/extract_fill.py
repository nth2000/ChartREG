import numpy as np
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_fill(
    return_value_list,
    elem_indices_list_list,
    ax_list,
    return_only_union=True,
    diff_thresh=0
):
    """Extract ax.fill polygons by pixel differencing. None selects all polygons; index lists are zero-based. Return class unions or individual polygon masks."""

    if len(return_value_list) != len(elem_indices_list_list) or len(return_value_list) != len(ax_list):
        raise ValueError("return_value_list, elem_indices_list_list, and ax_list must have equal lengths.")

    if len(ax_list) == 0:
        z = np.zeros((0, 0), dtype=bool)
        return {"filled_polygon": z} if return_only_union else {"filled_polygon": []}

    # ---------- helpers ----------
    def _polys_from_fill_ret(ret):
        """Normalize fill return to list of artists (typically Polygon)."""
        if ret is None:
            return []
        if isinstance(ret, (list, tuple)):
            return [x for x in ret if x is not None]
        return [ret]

    def _hide_artists(artists):
        saved = []
        for a in artists:
            if a is None:
                continue
            try:
                saved.append((a, a.get_visible()))
            except Exception:
                saved.append((a, True))
            try:
                a.set_visible(False)
            except Exception:
                pass
        return saved

    def _restore_artists(saved):
        for a, v in saved:
            try:
                a.set_visible(v)
            except Exception:
                pass

    # ---------- figure sanity ----------
    fig = ax_list[0].figure
    for ax in ax_list:
        if ax.figure is not fig:
            raise ValueError("extract_fill requires all inputs to belong to the same figure.")

    fig.canvas.draw()
    restore_legends = _freeze_legends(fig)

    try:
        img_full = fig_rgba_u8(fig)
        H, W = img_full.shape[:2]

        # collect selected polygons in stable order
        targets = []
        for ret, idxs in zip(return_value_list, elem_indices_list_list):
            polys = _polys_from_fill_ret(ret)
            n = len(polys)
            if n == 0:
                continue

            if idxs is None:
                sel = list(range(n))
            else:
                sel = [int(i) for i in idxs if i is not None and 0 <= int(i) < n]

            for i in sel:
                targets.append(polys[i])

        if not targets:
            z = np.zeros((H, W), dtype=bool)
            return {"filled_polygon": z} if return_only_union else {"filled_polygon": []}

        # =========================
        # UNION MODE
        # =========================
        if return_only_union:
            saved = _hide_artists(targets)
            img_hide = fig_rgba_u8(fig)
            _restore_artists(saved)

            # Compute the union by comparing visible and hidden renders.
            union_mask = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)
            return {"filled_polygon": union_mask}

        # =========================
        # PER-INSTANCE MODE
        # =========================
        res = {"filled_polygon": []}
        for a in targets:
            saved = _hide_artists([a])
            img_hide = fig_rgba_u8(fig)
            _restore_artists(saved)

            m = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)
            res["filled_polygon"].append(m)

        return res

    finally:
        restore_legends()
