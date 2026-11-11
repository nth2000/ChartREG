import numpy as np
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_treemap_squarify_plot(
    return_value_list,
    elem_indices_list_list,
    ax_list,
    return_only_union=True,
    diff_thresh=0,
):
    """Extract treemap rectangle masks from captured squarify returns or rectangle containers. None selects all rectangles; indices follow input order. Return unions or instance lists."""

    KEY = "treemap"

    if len(return_value_list) != len(elem_indices_list_list) or len(return_value_list) != len(ax_list):
        raise ValueError("return_value_list, elem_indices_list_list, and ax_list must have equal lengths.")

    if len(ax_list) == 0:
        z = np.zeros((0, 0), dtype=bool)
        return {KEY: z} if return_only_union else {KEY: []}

    # ---------- helpers ----------
    def _barcontainers_from_ret(ret):
        """Normalize one return into list of BarContainer-like objects."""
        if ret is None:
            return []
        # your custom tuple: (.., bar_container)
        if isinstance(ret, tuple) and len(ret) >= 2:
            ret = ret[1]

        # already BarContainer-like
        if hasattr(ret, "patches"):
            return [ret]

        # list/tuple of containers
        if isinstance(ret, (list, tuple)):
            out = []
            for x in ret:
                if x is None:
                    continue
                if isinstance(x, tuple) and len(x) >= 2 and hasattr(x[1], "patches"):
                    out.append(x[1])
                elif hasattr(x, "patches"):
                    out.append(x)
            return out
        return []

    def _collect_target_patches(ret, idxs):
        """Return list of Rectangle patches selected by idxs (or all if None)."""
        targets = []
        for bc in _barcontainers_from_ret(ret):
            try:
                patches = list(bc.patches)
            except Exception:
                # BarContainer is also iterable over patches, per docs
                patches = list(bc)

            n = len(patches)
            if n == 0:
                continue

            if idxs is None:
                sel = list(range(n))
            else:
                sel = [int(i) for i in idxs if i is not None and 0 <= int(i) < n]

            for i in sel:
                targets.append(patches[i])
        return targets

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
            raise ValueError("extract_treemap requires all inputs to belong to the same figure.")

    fig.canvas.draw()
    restore_legends = _freeze_legends(fig)

    try:
        img_full = fig_rgba_u8(fig)
        H, W = img_full.shape[:2]

        # collect all target rectangles in stable order (invocation order -> index order)
        targets = []
        for ret, idxs in zip(return_value_list, elem_indices_list_list):
            targets.extend(_collect_target_patches(ret, idxs))

        if not targets:
            z = np.zeros((H, W), dtype=bool)
            return {KEY: z} if return_only_union else {KEY: []}

        # =========================
        # UNION MODE
        # =========================
        if return_only_union:
            saved = _hide_artists(targets)
            img_hide = fig_rgba_u8(fig)
            _restore_artists(saved)

            union_mask = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)
            return {KEY: union_mask}

        # =========================
        # PER-INSTANCE MODE
        # =========================
        res = {KEY: []}
        for a in targets:
            saved = _hide_artists([a])
            img_hide = fig_rgba_u8(fig)
            _restore_artists(saved)

            res[KEY].append(diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh))

        return res

    finally:
        restore_legends()
