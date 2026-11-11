import numpy as np
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_bxp(
    return_value_list,
    elem_indices_list_list,
    ax_list,
    return_only_union=True,
    diff_thresh=0,
):
    """Extract complete boxes, box patches, and median lines from boxplot/bxp returns. None selects all boxes. Inputs must belong to one figure; return unions or instance lists."""

    if len(return_value_list) != len(elem_indices_list_list) or len(return_value_list) != len(ax_list):
        raise ValueError("return_value_list, elem_indices_list_list, and ax_list must have equal lengths.")

    if len(ax_list) == 0:
        z = np.zeros((0, 0), dtype=bool)
        if return_only_union:
            return {"boxplot": z, "median_line": z, "box_patch": z}
        return {"boxplot": [], "median_line": [], "box_patch": []}

    # ---------- helpers ----------
    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return [a for a in x if a is not None]
        return [x]

    def _get_n_boxes(bpdict):
        # Prefer boxes/medians lengths (typical n)
        boxes = _as_list(bpdict.get("boxes", []))
        meds = _as_list(bpdict.get("medians", []))
        means = _as_list(bpdict.get("means", []))
        fliers = _as_list(bpdict.get("fliers", []))
        n = 0
        for L in (len(boxes), len(meds), len(means), len(fliers)):
            n = max(n, L)
        return n

    def _two_per(lst, i):
        j0 = 2 * i
        j1 = j0 + 2
        if len(lst) >= j1:
            return [lst[j0], lst[j0 + 1]]
        return []

    def _one_or_two_means(means, i, n):
        # most common: len(means)==n; sometimes ==2n; else best-effort single
        if i < 0:
            return []
        if len(means) == n:
            return [means[i]] if i < len(means) else []
        if len(means) == 2 * n:
            return _two_per(means, i)
        return [means[i]] if i < len(means) else []

    def _artists_for_instance(bpdict, i):
        """
        Returns:
          - artists_boxall: all components except fliers for index i
          - artist_median: median line (single)
          - artist_boxpatch: box patch (single)
        """
        boxes = _as_list(bpdict.get("boxes", []))
        medians = _as_list(bpdict.get("medians", []))
        whiskers = _as_list(bpdict.get("whiskers", []))
        caps = _as_list(bpdict.get("caps", []))
        means = _as_list(bpdict.get("means", []))
        # Exclude fliers from complete box instances.

        n = _get_n_boxes(bpdict)

        arts_boxall = []
        if i < len(boxes):
            arts_boxall.append(boxes[i])
        if i < len(medians):
            arts_boxall.append(medians[i])

        arts_boxall.extend(_two_per(whiskers, i))
        arts_boxall.extend(_two_per(caps, i))
        arts_boxall.extend(_one_or_two_means(means, i, n))

        art_median = medians[i] if i < len(medians) else None
        art_box = boxes[i] if i < len(boxes) else None
        return arts_boxall, art_median, art_box

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

    # ---------- figure consistency ----------
    fig = ax_list[0].figure
    for ax in ax_list:
        if ax.figure is not fig:
            raise ValueError("extract_bxp requires all inputs to belong to the same figure.")

    fig.canvas.draw()
    restore_legends = _freeze_legends(fig)
    try:
        img_full = fig_rgba_u8(fig)
        H, W = img_full.shape[:2]

        # gather selected indices per bxp result
        plans = []
        for bpdict, idxs in zip(return_value_list, elem_indices_list_list):
            if not isinstance(bpdict, dict):
                continue
            n = _get_n_boxes(bpdict)
            if n <= 0:
                continue
            if idxs is None:
                sel = list(range(n))
            else:
                sel = [int(i) for i in idxs if i is not None and 0 <= int(i) < n]
            plans.append((bpdict, sel))

        if return_only_union:
            z = np.zeros((H, W), dtype=bool)
            out = {"boxplot": z.copy(), "median_line": z.copy(), "box_patch": z.copy()}

            # Union of complete boxplot instances.
            to_hide = []
            for bpdict, sel in plans:
                for i in sel:
                    arts_boxall, _, _ = _artists_for_instance(bpdict, i)
                    to_hide.extend(arts_boxall)
            saved = _hide_artists(to_hide)
            img_hide = fig_rgba_u8(fig)
            _restore_artists(saved)
            out["boxplot"] = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)

            # --- union: boxPlot (median line) ---
            to_hide = []
            for bpdict, sel in plans:
                for i in sel:
                    _, art_median, _ = _artists_for_instance(bpdict, i)
                    if art_median is not None:
                        to_hide.append(art_median)
            saved = _hide_artists(to_hide)
            img_hide = fig_rgba_u8(fig)
            _restore_artists(saved)
            out["median_line"] = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)

            # --- union: BoxPlot(Boxpatch) ---
            to_hide = []
            for bpdict, sel in plans:
                for i in sel:
                    _, _, art_box = _artists_for_instance(bpdict, i)
                    if art_box is not None:
                        to_hide.append(art_box)
            saved = _hide_artists(to_hide)
            img_hide = fig_rgba_u8(fig)
            _restore_artists(saved)
            out["box_patch"] = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)

            return out

        # return_only_union == False : per-instance masks
        res = {"boxplot": [], "median_line": [], "box_patch": []}

        for bpdict, sel in plans:
            for i in sel:
                arts_boxall, art_median, art_box = _artists_for_instance(bpdict, i)

                # Complete box instances exclude fliers.
                saved = _hide_artists(arts_boxall)
                img_hide = fig_rgba_u8(fig)
                _restore_artists(saved)
                res["boxplot"].append(diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh))

                # boxPlot instance: median line only
                if art_median is not None:
                    saved = _hide_artists([art_median])
                    img_hide = fig_rgba_u8(fig)
                    _restore_artists(saved)
                    res["median_line"].append(diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh))

                # BoxPlot(Boxpatch) instance: box patch only
                if art_box is not None:
                    saved = _hide_artists([art_box])
                    img_hide = fig_rgba_u8(fig)
                    _restore_artists(saved)
                    res["box_patch"].append(diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh))

        return res

    finally:
        restore_legends()
