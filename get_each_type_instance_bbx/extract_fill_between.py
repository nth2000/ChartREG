import numpy as np
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_fill_between(
    return_value_list,
    elem_indices_list_list,
    ax_list,
    return_only_union=True,
    diff_thresh=0,
):
    """Extract fill_between polygon paths using pixel differences. None selects all paths; index lists are zero-based. Inputs must share a figure. Return unions or per-path masks."""

    KEY = "fill_between"

    if len(return_value_list) != len(elem_indices_list_list) or len(return_value_list) != len(ax_list):
        raise ValueError("return_value_list, elem_indices_list_list, and ax_list must have equal lengths.")

    if len(ax_list) == 0:
        z = np.zeros((0, 0), dtype=bool)
        return {KEY: z} if return_only_union else {KEY: []}

    # -------- helpers --------
    def _collections_from_ret(ret):
        """
        Normalize a fill_between return to a list of collection-like artists.
        fill_between normally returns a FillBetweenPolyCollection.
        """
        if ret is None:
            return []
        # already a Collection-like
        if hasattr(ret, "get_paths") or hasattr(ret, "set_paths") or hasattr(ret, "set_verts"):
            return [ret]
        # list/tuple of collections
        if isinstance(ret, (list, tuple)):
            out = []
            for x in ret:
                if x is None:
                    continue
                if hasattr(x, "get_paths") or hasattr(x, "set_paths") or hasattr(x, "set_verts"):
                    out.append(x)
            return out
        return []

    def _empty_path():
        from matplotlib.path import Path
        return Path(np.empty((0, 2), dtype=float))

    def _get_paths_safe(coll):
        try:
            paths = coll.get_paths()  # Collections have get_paths().
            if paths is None:
                return None
            return list(paths)
        except Exception:
            return None

    def _set_paths_safe(coll, paths):
        # Prefer set_paths (documented in older collections API; still present on Collections).
        if hasattr(coll, "set_paths"):
            try:
                coll.set_paths(paths)
                return True
            except Exception:
                pass
        # Fallback: set_verts (drops codes; best-effort)
        if hasattr(coll, "set_verts"):
            try:
                verts_list = []
                for p in paths:
                    try:
                        verts_list.append(np.asarray(p.vertices, dtype=float))
                    except Exception:
                        verts_list.append(np.empty((0, 2), dtype=float))
                coll.set_verts(verts_list)
                return True
            except Exception:
                pass
        return False

    def _hide_by_paths(coll, hide_indices):
        """
        Hide selected path indices by replacing those paths with an empty Path.
        Returns: state dict to restore, or None if unsupported.
        """
        orig_paths = _get_paths_safe(coll)
        if orig_paths is None:
            return None  # can't do per-path; caller may fall back to set_visible

        m = len(orig_paths)
        hide = sorted({int(i) for i in hide_indices if i is not None and 0 <= int(i) < m})
        if not hide:
            return {"coll": coll, "paths": orig_paths, "did": False}

        new_paths = orig_paths.copy()
        ep = _empty_path()
        for i in hide:
            new_paths[i] = ep

        ok = _set_paths_safe(coll, new_paths)
        return {"coll": coll, "paths": orig_paths, "did": ok}

    def _restore_paths(state):
        if not state:
            return
        if not state.get("did", False):
            return
        coll = state["coll"]
        orig_paths = state["paths"]
        _set_paths_safe(coll, orig_paths)

    # -------- figure sanity --------
    fig = ax_list[0].figure
    for ax in ax_list:
        if ax.figure is not fig:
            raise ValueError("extract_fill_between requires all inputs to belong to the same figure.")

    fig.canvas.draw()
    restore_legends = _freeze_legends(fig)

    try:
        img_full = fig_rgba_u8(fig)
        H, W = img_full.shape[:2]

        # -------- collect (coll, selected_path_indices) plan --------
        plans = []
        for ret, idxs in zip(return_value_list, elem_indices_list_list):
            colls = _collections_from_ret(ret)
            for coll in colls:
                paths = _get_paths_safe(coll)
                if paths is None:
                    # fallback mode: treat the whole collection as a single element (index 0)
                    if idxs is None:
                        sel = [0]
                    else:
                        sel = [0] if any(int(i) == 0 for i in idxs if i is not None) else []
                    plans.append(("visible_fallback", coll, sel))
                    continue

                m = len(paths)
                if m == 0:
                    continue

                if idxs is None:
                    sel = list(range(m))
                else:
                    sel = [int(i) for i in idxs if i is not None and 0 <= int(i) < m]
                plans.append(("paths", coll, sel))

        # nothing selected
        if not plans:
            z = np.zeros((H, W), dtype=bool)
            return {KEY: z} if return_only_union else {KEY: []}

        # =========================================================
        # UNION MODE
        # =========================================================
        if return_only_union:
            states = []
            saved_vis = []

            for mode, coll, sel in plans:
                if not sel:
                    continue
                if mode == "paths":
                    st = _hide_by_paths(coll, sel)
                    if st is not None:
                        states.append(st)
                else:
                    # fallback: hide whole collection
                    try:
                        saved_vis.append((coll, coll.get_visible()))
                    except Exception:
                        saved_vis.append((coll, True))
                    try:
                        coll.set_visible(False)
                    except Exception:
                        pass

            img_hide = fig_rgba_u8(fig)

            # restore
            for st in states:
                _restore_paths(st)
            for coll, v in saved_vis:
                try:
                    coll.set_visible(v)
                except Exception:
                    pass

            union_mask = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)
            return {KEY: union_mask}

        # =========================================================
        # PER-INSTANCE MODE (each selected path one mask)
        # =========================================================
        res = {KEY: []}

        for mode, coll, sel in plans:
            if not sel:
                continue

            if mode == "paths":
                # for each path index: hide only that one and diff
                for i in sel:
                    st = _hide_by_paths(coll, [i])
                    if st is None:
                        continue
                    img_hide = fig_rgba_u8(fig)
                    _restore_paths(st)
                    res[KEY].append(diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh))
            else:
                # fallback: whole collection as one instance (if selected)
                try:
                    vis0 = coll.get_visible()
                except Exception:
                    vis0 = True
                try:
                    coll.set_visible(False)
                except Exception:
                    continue
                img_hide = fig_rgba_u8(fig)
                try:
                    coll.set_visible(vis0)
                except Exception:
                    pass
                res[KEY].append(diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh))

        return res

    finally:
        restore_legends()
