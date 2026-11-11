import numpy as np
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_bar(
    return_value_list,
    elem_indices_list_list,
    ax_list,
    to_deal_markevery_list=None,
    return_only_union=True,
    diff_thresh=0
):
    """Extract bars, errorbars, and combined instances by orientation. Union mode returns the logical OR of masks for each class."""

    if len(return_value_list) != len(elem_indices_list_list) or len(return_value_list) != len(ax_list):
        raise ValueError("return_value_list, elem_indices_list_list, and ax_list must have equal lengths.")

    if to_deal_markevery_list is None:
        to_deal_markevery_list = [None] * len(return_value_list)
    if len(to_deal_markevery_list) != len(return_value_list):
        raise ValueError("to_deal_markevery_list must match the length of return_value_list.")

    if len(ax_list) == 0:
        empty = np.zeros((0, 0), dtype=bool)
        return {
            "horizontal_bar": np.zeros((0, 0), dtype=bool), "vertical_bar": np.zeros((0, 0), dtype=bool), "polar_bar": np.zeros((0, 0), dtype=bool),
            "horizontal_bar_with_errorbar": np.zeros((0, 0), dtype=bool), "vertical_bar_with_errorbar": np.zeros((0, 0), dtype=bool), "errorbar": np.zeros((0, 0), dtype=bool)
        } if return_only_union else {
            "horizontal_bar": [], "vertical_bar": [], "polar_bar": [],
            "horizontal_bar_with_errorbar": [], "vertical_bar_with_errorbar": [], "errorbar": []
        }

    # ---------------- helpers ----------------
    def _is_polar_ax(ax):
        return (getattr(ax, "name", None) == "polar") or ax.__class__.__name__.lower().startswith("polar")

    def _select_ref_segments(barlinecols):
        for lc in barlinecols:
            try:
                segs = lc.get_segments()
            except Exception:
                continue
            if segs is not None and len(segs) > 0:
                return segs
        return None

    def _segment_centers(segments):
        return np.asarray([np.mean(np.asarray(seg), axis=0) for seg in segments], dtype=float)

    def _resolve_markevery_indices(markevery, N):
        if markevery is None:
            return list(range(N))

        # bool mask
        if isinstance(markevery, (list, np.ndarray)) and len(markevery) == N:
            try:
                arr = np.asarray(markevery)
                if arr.dtype == bool:
                    return list(np.nonzero(arr)[0].astype(int))
            except Exception:
                pass

        if isinstance(markevery, (int, np.integer)):
            step = int(markevery)
            if step <= 0:
                return list(range(N))
            return list(range(0, N, step))

        if isinstance(markevery, tuple) and len(markevery) == 2:
            a, b = markevery
            if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
                start = int(a)
                step = int(b)
                if step <= 0:
                    return list(range(N))
                if start < 0:
                    start = max(0, N + start)
                return list(range(start, N, step))
            return None  # float forms

        if isinstance(markevery, slice):
            return list(range(N))[markevery]

        if isinstance(markevery, (list, np.ndarray)):
            values = np.asarray(markevery)
            if values.dtype == bool:
                if len(values) != N:
                    raise ValueError("Boolean markevery must match the number of points.")
                return list(np.flatnonzero(values))
            out = []
            for v in list(markevery):
                try:
                    i = int(v)
                except Exception:
                    continue
                if i < 0:
                    i = N + i
                if 0 <= i < N:
                    out.append(i)
            return sorted(set(out))

        return None

    def _bar_anchor_points(patches, orientation):
        pts = []
        for r in patches:
            try:
                x0, y0 = r.get_x(), r.get_y()
                w, h = r.get_width(), r.get_height()
            except Exception:
                pts.append((np.nan, np.nan))
                continue
            if orientation == "vertical":
                pts.append((float(x0 + 0.5 * w), float(y0 + h)))
            else:
                pts.append((float(x0 + w), float(y0 + 0.5 * h)))
        return np.asarray(pts, dtype=float)

    def _infer_orientation(patches, ref_segments, container_orientation=None):
        if container_orientation in ("vertical", "horizontal"):
            return container_orientation

        if ref_segments is None or len(ref_segments) == 0 or len(patches) == 0:
            hs, ws = [], []
            for p in patches:
                try:
                    hs.append(abs(p.get_height()))
                    ws.append(abs(p.get_width()))
                except Exception:
                    pass
            if len(hs) == 0 or len(ws) == 0:
                return "vertical"
            return "vertical" if (np.nanmean(hs) >= np.nanmean(ws)) else "horizontal"

        centers = _segment_centers(ref_segments)
        pts_v = _bar_anchor_points(patches, "vertical")
        pts_h = _bar_anchor_points(patches, "horizontal")
        if np.isnan(pts_v).any() or np.isnan(pts_h).any():
            return "vertical"

        dv = ((centers[:, None, :] - pts_v[None, :, :]) ** 2).sum(axis=2).min(axis=1).sum()
        dh = ((centers[:, None, :] - pts_h[None, :, :]) ** 2).sum(axis=2).min(axis=1).sum()
        return "vertical" if dv <= dh else "horizontal"

    def _infer_bar_index_to_segment_index(bar_pts, ref_segments):
        centers = _segment_centers(ref_segments)
        d2 = ((centers[:, None, :] - bar_pts[None, :, :]) ** 2).sum(axis=2)  # (ne, N)
        ne, N = d2.shape

        i_for_j = -np.ones(ne, dtype=int)   # segment j -> bar i
        used = set()
        order = np.argsort(d2.min(axis=1))
        for j in order:
            for i in np.argsort(d2[j]):
                ii = int(i)
                if ii not in used:
                    i_for_j[j] = ii
                    used.add(ii)
                    break

        j_of_i = {}
        for j, i in enumerate(i_for_j):
            if i >= 0:
                j_of_i[int(i)] = int(j)
        return j_of_i

    def _apply_errorbar_nan_mask(data_line, caplines, barlinecols, hide_js, ne):
        state = {"lc_segments": [], "cap_xy": [], "data_xy": None}
        if ne is None or ne <= 0:
            return state

        hide = sorted({int(j) for j in hide_js if j is not None and 0 <= int(j) < ne})
        if not hide:
            return state

        nan_seg = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=float)

        # stems
        for lc in barlinecols:
            try:
                segs = lc.get_segments()
            except Exception:
                continue
            if segs is None or len(segs) == 0:
                continue
            segs_orig = [np.array(s, copy=True) for s in segs]
            segs_mod = segs_orig.copy()
            for j in hide:
                if j < len(segs_mod):
                    segs_mod[j] = nan_seg
            try:
                lc.set_segments(segs_mod)
            except Exception:
                continue
            state["lc_segments"].append((lc, segs_orig))

        # caps
        # for cap in caplines:
        #     try:
        #         x = np.asarray(cap.get_xdata(), dtype=float)
        #         y = np.asarray(cap.get_ydata(), dtype=float)
        #     except Exception:
        #         continue
        #     if x.size == 0 or y.size == 0:
        #         continue
        #     L = min(len(x), len(y))
        #     if L == 0 or (L % ne != 0):
        #         continue
        #     g = L // ne
        #     x_orig, y_orig = x.copy(), y.copy()
        #     for j in hide:
        #         j0, j1 = g * j, min(L, g * (j + 1))
        #         x[j0:j1] = np.nan
        #         y[j0:j1] = np.nan
        #     try:
        #         cap.set_data(x, y)
        #     except Exception:
        #         continue
        #     state["cap_xy"].append((cap, x_orig, y_orig))

        for cap in caplines:
            # Read raw coordinates without forcing categorical values to float.
            # Coordinates may contain categorical strings or numeric values.
            x_raw = cap.get_xdata().copy()
            y_raw = cap.get_ydata().copy()

            L = len(x_raw)

            # Validate the input arrays.
            if L == 0 or (L % ne != 0):
                continue

            # Determine which coordinate axis can be modified.
            # First try converting x coordinates to float.
            # If x is numeric, modify its coordinates.
            # If x is categorical, modify the numeric y coordinates instead.

            target_axis = 'x'  # Try modifying x coordinates first.
            x_working = None
            y_working = None

            try:
                # Try creating a floating-point copy of x.
                x_working = np.asarray(x_raw, dtype=float).copy()
                y_working = y_raw  # Preserve the original y coordinates, whether numeric or categorical.
            except (ValueError, TypeError):
                # A failed float conversion indicates categorical x coordinates.
                target_axis = 'y'
                try:
                    # The y coordinates must be numeric in this case.
                    x_working = x_raw  # Preserve the original x coordinates.
                    y_working = np.asarray(y_raw, dtype=float).copy()
                except Exception:
                    # Skip invalid data if neither coordinate axis is numeric.
                    continue

            # Apply the visibility changes.
            # Insert NaNs only into the numeric working copy.
            g = L // ne

            # Select the working coordinate array.
            arr_to_modify = x_working if target_axis == 'x' else y_working

            for j in hide:
                j0, j1 = g * j, min(L, g * (j + 1))
                arr_to_modify[j0:j1] = np.nan

            # Apply changes and retain the original state.
            try:
                # set_data accepts one numeric array containing NaNs and one original array.
                cap.set_data(x_working, y_working)

                # Store the original raw coordinate arrays for restoration.
                # Restore categorical coordinates from the original raw arrays.
                state["cap_xy"].append((cap, x_raw, y_raw))
            except Exception:
                continue













        # Support the optional data line attached to an errorbar container.
        if data_line is not None:
            try:
                x = np.asarray(data_line.get_xdata(), dtype=float)
                y = np.asarray(data_line.get_ydata(), dtype=float)
            except Exception:
                x = y = None
            if x is not None and y is not None and x.size and y.size:
                L = min(len(x), len(y))
                if L and (L % ne == 0):
                    g = L // ne
                    x_orig, y_orig = x.copy(), y.copy()
                    for j in hide:
                        j0, j1 = g * j, min(L, g * (j + 1))
                        x[j0:j1] = np.nan
                        y[j0:j1] = np.nan
                    try:
                        data_line.set_data(x, y)
                        state["data_xy"] = (data_line, x_orig, y_orig)
                    except Exception:
                        pass

        return state

    def _restore_errorbar_nan_mask(state):
        for lc, segs_orig in state.get("lc_segments", []):
            try:
                lc.set_segments(segs_orig)
            except Exception:
                pass
        for cap, x_orig, y_orig in state.get("cap_xy", []):
            try:
                cap.set_data(x_orig, y_orig)
            except Exception:
                pass
        if state.get("data_xy", None) is not None:
            dl, x_orig, y_orig = state["data_xy"]
            try:
                dl.set_data(x_orig, y_orig)
            except Exception:
                pass

    # ---------------- figure sanity ----------------
    fig = ax_list[0].figure
    for ax in ax_list:
        if ax.figure is not fig:
            raise ValueError("extract_bar requires all inputs to belong to the same figure.")

    fig.canvas.draw()
    restore_legends = _freeze_legends(fig)

    try:
        # =========================
        # return_only_union == True
        # =========================
        if return_only_union:
            # Match the per-instance schema, replacing each list with its union mask.
            img_full = fig_rgba_u8(fig)
            H, W = img_full.shape[:2]
            z = np.zeros((H, W), dtype=bool)

            res_union = {
                "horizontal_bar": z.copy(),
                "vertical_bar": z.copy(),
                "polar_bar": z.copy(),
                "horizontal_bar_with_errorbar": z.copy(),
                "vertical_bar_with_errorbar": z.copy(),
                "errorbar": z.copy(),
            }

            # Collect selected bar indices and category metadata for each container.
            infos = []
            for bc, idxs, ax, mv in zip(return_value_list, elem_indices_list_list, ax_list, to_deal_markevery_list):
                patches = list(getattr(bc, "patches", list(bc)))
                n = len(patches)
                if n == 0:
                    continue

                # Select candidate element indices.
                if idxs is None:
                    cand = list(range(n))
                else:
                    cand = [int(i) for i in idxs if i is not None and 0 <= int(i) < n]

                # Apply marker sampling at the bar level.
                vis_idx = _resolve_markevery_indices(mv, n)
                if vis_idx is not None:
                    vis_set = set(vis_idx)
                    cand = [i for i in cand if i in vis_set]

                is_polar = _is_polar_ax(ax)

                # Precompute the point-to-errorbar mapping.
                err = getattr(bc, "errorbar", None)
                data_line = None
                caplines = ()
                barlinecols = ()
                ref_segments = None
                ne = 0
                j_of_i = {}

                if err is not None:
                    try:
                        data_line, caplines, barlinecols = err.lines
                        ref_segments = _select_ref_segments(barlinecols)
                        if ref_segments is not None:
                            ne = len(ref_segments)
                    except Exception:
                        err = None

                # Determine bar orientation; classify polar bars separately.
                orientation = "vertical"
                if not is_polar:
                    orientation = _infer_orientation(patches, ref_segments, getattr(bc, "orientation", None))

                # Map bar indices to errorbar segment indices when errorbars are present.
                if err is not None and ref_segments is not None and ne > 0:
                    if ne == n:
                        j_of_i = {i: i for i in range(n)}
                    else:
                        # Infer the mapping from geometry.
                        bar_pts = _bar_anchor_points(patches, orientation if not is_polar else "vertical")
                        if not np.isnan(bar_pts).any():
                            j_of_i = _infer_bar_index_to_segment_index(bar_pts, ref_segments)

                # Determine the bar category.
                if is_polar:
                    cat = "polar"
                else:
                    cat = "horizontal" if orientation == "horizontal" else "vertical"

                infos.append({
                    "bc": bc,
                    "patches": patches,
                    "n": n,
                    "cand": cand,  # This sequence may be empty.
                    "cat": cat,
                    "is_polar": is_polar,
                    "orientation": orientation,
                    "err": err,
                    "data_line": data_line,
                    "caplines": caplines,
                    "barlinecols": barlinecols,
                    "ne": ne,
                    "j_of_i": j_of_i,
                })

            # Hide bar patches by category to compute bar-only unions.
            def _bar_union_mask_for_cat(cat):
                touched = []
                for info in infos:
                    if info["cat"] != cat:
                        continue
                    for i in info["cand"]:
                        try:
                            p = info["patches"][i]
                        except Exception:
                            continue
                        try:
                            touched.append((p, p.get_visible()))
                            p.set_visible(False)
                        except Exception:
                            pass

                img_hide = fig_rgba_u8(fig)
                m = diff_to_mask(img_hide, img_full, diff_thresh=diff_thresh)

                # restore
                for p, v in touched:
                    try:
                        p.set_visible(v)
                    except Exception:
                        pass
                return m

            # Hide errorbar segments by category to compute errorbar-only unions.
            def _err_union_mask_for_cat(cat):
                states = []
                for info in infos:
                    if info["cat"] != cat:
                        continue
                    if info["err"] is None or info["ne"] <= 0 or not info["j_of_i"]:
                        continue
                    hide_js = [info["j_of_i"].get(i, None) for i in info["cand"] if i in info["j_of_i"]]
                    st = _apply_errorbar_nan_mask(
                        info["data_line"], info["caplines"], info["barlinecols"], hide_js, info["ne"]
                    )
                    states.append(st)

                img_hide = fig_rgba_u8(fig)
                m = diff_to_mask(img_hide, img_full, diff_thresh=diff_thresh)

                for st in states:
                    _restore_errorbar_nan_mask(st)
                return m

            # 1) bar-only unions
            bar_h = _bar_union_mask_for_cat("horizontal")
            bar_v = _bar_union_mask_for_cat("vertical")
            bar_p = _bar_union_mask_for_cat("polar")

            res_union["horizontal_bar"] = bar_h
            res_union["vertical_bar"] = bar_v
            res_union["polar_bar"] = bar_p

            # Errorbar-only unions, grouped by bar category.
            err_h = _err_union_mask_for_cat("horizontal")
            err_v = _err_union_mask_for_cat("vertical")
            err_p = _err_union_mask_for_cat("polar")

            err_all = err_h | err_v | err_p
            res_union["errorbar"] = err_all

            # Cartesian bars with errorbars, grouped by orientation.
            res_union["horizontal_bar_with_errorbar"] = bar_h | err_h
            res_union["vertical_bar_with_errorbar"] = bar_v | err_v

            return res_union

        # ==========================
        # Preserve per-instance extraction behavior.
        # ==========================
        res = {
            "horizontal_bar": [],
            "vertical_bar": [],
            "polar_bar": [],
            "horizontal_bar_with_errorbar": [],
            "vertical_bar_with_errorbar": [],
            "errorbar": [],
        }

        for bc, idxs, ax, mv in zip(return_value_list, elem_indices_list_list, ax_list, to_deal_markevery_list):
            patches = list(getattr(bc, "patches", list(bc)))
            n = len(patches)
            if n == 0:
                continue

            if idxs is None:
                cand = list(range(n))
            else:
                cand = [int(i) for i in idxs if i is not None and 0 <= int(i) < n]
            if not cand:
                continue

            vis_idx = _resolve_markevery_indices(mv, n)
            if vis_idx is not None:
                vis_set = set(vis_idx)
                cand = [i for i in cand if i in vis_set]
                if not cand:
                    continue

            is_polar = _is_polar_ax(ax)

            # errorbar info
            err = getattr(bc, "errorbar", None)
            ref_segments = None
            ne = 0
            data_line = None
            caplines = ()
            barlinecols = ()
            j_of_i = {}

            if err is not None:
                try:
                    data_line, caplines, barlinecols = err.lines
                    ref_segments = _select_ref_segments(barlinecols)
                    if ref_segments is not None:
                        ne = len(ref_segments)
                except Exception:
                    err = None

            orientation = _infer_orientation(patches, ref_segments, getattr(bc, "orientation", None))

            if err is not None and ref_segments is not None and ne > 0:
                if ne == n:
                    j_of_i = {i: i for i in range(n)}
                else:
                    bar_pts = _bar_anchor_points(patches, orientation)
                    if not np.isnan(bar_pts).any():
                        j_of_i = _infer_bar_index_to_segment_index(bar_pts, ref_segments)

            # (A) bar-only masks
            bar_masks = {}
            for i in cand:
                rect = patches[i]
                vis0 = rect.get_visible()

                rect.set_visible(False)
                img_base = fig_rgba_u8(fig)

                rect.set_visible(vis0)
                img_on = fig_rgba_u8(fig)

                m_bar = diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)
                bar_masks[i] = m_bar

                if is_polar:
                    res["polar_bar"].append(m_bar)
                else:
                    if orientation == "horizontal":
                        res["horizontal_bar"].append(m_bar)
                    else:
                        res["vertical_bar"].append(m_bar)

            # (B) errorbar-only masks + (C) bar+errorbar union instance
            if err is not None and ref_segments is not None and ne > 0 and len(j_of_i) > 0:
                for i in cand:
                    j = j_of_i.get(i, None)
                    if j is None:
                        continue

                    st = _apply_errorbar_nan_mask(data_line, caplines, barlinecols, [j], ne)
                    img_base = fig_rgba_u8(fig)
                    _restore_errorbar_nan_mask(st)
                    img_on = fig_rgba_u8(fig)

                    m_err = diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)
                    res["errorbar"].append(m_err)

                    m_union = bar_masks.get(i, np.zeros_like(m_err)) | m_err
                    if orientation == "horizontal":
                        res["horizontal_bar_with_errorbar"].append(m_union)
                    else:
                        res["vertical_bar_with_errorbar"].append(m_union)

        return res

    finally:
        restore_legends()
