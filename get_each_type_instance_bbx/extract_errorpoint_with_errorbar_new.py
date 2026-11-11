import numpy as np
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_errorpoint_with_errorbar(
    return_value_list,
    elem_indices_list_list,
    ax_list,
    to_deal_errorevery_list,
    return_only_union=True,
    diff_thresh=0
):
    """Extract errorbar components by hiding the container and restoring selected elements. Map point indices to segments using errorevery. Marker-only series are scatter points; full-line masks require a drawn line."""

    # ========= basic checks =========
    if len(return_value_list) != len(elem_indices_list_list) or len(return_value_list) != len(ax_list):
        raise ValueError("return_value_list, elem_indices_list_list, and ax_list must have equal lengths.")
    if to_deal_errorevery_list is None:
        to_deal_errorevery_list = [None] * len(return_value_list)
    if len(to_deal_errorevery_list) != len(return_value_list):
        raise ValueError("to_deal_errorevery_list must match the length of return_value_list.")

    if len(ax_list) == 0:
        if return_only_union:
            return np.zeros((0, 0), dtype=bool)
        return {
            "point_with_errorbar": [],
            "line_marker": [],
            "scatter": [],
            "errorbar_line": [],
            "errorbar": [],
        }

    # ========= helpers =========
    def _artists_from_errc(errc):
        try:
            data_line, caplines, barlinecols = errc.lines
        except Exception:
            return None, [], []
        return (
            data_line,
            list(caplines) if caplines is not None else [],
            list(barlinecols) if barlinecols is not None else [],
        )

    def _save_state_errc(errc):
        data_line, caplines, barlinecols = _artists_from_errc(errc)
        st = {
            "data_line": data_line,
            "data_vis": None,
            "data_ls": None,
            "data_markevery": None,
            "caplines": caplines,
            "cap_vis": [],
            "cap_xy": [],
            "barlinecols": barlinecols,
            "lc_vis": [],
            "lc_segments": [],
        }

        if data_line is not None:
            try:
                st["data_vis"] = data_line.get_visible()
                st["data_ls"] = data_line.get_linestyle()
                st["data_markevery"] = data_line.get_markevery()
            except Exception:
                pass

        for cap in caplines:
            try:
                st["cap_vis"].append(cap.get_visible())
            except Exception:
                st["cap_vis"].append(True)
            try:
                x = np.asarray(cap.get_xdata())
                y = np.asarray(cap.get_ydata())
                st["cap_xy"].append((cap, x.copy(), y.copy()))
            except Exception:
                st["cap_xy"].append((cap, None, None))

        for lc in barlinecols:
            try:
                st["lc_vis"].append(lc.get_visible())
            except Exception:
                st["lc_vis"].append(True)
            try:
                segs = lc.get_segments()
                if segs is None:
                    st["lc_segments"].append((lc, None))
                else:
                    st["lc_segments"].append((lc, [np.array(s, copy=True) for s in segs]))
            except Exception:
                st["lc_segments"].append((lc, None))

        return st

    def _restore_state_errc(st):
        data_line = st.get("data_line", None)
        if data_line is not None:
            if st.get("data_vis", None) is not None:
                try:
                    data_line.set_visible(st["data_vis"])
                except Exception:
                    pass
            if st.get("data_ls", None) is not None:
                try:
                    data_line.set_linestyle(st["data_ls"])
                except Exception:
                    pass
            try:
                data_line.set_markevery(st.get("data_markevery", None))
            except Exception:
                pass

        for cap, vis in zip(st.get("caplines", []), st.get("cap_vis", [])):
            try:
                cap.set_visible(vis)
            except Exception:
                pass
        for cap, x, y in st.get("cap_xy", []):
            if x is None or y is None:
                continue
            try:
                cap.set_data(x, y)
            except Exception:
                pass

        for lc, vis in zip(st.get("barlinecols", []), st.get("lc_vis", [])):
            try:
                lc.set_visible(vis)
            except Exception:
                pass
        for lc, segs in st.get("lc_segments", []):
            if segs is None:
                continue
            try:
                lc.set_segments(segs)
            except Exception:
                pass

    def _set_all_invisible(errc):
        data_line, caplines, barlinecols = _artists_from_errc(errc)
        if data_line is not None:
            try:
                data_line.set_visible(False)
            except Exception:
                pass
        for cap in caplines:
            try:
                cap.set_visible(False)
            except Exception:
                pass
        for lc in barlinecols:
            try:
                lc.set_visible(False)
            except Exception:
                pass

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

    def _infer_point_to_segment_map(x, y, ref_segments):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        pts = np.stack([x, y], axis=1)  # (N,2)

        centers = _segment_centers(ref_segments)  # (ne,2)
        d2 = ((centers[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)  # (ne,N)

        ne, N = d2.shape
        i_for_j = -np.ones(ne, dtype=int)
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

    def _resolve_markevery_indices(markevery, N):
        if isinstance(markevery, (float, np.floating)) or (
            isinstance(markevery, tuple) and len(markevery) == 2
            and any(isinstance(value, (float, np.floating)) for value in markevery)
        ):
            raise ValueError("Distance-based float markevery is unsupported; use discrete indices or an integer step.")
        if markevery is None:
            return list(range(N))

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

    def _resolve_errorevery_indices(errorevery, N):
        """Resolve errorbar point indices in ascending order: N selects x[::N], and (start, N) selects x[start::N]."""
        if errorevery is None:
            # Default errorevery to 1, which includes every point.
            return list(range(N))

        if isinstance(errorevery, (int, np.integer)):
            step = int(errorevery)
            if step <= 0:
                return list(range(N))
            return list(range(0, N, step))

        if isinstance(errorevery, tuple) and len(errorevery) == 2:
            start, step = errorevery
            if not isinstance(start, (int, np.integer)) or not isinstance(step, (int, np.integer)):
                return list(range(N))
            start = int(start)
            step = int(step)
            if step <= 0:
                return list(range(N))
            if start < 0:
                start = max(0, N + start)
            return list(range(start, N, step))

        # Fall back to selecting all points for unsupported sampling types.
        return list(range(N))

    def _marker_is_drawn(data_line, visible_override=True):
        """Check that a marker exists, has positive size, and was originally visible."""
        if data_line is None:
            return False
        if not bool(visible_override):
            return False
        try:
            marker = data_line.get_marker()
            if marker in (None, "", " ", "None"):
                return False
            ms = float(data_line.get_markersize() or 0.0)
            if ms <= 0:
                return False
            a = data_line.get_alpha()
            if a == 0:
                return False
            return True
        except Exception:
            return False

    # Determine whether the artist actually draws a line.
    def _line_is_drawn(data_line, visible_override=True, ls_override=None):
        """Check visibility, line style, positive linewidth, and nonzero alpha; None alpha is valid."""
        if data_line is None:
            return False
        if not bool(visible_override):
            return False
        try:
            ls = ls_override if ls_override is not None else data_line.get_linestyle()
            if ls is None:
                return False
            if isinstance(ls, str) and ls.strip().lower() == "none":
                return False
            if isinstance(ls, str) and ls.strip() == "":
                return False
            lw = float(data_line.get_linewidth() or 0.0)
            if lw <= 0:
                return False
            a = data_line.get_alpha()
            if a is not None and float(a) == 0.0:
                return False
            return True
        except Exception:
            return False

    def _apply_marker_only(data_line, keep_is):
        if data_line is None:
            return
        try:
            data_line.set_visible(True)
        except Exception:
            pass
        try:
            data_line.set_linestyle("None")
        except Exception:
            try:
                data_line.set_linestyle(None)
            except Exception:
                pass
        try:
            data_line.set_markevery(list(keep_is))
        except Exception:
            pass

    def _apply_errorbar_only(errc, keep_js, ne):
        _, caplines, barlinecols = _artists_from_errc(errc)
        keep = sorted({int(j) for j in keep_js if j is not None and 0 <= int(j) < ne})
        keep_set = set(keep)

        nan_seg = np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=float)

        for lc in barlinecols:
            try:
                segs_orig = lc.get_segments()
            except Exception:
                segs_orig = None
            if segs_orig is None or len(segs_orig) == 0:
                continue

            segs_mod = []
            for j, s in enumerate(segs_orig):
                segs_mod.append(np.asarray(s) if j in keep_set else nan_seg)

            try:
                lc.set_segments(segs_mod)
                lc.set_visible(True)
            except Exception:
                pass

        # for cap in caplines:
        #     try:
        #         x = np.asarray(cap.get_xdata(),dtype=float)
        #         y = np.asarray(cap.get_ydata(),dtype=float)
        #     except Exception:
        #         continue
        #     L = min(len(x), len(y))
        #     if L == 0 or (L % ne != 0):
        #         continue
        #     g = L // ne
        #
        #     x2, y2 = x.copy(), y.copy()
        #     for j in range(ne):
        #         if j in keep_set:
        #             continue
        #         j0, j1 = g * j, min(L, g * (j + 1))
        #         x2[j0:j1] = np.nan
        #         y2[j0:j1] = np.nan
        #
        #     try:
        #         cap.set_data(x2, y2)
        #         cap.set_visible(True)
        #     except Exception:
        #         pass

        for cap in caplines:
            # Preserve the original coordinate arrays.
            x_raw = cap.get_xdata()
            y_raw = cap.get_ydata()

            # Determine the number of data points.
            L = len(x_raw)
            if L == 0: continue

            # Initialize replacement coordinates from the originals.
            x_final = x_raw
            y_final = y_raw

            # Select the numeric axis in which hidden points can be replaced by NaN.
            # NaNs can be inserted into whichever coordinate axis is numeric.

            target_axis = None  # 'x' or 'y'

            # Check whether x coordinates are numeric.
            try:
                x_float = np.asarray(x_raw, dtype=float)
                # Successful conversion means x is numeric.
                target_axis = 'x'
            except (ValueError, TypeError):
                # Categorical x coordinates imply a numeric y axis.
                # The y axis must be numeric in this case.
                target_axis = 'y'

            # Check y as well after confirming that x is numeric.
            # One numeric coordinate axis is sufficient for hiding points with NaN.

            # Apply the visibility changes.
            # Group and slice the selected components.
            g = L // ne

            # False entries identify points to hide.
            mask = np.ones(L, dtype=bool)

            for j in range(ne):
                if j in keep_set:
                    continue  # Keep this component unchanged.

                # Determine which point indices to hide.
                j0, j1 = g * j, min(L, g * (j + 1))
                mask[j0:j1] = False

            # Modify coordinates on the selected axis.
            if target_axis == 'x':
                # Make a floating-point copy of x.
                x_new = np.asarray(x_raw, dtype=float).copy()
                # Replace hidden coordinates with NaN.
                x_new[~mask] = np.nan
                x_final = x_new

            elif target_axis == 'y':
                # Make a floating-point copy of y.
                y_new = np.asarray(y_raw, dtype=float).copy()
                y_new[~mask] = np.nan
                y_final = y_new

            try:
                cap.set_data(x_final, y_final)
                cap.set_visible(True)
            except Exception:
                pass

    # ========= figure consistency =========
    fig = ax_list[0].figure
    for ax in ax_list:
        if ax.figure is not fig:
            raise ValueError("extract_errorbar requires all inputs to belong to the same figure.")
    fig.canvas.draw()
    restore_legends = _freeze_legends(fig)

    try:
        # =========================================================
        # UNION MODE
        # =========================================================
        if return_only_union:
            states = [_save_state_errc(errc) for errc in return_value_list]

            # base: hide all selected containers
            for errc in return_value_list:
                _set_all_invisible(errc)
            img_base = fig_rgba_u8(fig)

            # on: add back selected point+errorbar
            for errc, idxs, st, ev in zip(return_value_list, elem_indices_list_list, states, to_deal_errorevery_list):
                data_line, _, barlinecols = _artists_from_errc(errc)
                if data_line is None:
                    continue

                # A point-with-errorbar target requires a marker.
                if not _marker_is_drawn(data_line, visible_override=st.get("data_vis", True)):
                    continue

                # N points
                try:
                    x = np.asarray(data_line.get_xdata())
                    y = np.asarray(data_line.get_ydata())
                except Exception:
                    continue
                N = min(len(x), len(y))
                if N <= 0:
                    continue

                # indices to consider
                if idxs is None:
                    cand_is = list(range(N))
                else:
                    cand_is = [int(i) for i in idxs if i is not None and 0 <= int(i) < N]
                if not cand_is:
                    continue

                # markevery gating
                vis_idx = _resolve_markevery_indices(st.get("data_markevery", None), N)
                if vis_idx is not None:
                    vis_set = set(vis_idx)
                    cand_is = [i for i in cand_is if i in vis_set]
                    if not cand_is:
                        continue

                # segments
                ref_segments = _select_ref_segments(barlinecols)
                if ref_segments is None or len(ref_segments) == 0:
                    continue
                ne = len(ref_segments)

                # Map point indices to errorbar segments using errorevery.
                err_points = _resolve_errorevery_indices(ev, N)
                if ne == len(err_points):
                    j_of_i = {int(i): int(j) for j, i in enumerate(err_points)}
                elif ne == N:
                    j_of_i = {i: i for i in range(N)}
                else:
                    # Fall back to geometry when lengths do not match.
                    j_of_i = _infer_point_to_segment_map(x, y, ref_segments)

                qualify_is = [i for i in cand_is if i in j_of_i]  # This point has an associated errorbar segment.
                if not qualify_is:
                    continue

                keep_js = [j_of_i[i] for i in qualify_is]
                _apply_marker_only(data_line, qualify_is)
                _apply_errorbar_only(errc, keep_js, ne)

            img_on = fig_rgba_u8(fig)
            union_mask = diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)

            for st in states:
                _restore_state_errc(st)
            return {"point_with_errorbar": union_mask}

        # =========================================================
        # Per-instance mode uses the same errorevery mapping.
        # =========================================================
        res = {
            "point_with_errorbar": [],
            "line_marker": [],
            "scatter": [],
            "errorbar_line": [],
            "errorbar": [],
        }

        img_full = fig_rgba_u8(fig)

        for errc, idxs, ev in zip(return_value_list, elem_indices_list_list, to_deal_errorevery_list):
            st = _save_state_errc(errc)
            data_line, _, barlinecols = _artists_from_errc(errc)

            # Determine whether the data line actually draws a line.
            line_drawn = _line_is_drawn(
                data_line,
                visible_override=st.get("data_vis", True),
                ls_override=st.get("data_ls", None)
            )

            _set_all_invisible(errc)
            img_hide_all = fig_rgba_u8(fig)

            # Do not emit a full-line instance when the artist does not draw a line.
            if line_drawn:
                res["errorbar_line"].append(
                    diff_to_mask(img_hide_all, img_full, diff_thresh=diff_thresh)
                )

            img_base = img_hide_all

            if data_line is None:
                # fmt='none' still contains valid errorbar stems and caps.
                ref_segments = _select_ref_segments(barlinecols)
                ne = len(ref_segments) if ref_segments is not None else 0
                count = getattr(errc, "_mask_point_count", ne)
                points = _resolve_errorevery_indices(ev, count)
                selected = set(points) if idxs is None else set(idxs)
                try:
                    for j, point in enumerate(points[:ne]):
                        if point not in selected:
                            continue
                        _restore_state_errc(st)
                        _set_all_invisible(errc)
                        _apply_errorbar_only(errc, [j], ne)
                        res["errorbar"].append(diff_to_mask(fig_rgba_u8(fig), img_base, diff_thresh))
                finally:
                    _restore_state_errc(st)
                continue

            try:
                x = np.asarray(data_line.get_xdata())
                y = np.asarray(data_line.get_ydata())
            except Exception:
                _restore_state_errc(st)
                continue
            N = min(len(x), len(y))
            if N <= 0:
                _restore_state_errc(st)
                continue

            if idxs is None:
                cand_is = list(range(N))
            else:
                cand_is = [int(i) for i in idxs if i is not None and 0 <= int(i) < N]
            if not cand_is:
                _restore_state_errc(st)
                continue

            marker_ok = _marker_is_drawn(data_line, visible_override=st.get("data_vis", True))
            vis_idx = _resolve_markevery_indices(st.get("data_markevery", None), N) if marker_ok else []
            vis_set = set(vis_idx) if (marker_ok and vis_idx is not None) else None

            marker_is = []
            if marker_ok:
                marker_is = cand_is[:] if vis_set is None else [i for i in cand_is if i in vis_set]

            ref_segments = _select_ref_segments(barlinecols)
            j_of_i = {}
            ne = 0
            if ref_segments is not None and len(ref_segments) > 0:
                ne = len(ref_segments)

                err_points = _resolve_errorevery_indices(ev, N)
                if ne == len(err_points):
                    j_of_i = {int(i): int(j) for j, i in enumerate(err_points)}
                elif ne == N:
                    j_of_i = {i: i for i in range(N)}
                else:
                    j_of_i = _infer_point_to_segment_map(x, y, ref_segments)

            err_is = [i for i in cand_is if i in j_of_i]
            combo_is = [i for i in marker_is if i in j_of_i]

            # Classify markers as scatter points when no line is drawn.
            for i in marker_is:
                _restore_state_errc(st)
                _set_all_invisible(errc)
                _apply_marker_only(data_line, [i])
                img_on = fig_rgba_u8(fig)
                if line_drawn:
                    res["line_marker"].append(diff_to_mask(img_on, img_base, diff_thresh=diff_thresh))
                else:
                    res["scatter"].append(diff_to_mask(img_on, img_base, diff_thresh=diff_thresh))

            if ne > 0 and len(j_of_i) > 0:
                for i in err_is:
                    j = j_of_i.get(i, None)
                    if j is None:
                        continue
                    _restore_state_errc(st)
                    _set_all_invisible(errc)
                    _apply_errorbar_only(errc, [j], ne)
                    img_on = fig_rgba_u8(fig)
                    res["errorbar"].append(diff_to_mask(img_on, img_base, diff_thresh=diff_thresh))

                for i in combo_is:
                    j = j_of_i.get(i, None)
                    if j is None:
                        continue
                    _restore_state_errc(st)
                    _set_all_invisible(errc)
                    _apply_marker_only(data_line, [i])
                    _apply_errorbar_only(errc, [j], ne)
                    img_on = fig_rgba_u8(fig)
                    res["point_with_errorbar"].append(
                        diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)
                    )

            _restore_state_errc(st)

        return res

    finally:
        restore_legends()
