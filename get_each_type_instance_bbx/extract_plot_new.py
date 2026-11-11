import numpy as np
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_plotpoint_with_plotline(
    return_value_list,
    elem_indices_list_list,
    ax_list,
    return_only_union=True,
    diff_thresh=0
):
    """Extract complete line series and individual markers by hiding and restoring artists. Classify marker-only series as scatter; keep Cartesian and polar labels separate. Return unions or instance lists."""

    # ========= basic checks =========
    if len(return_value_list) != len(elem_indices_list_list) or len(return_value_list) != len(ax_list):
        raise ValueError("return_value_list, elem_indices_list_list, and ax_list must have equal lengths.")

    if len(ax_list) == 0:
        z = np.zeros((0, 0), dtype=bool)
        if return_only_union:
            return {
                "line_marker": z,
                "line": z,
                "scatter": z,
                "polar_line_marker": z,
                "polar_line": z,
                "polar_scatter": z,
            }
        return {
            "line_marker": [],
            "line": [],
            "scatter": [],
            "polar_line_marker": [],
            "polar_line": [],
            "polar_scatter": [],
        }

    # ========= helpers =========
    def _is_polar_ax(ax):
        return (getattr(ax, "name", None) == "polar") or ax.__class__.__name__.lower().startswith("polar")

    def _lines_from_plot_ret(ret):
        # Axes.plot returns list of Line2D
        if ret is None:
            return []
        if isinstance(ret, (list, tuple)):
            return [x for x in ret if x is not None]
        return [ret]

    def _save_state_line(ln):
        st = {"line": ln, "vis": True, "ls": None, "lw": None, "alpha": None, "markevery": None}
        try:
            st["vis"] = ln.get_visible()
        except Exception:
            pass
        try:
            st["ls"] = ln.get_linestyle()
        except Exception:
            pass
        try:
            st["lw"] = ln.get_linewidth()
        except Exception:
            pass
        try:
            st["alpha"] = ln.get_alpha()
        except Exception:
            pass
        try:
            st["markevery"] = ln.get_markevery()
        except Exception:
            pass
        return st

    def _restore_state_line(st):
        ln = st["line"]
        try:
            ln.set_visible(st["vis"])
        except Exception:
            pass
        try:
            ln.set_linestyle(st["ls"])
        except Exception:
            pass
        try:
            if st.get("lw", None) is not None:
                ln.set_linewidth(st["lw"])
        except Exception:
            pass
        try:
            # Restore alpha even when its original value is None.
            ln.set_alpha(st.get("alpha", None))
        except Exception:
            pass
        try:
            ln.set_markevery(st["markevery"])
        except Exception:
            pass

    def _set_line_invisible(ln):
        try:
            ln.set_visible(False)
        except Exception:
            pass

    def _resolve_markevery_indices(markevery, N):
        """Resolve discrete marker sampling: None, integer, integer tuple, slice, or index array. Reject distance-based float sampling."""
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
            return None

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

    def _marker_is_drawn(ln, visible_override=True):
        if ln is None:
            return False
        if not bool(visible_override):
            return False
        try:
            marker = ln.get_marker()
            if marker in (None, "", " ", "None"):
                return False
            ms = float(ln.get_markersize() or 0.0)
            if ms <= 0:
                return False
            a = ln.get_alpha()
            if a == 0:
                return False
            return True
        except Exception:
            return False

    def _line_is_drawn(ln, st):
        """Check visibility, line style, positive linewidth, and nonzero alpha; None alpha is valid."""
        if ln is None:
            return False
        if not bool(st.get("vis", True)):
            return False
        ls = st.get("ls", None)
        if ls is None:
            return False
        if isinstance(ls, str) and ls.strip().lower() == "none":
            return False
        if isinstance(ls, str) and ls.strip() == "":
            return False
        try:
            lw = st.get("lw", None)
            lw = float(lw) if lw is not None else float(ln.get_linewidth() or 0.0)
        except Exception:
            lw = 0.0
        if lw <= 0:
            return False
        a = st.get("alpha", None)
        if a is not None:
            try:
                if float(a) == 0.0:
                    return False
            except Exception:
                pass
        return True

    def _apply_marker_only(ln, keep_is):
        if ln is None:
            return
        try:
            ln.set_visible(True)
        except Exception:
            pass
        try:
            ln.set_linestyle("None")
        except Exception:
            try:
                ln.set_linestyle(None)
            except Exception:
                pass
        try:
            ln.set_markevery(list(keep_is))
        except Exception:
            pass

    def _apply_line_full(ln, st):
        if ln is None:
            return
        try:
            ln.set_visible(bool(st.get("vis", True)))
        except Exception:
            pass
        try:
            ln.set_linestyle(st.get("ls", None))
        except Exception:
            pass
        try:
            if st.get("lw", None) is not None:
                ln.set_linewidth(st["lw"])
        except Exception:
            pass
        try:
            ln.set_alpha(st.get("alpha", None))
        except Exception:
            pass
        try:
            ln.set_markevery(st.get("markevery", None))
        except Exception:
            pass

    # ========= figure consistency =========
    fig = ax_list[0].figure
    for ax in ax_list:
        if ax.figure is not fig:
            raise ValueError("extract_plot requires all inputs to belong to the same figure.")
    fig.canvas.draw()

    restore_legends = _freeze_legends(fig)
    try:
        # ---- build flattened (line, idxs, is_polar) list ----
        inv = []
        for rv, idxs, ax in zip(return_value_list, elem_indices_list_list, ax_list):
            lines = _lines_from_plot_ret(rv)
            pol = _is_polar_ax(ax)
            for ln in lines:
                inv.append((ln, idxs, pol))

        # uniq lines for state save/restore
        uniq_lines = []
        seen = set()
        for ln, _, _ in inv:
            if ln is None:
                continue
            k = id(ln)
            if k not in seen:
                seen.add(k)
                uniq_lines.append(ln)

        states = {id(ln): _save_state_line(ln) for ln in uniq_lines}

        def _hide_all_lines():
            for ln in uniq_lines:
                _set_line_invisible(ln)

        def _restore_all_lines():
            for ln in uniq_lines:
                _restore_state_line(states[id(ln)])

        # =========================================================
        # UNION MODE
        # =========================================================
        if return_only_union:
            # 1) base: hide all lines
            _hide_all_lines()
            img_base = fig_rgba_u8(fig)

            # Precompute marker indices by semantic category and coordinate system.
            want_marker_cart_online = {}   # id(ln) -> set(i)   (line_drawn=True)
            want_marker_polar_online = {}
            want_scatter_cart = {}         # id(ln) -> set(i)   (line_drawn=False)
            want_scatter_polar = {}
            want_line_cart = set()         # id(ln)            (line_drawn=True)
            want_line_polar = set()

            for ln, idxs, is_polar in inv:
                if ln is None:
                    continue
                st = states.get(id(ln))
                if st is None:
                    continue

                line_drawn = _line_is_drawn(ln, st)

                # N points
                try:
                    x = np.asarray(ln.get_xdata())
                    y = np.asarray(ln.get_ydata())
                except Exception:
                    continue
                N = min(len(x), len(y))
                if N <= 0:
                    continue

                # candidate indices
                if idxs is None:
                    cand_is = list(range(N))
                else:
                    cand_is = [int(i) for i in idxs if i is not None and 0 <= int(i) < N]
                if not cand_is:
                    continue

                # Include a full-line mask only when a line is drawn.
                if line_drawn:
                    if is_polar:
                        want_line_polar.add(id(ln))
                    else:
                        want_line_cart.add(id(ln))

                # Classify visible markers as line markers or standalone scatter points.
                if not _marker_is_drawn(ln, visible_override=st.get("vis", True)):
                    continue

                vis_idx = _resolve_markevery_indices(st.get("markevery", None), N)
                if vis_idx is not None:
                    vis_set = set(vis_idx)
                    cand_is = [i for i in cand_is if i in vis_set]
                    if not cand_is:
                        continue

                # Classify markers according to whether their parent artist draws a line.
                if line_drawn:
                    d = want_marker_polar_online if is_polar else want_marker_cart_online
                else:
                    d = want_scatter_polar if is_polar else want_scatter_cart

                s = d.get(id(ln))
                if s is None:
                    d[id(ln)] = set(cand_is)
                else:
                    s.update(cand_is)

            # Cartesian line markers.
            _restore_all_lines()
            _hide_all_lines()
            for ln in uniq_lines:
                ii = id(ln)
                if ii in want_marker_cart_online:
                    _apply_marker_only(ln, sorted(want_marker_cart_online[ii]))
            img_on = fig_rgba_u8(fig)
            union_points_cart = diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)

            # Polar line markers.
            _restore_all_lines()
            _hide_all_lines()
            for ln in uniq_lines:
                ii = id(ln)
                if ii in want_marker_polar_online:
                    _apply_marker_only(ln, sorted(want_marker_polar_online[ii]))
            img_on = fig_rgba_u8(fig)
            union_points_polar = diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)

            # Standalone Cartesian markers.
            _restore_all_lines()
            _hide_all_lines()
            for ln in uniq_lines:
                ii = id(ln)
                if ii in want_scatter_cart:
                    _apply_marker_only(ln, sorted(want_scatter_cart[ii]))
            img_on = fig_rgba_u8(fig)
            union_scatter_cart = diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)

            # Standalone polar markers.
            _restore_all_lines()
            _hide_all_lines()
            for ln in uniq_lines:
                ii = id(ln)
                if ii in want_scatter_polar:
                    _apply_marker_only(ln, sorted(want_scatter_polar[ii]))
            img_on = fig_rgba_u8(fig)
            union_scatter_polar = diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)

            # Cartesian series that actually draw a line.
            _restore_all_lines()
            _hide_all_lines()
            for ln in uniq_lines:
                if id(ln) in want_line_cart:
                    _apply_line_full(ln, states[id(ln)])
            img_on = fig_rgba_u8(fig)
            union_lines_cart = diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)

            # Polar series that actually draw a line.
            _restore_all_lines()
            _hide_all_lines()
            for ln in uniq_lines:
                if id(ln) in want_line_polar:
                    _apply_line_full(ln, states[id(ln)])
            img_on = fig_rgba_u8(fig)
            union_lines_polar = diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)

            # restore all
            _restore_all_lines()

            return {
                "line_marker": union_points_cart,
                "line": union_lines_cart,
                "scatter": union_scatter_cart|union_scatter_polar,
                "polar_line_marker": union_points_polar,
                "polar_line": union_lines_polar,
            }

        # =========================================================
        # PER-INSTANCE MODE
        # =========================================================
        res = {
            "line_marker": [],
            "line": [],
            "scatter": [],
            "polar_line_marker": [],
            "polar_line": [],
        }

        img_full = fig_rgba_u8(fig)

        for ln, idxs, is_polar in inv:
            if ln is None:
                continue
            st = _save_state_line(ln)
            line_drawn = _line_is_drawn(ln, st)

            # Whole-line mask: generate only when a line is actually drawn.
            _set_line_invisible(ln)
            img_hide = fig_rgba_u8(fig)
            if line_drawn:
                m_line = diff_to_mask(img_hide, img_full, diff_thresh=diff_thresh)
                if is_polar:
                    res["polar_line"].append(m_line)
                else:
                    res["line"].append(m_line)

            img_base = img_hide

            # points
            try:
                x = np.asarray(ln.get_xdata())
                y = np.asarray(ln.get_ydata())
            except Exception:
                _restore_state_line(st)
                continue
            N = min(len(x), len(y))
            if N <= 0:
                _restore_state_line(st)
                continue

            if idxs is None:
                cand_is = list(range(N))
            else:
                cand_is = [int(i) for i in idxs if i is not None and 0 <= int(i) < N]
            if not cand_is:
                _restore_state_line(st)
                continue

            marker_ok = _marker_is_drawn(ln, visible_override=st.get("vis", True))
            marker_is = []
            if marker_ok:
                vis_idx = _resolve_markevery_indices(st.get("markevery", None), N)
                if vis_idx is None:
                    marker_is = cand_is[:]  # float markevery -> no restriction
                else:
                    vis_set = set(vis_idx)
                    marker_is = [i for i in cand_is if i in vis_set]

            for i in marker_is:
                _restore_state_line(st)
                _set_line_invisible(ln)
                _apply_marker_only(ln, [i])
                img_on = fig_rgba_u8(fig)
                m_pt = diff_to_mask(img_on, img_base, diff_thresh=diff_thresh)

                # Use scatter labels when no line is drawn, and line-marker labels otherwise.
                if line_drawn:
                    if is_polar:
                        res["polar_line_marker"].append(m_pt)
                    else:
                        res["line_marker"].append(m_pt)
                else:
                    if is_polar:
                        res["scatter"].append(m_pt)
                    else:
                        res["scatter"].append(m_pt)

            _restore_state_line(st)

        return res

    finally:
        restore_legends()
