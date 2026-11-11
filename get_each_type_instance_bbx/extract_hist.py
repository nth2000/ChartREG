import numpy as np
from utils_ import fig_rgba_u8, diff_to_mask
from freeze_legend import _freeze_legends


def extract_hist(
    return_value_list,
    elem_indices_list_list,
    ax_list,
    return_union=True,
    diff_thresh=0,
):
    """Extract histogram bar masks from hist tuples or patch containers. None selects all bars; index lists apply to each dataset container. return_union selects union or per-bar output."""

    # ========= basic checks =========
    if len(return_value_list) != len(elem_indices_list_list) or len(return_value_list) != len(ax_list):
        raise ValueError("return_value_list, elem_indices_list_list, and ax_list must have equal lengths.")

    if len(ax_list) == 0:
        z = np.zeros((0, 0), dtype=bool)
        return {"histogram": z} if return_union else {"histogram": []}

    # ========= helpers =========
    def _extract_patches_from_hist_ret(ret):
        """
        Normalize a single hist return to a list of "containers",
        where each container is a list of patch-like artists (typically Rectangles).

        hist() returns (n, bins, patches). patches can be:
          - BarContainer
          - list[BarContainer]
          - list[Polygon] (step hist)  ... best-effort supported
       
        """
        patches = ret
        if isinstance(ret, tuple) and len(ret) >= 3:
            patches = ret[2]

        containers = []

        # BarContainer-like: has .patches
        if hasattr(patches, "patches"):
            try:
                containers.append(list(patches.patches))
                return containers
            except Exception:
                pass

        # list/tuple of BarContainer or list of artists
        if isinstance(patches, (list, tuple)):
            if len(patches) == 0:
                return []
            # list of BarContainers
            if all(hasattr(x, "patches") for x in patches):
                for bc in patches:
                    try:
                        containers.append(list(bc.patches))
                    except Exception:
                        pass
                return containers
            # list of artists (Rectangles/Polygons/etc.) -> treat as a single container
            containers.append([x for x in patches if x is not None])
            return containers

        # single artist fallback
        if patches is not None:
            containers.append([patches])
        return containers

    def _collect_targets_for_one_ret(ret, idxs):
        """
        For one hist-return, return list of (patch_artist) selected by idxs,
        applying idxs to EACH container from that return.
        """
        containers = _extract_patches_from_hist_ret(ret)
        targets = []

        for cont in containers:
            m = len(cont)
            if m == 0:
                continue

            if idxs is None:
                sel = list(range(m))
            else:
                sel = [int(i) for i in idxs if i is not None and 0 <= int(i) < m]

            # An empty index list selects no bars from this container.
            for i in sel:
                targets.append(cont[i])

        return targets

    # ========= figure consistency =========
    fig = ax_list[0].figure
    for ax in ax_list:
        if ax.figure is not fig:
            raise ValueError("extract_hist requires all inputs to belong to the same figure.")

    fig.canvas.draw()

    restore_legends = _freeze_legends(fig)
    try:
        # Snapshot the full figure after freezing legends.
        img_full = fig_rgba_u8(fig)

        # ===== collect all target patch artists =====
        target_patches = []
        for ret, idxs in zip(return_value_list, elem_indices_list_list):
            target_patches.extend(_collect_targets_for_one_ret(ret, idxs))

        # Return an all-False image-sized mask when no bars are selected.
        if len(target_patches) == 0:
            z = np.zeros(img_full.shape[:2], dtype=bool)
            return {"histogram": z} if return_union else {"histogram": []}

        # =========================================================
        # UNION MODE
        # =========================================================
        if return_union:
            saved = []
            for p in target_patches:
                try:
                    saved.append((p, p.get_visible()))
                except Exception:
                    saved.append((p, True))
                try:
                    p.set_visible(False)
                except Exception:
                    pass

            img_hide = fig_rgba_u8(fig)

            # restore
            for p, v in saved:
                try:
                    p.set_visible(v)
                except Exception:
                    pass

            union_mask = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)
            return {"histogram": union_mask}

        # =========================================================
        # PER-INSTANCE MODE: each bar one mask
        # =========================================================
        res = {"histogram": []}

        for p in target_patches:
            try:
                vis0 = p.get_visible()
            except Exception:
                vis0 = True

            try:
                p.set_visible(False)
            except Exception:
                # Skip artists that do not support visibility changes.
                continue

            img_hide = fig_rgba_u8(fig)

            try:
                p.set_visible(vis0)
            except Exception:
                pass

            m = diff_to_mask(img_full, img_hide, diff_thresh=diff_thresh)
            res["histogram"].append(m)

        return res

    finally:
        restore_legends()
