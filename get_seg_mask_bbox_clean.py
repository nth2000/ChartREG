# -*- coding: utf-8 -*-
import os
import re
import ast
import functools
import inspect
import sys
from contextlib import contextmanager
import numpy as np
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

from utils_ import *
from get_each_type_instance_bbx.extract_scatter import *
from get_each_type_instance_bbx.extract_errorpoint_with_errorbar_new import *
from get_each_type_instance_bbx.extract_bar_new_new import *
from get_each_type_instance_bbx.extract_plot_new import *
from get_each_type_instance_bbx.extract_hist import *
from get_each_type_instance_bbx.extract_box import *
from get_each_type_instance_bbx.extract_fill import *
from get_each_type_instance_bbx.extract_fill_between import *
from get_each_type_instance_bbx.extract_stackplot import *
from get_each_type_instance_bbx.extract_treemap import *
from get_each_type_instance_bbx.extract_pie import *

# Plotting APIs eligible for capture.
allowed_target_api_name = {
    'stackplot', 'fill_between', 'hist', 'bar', 'barh', 'errorbar',
    'plot', 'fill', 'boxplot', 'pie', 'scatter', 'bxp'
}
def _is_treemap_squarify_plot_return(ret):
    """Recognize a treemap return containing an Axes and rectangle container."""
    if not isinstance(ret, tuple) or len(ret) != 2:
        return False

    ax, bar_ret = ret
    if ax is None or (not hasattr(ax, "bar")):
        return False

    # Accept a BarContainer or a sequence of rectangle patches.
    if hasattr(bar_ret, "patches"):
        return True
    if isinstance(bar_ret, (list, tuple)) and len(bar_ret) > 0:
        p0 = bar_ret[0]
        if hasattr(p0, "get_bbox") or hasattr(p0, "get_xy"):
            return True

    return False


def _mask_or(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a.shape != b.shape:
        raise ValueError(f"Mask shape mismatch: {a.shape} vs {b.shape}")

    # Use bitwise OR for Boolean or integer masks.
    if a.dtype == bool or np.issubdtype(a.dtype, np.integer):
        return np.bitwise_or(a, b)
    return np.logical_or(a, b)

def extract_hash_int_suffix(s: str):
    """Read an integer marker such as #123 at the end of a line."""
    m = re.search(r'#([+-]?\d+)\Z', s)
    return int(m.group(1)) if m else None


@contextmanager
def fresh_matplotlib_state():
    """Close figures, reset plotting defaults, and restore rcParams and interactive mode on exit."""
    old_interactive = plt.isinteractive()
    with matplotlib.rc_context():
        plt.ioff()
        plt.close('all')
        plt.rcdefaults()
        try:
            yield
        finally:
            plt.close('all')
            if old_interactive:
                plt.ion()
            else:
                plt.ioff()


def _parse_allowed_lines(code_text: str, json_dict):
    """Parse plotting calls, including multiline calls and whitespace before parentheses."""
    tree = ast.parse(code_text)
    allowed = {}
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", None))
            if name in allowed_target_api_name:
                spans.append((node.lineno, node.end_lineno, name))
                for line in range(node.lineno, node.end_lineno + 1):
                    allowed[line] = name
    markers = None
    if json_dict is not None:
        markers = {}
        for lineno, line in enumerate(code_text.splitlines(), 1):
            number = extract_hash_int_suffix(line.strip())
            if number is not None:
                matches = [start for start, end, _ in spans if start <= lineno <= end]
                if not matches:
                    raise ValueError(f"No supported plotting call at marker #{number}.")
                key = f"#{number}"
                if key in markers:
                    raise ValueError(f"Duplicate source marker: {key}")
                markers[key] = min(matches)
    return code_text, allowed, markers


def _exec_and_trace_once(code_text: str, json_dict=None):
    """Capture API returns with temporary wrappers, restoring methods even on failure.

    Wrappers support decorated APIs, pyplot calls, loops, and helper functions.
    This function mutates Matplotlib methods temporarily and is not thread-safe.
    """
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer

    code_text, allowed_lines, markers = _parse_allowed_lines(code_text, json_dict)
    filename = "<dynamic_seg_code>"
    compiled = compile(code_text, filename, "exec")
    line2artists = {}
    patches = []
    depth = 0
    call_spans = [node for node in ast.walk(ast.parse(code_text))
                  if isinstance(node, ast.Call)
                  and getattr(node.func, "attr", getattr(node.func, "id", None)) in allowed_target_api_name]

    def install(owner, name):
        original = getattr(owner, name)

        @functools.wraps(original)
        def capture(*args, **kwargs):
            nonlocal depth
            frame = sys._getframe(1)
            while frame is not None and frame.f_code.co_filename != filename:
                frame = frame.f_back
            if depth or frame is None:
                return original(*args, **kwargs)
            lineno = frame.f_lineno
            matching = [node for node in call_spans
                        if node.lineno <= lineno <= node.end_lineno
                        and getattr(node.func, "attr", getattr(node.func, "id", None)) == name]
            source_line = min((node.lineno for node in matching), default=lineno)
            is_treemap = getattr(owner, "__name__", "") == "squarify"
            previous_patches = {id(p) for ax in plt.gcf().axes for p in ax.patches} if is_treemap else set()
            depth += 1
            try:
                result = original(*args, **kwargs)
                public_result = result
            finally:
                depth -= 1
            ax = args[0] if args and isinstance(args[0], Axes) else kwargs.get("ax")
            if is_treemap:
                ax = result[0] if isinstance(result, tuple) else result
                if isinstance(ax, Axes) and not isinstance(result, tuple):
                    rectangles = [p for p in ax.patches if id(p) not in previous_patches]
                    result = (ax, BarContainer(rectangles))
            if ax is None:
                ax = plt.gca()
            try:
                bound = inspect.signature(original).bind_partial(*args, **kwargs)
                params = bound.arguments
            except (ValueError, TypeError):
                params = kwargs
            if name == "errorbar":
                result._mask_point_count = int(np.size(params.get("x", [])))
            error_kw = params.get("error_kw") or {}
            errorevery = params.get("errorevery", error_kw.get("errorevery"))
            info = line2artists.setdefault(source_line, {"ax": ax, "api_name": name, "errorevery": errorevery,
                                                       "axes_by_invocation": {}, "errors_by_invocation": {}})
            if info['api_name'] != name:
                raise ValueError("Use separate source lines for different plotting APIs.")
            index = sum(isinstance(k, int) for k in info)
            info[index] = result
            info["axes_by_invocation"][index] = ax
            info["errors_by_invocation"][index] = errorevery
            return public_result

        setattr(owner, name, capture)
        patches.append((owner, name, original))

    try:
        for name in sorted(allowed_target_api_name):
            if hasattr(Axes, name):
                install(Axes, name)
        try:
            import squarify
        except ImportError:
            squarify = None
        if squarify is not None:
            install(squarify, "plot")
        namespace = {"__builtins__": __builtins__, "__name__": "__main__"}
        exec(compiled, namespace, namespace)
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)

    figures = {ax.figure for info in line2artists.values() for ax in info["axes_by_invocation"].values()}
    if len(figures) > 1:
        raise ValueError("Each synthesis call must render exactly one figure; multiple subplots are supported.")
    fig = next(iter(figures)) if figures else plt.gcf()
    if not plt.fignum_exists(fig.number):
        raise ValueError("Keep the figure open until synthesis completes; remove plt.close() from the input script.")
    plt.figure(fig.number)
    image = fig_rgba_u8(fig)
    if hasattr(fig, "set_layout_engine"):
        fig.set_layout_engine(None)
    return line2artists, image, markers


def get_instance_mask_bbox(results_list, markline2codeline, line2artist_list,
                           ref_type, return_only_union=True, diff_thresh=0):
    """Extract selected instances from captured artists. Return class-to-union-mask or class-to-instance-list mappings. Keep figures open during extraction."""
    if results_list is None:
        raise NotImplementedError("Use get_seg_mask_bbox with json_dict=None for full extraction.")

    if markline2codeline is None:
        raise ValueError("markline2codeline is required when results_list is not None.")

    # Collect selected artists, element indices, axes, and error sampling.
    code_line2selected_invocation_count = {}
    code_line2selected_elem_indices_list = {}
    for item in results_list:
        code_line = markline2codeline[item['line']]
        code_line2selected_invocation_count.setdefault(code_line, []).append(item['invocation_count'])
        code_line2selected_elem_indices_list.setdefault(code_line, []).append(item['element_indices'])

    to_deal_artist_list = []
    to_deal_elem_indices_list = []
    to_deal_ax_list = []
    to_deal_errorevery_list = []

    for code_line, invocation_count_list in code_line2selected_invocation_count.items():
        element_indices_list_list = code_line2selected_elem_indices_list[code_line]

        for invocation_count, element_indices_list in zip(invocation_count_list, element_indices_list_list):
            if invocation_count is None:
                continue
            to_deal_artist_list.append(line2artist_list[code_line][invocation_count])
            to_deal_elem_indices_list.append(element_indices_list)
            info = line2artist_list[code_line]
            to_deal_ax_list.append(info['axes_by_invocation'][invocation_count])
            to_deal_errorevery_list.append(info['errors_by_invocation'][invocation_count])

    # Dispatch to the appropriate extractor.
    if ref_type == 'scatter':
        res = extract_scatter(
            to_deal_artist_list, to_deal_elem_indices_list,
            return_only_union=False, diff_thresh=diff_thresh
        )
    elif ref_type == 'point_with_errorbar':
        res = extract_errorpoint_with_errorbar(
            to_deal_artist_list, to_deal_elem_indices_list,
            to_deal_ax_list, to_deal_errorevery_list,
            return_only_union=False, diff_thresh=diff_thresh
        )
    elif ref_type in ["vertical_bar", "horizontal_bar", "polar_bar", "vertical_bar_with_errorbar", "errorbar", "horizontal_bar_with_errorbar"]:
        res = extract_bar(
            to_deal_artist_list, to_deal_elem_indices_list,
            to_deal_ax_list, to_deal_errorevery_list,
            return_only_union=False, diff_thresh=diff_thresh
        )
    elif ref_type in ["line", "line_marker", "polar_line", "polar_line_marker"]:
        res = extract_plotpoint_with_plotline(
            to_deal_artist_list, to_deal_elem_indices_list, to_deal_ax_list,
            return_only_union=False, diff_thresh=diff_thresh
        )
    elif ref_type in ['histogram']:
        res = extract_hist(
            to_deal_artist_list, to_deal_elem_indices_list, to_deal_ax_list,
            return_union=False, diff_thresh=diff_thresh
        )
    elif ref_type in ["boxplot", "median_line", "box_patch"]:
        res = extract_bxp(
            to_deal_artist_list, to_deal_elem_indices_list, to_deal_ax_list,
            return_only_union=False, diff_thresh=diff_thresh
        )
    elif ref_type in ['filled_polygon']:
        res = extract_fill(
            to_deal_artist_list, to_deal_elem_indices_list, to_deal_ax_list,
            return_only_union=False, diff_thresh=diff_thresh
        )
    elif ref_type in ['fill_between']:
        res = extract_fill_between(
            to_deal_artist_list, to_deal_elem_indices_list, to_deal_ax_list,
            return_only_union=False, diff_thresh=diff_thresh
        )
    elif ref_type in ['stacked_area']:
        res = extract_stackplot(
            to_deal_artist_list, to_deal_elem_indices_list, to_deal_ax_list,
            return_only_union=False, diff_thresh=diff_thresh
        )
    elif ref_type in ['treemap']:
        res = extract_treemap_squarify_plot(
            to_deal_artist_list, to_deal_elem_indices_list, to_deal_ax_list,
            return_only_union=False, diff_thresh=diff_thresh
        )
    elif ref_type in ['pie_wedge']:
        res = extract_pie_wedge(
            to_deal_artist_list, to_deal_elem_indices_list, to_deal_ax_list,
            return_union=False, diff_thresh=diff_thresh
        )
    else:
        raise ValueError(f"Unsupported ref_type: {ref_type}")

    if return_only_union:
        shape = fig_rgba_u8(plt.gcf()).shape[:2]
        return {key: np.logical_or.reduce(masks) if masks else np.zeros(shape, dtype=bool)
                for key, masks in res.items()}
    return res


def get_seg_mask_bbox(code_text, json_dict, ref_type, return_only_union=True, diff_thresh=0):
    """Execute, trace, and extract in a fresh plotting context. Return class masks and the original RGBA image."""
    with fresh_matplotlib_state():
        # PART1/2:exec + trace
        line2artist_list, ori_img, markline2codeline = _exec_and_trace_once(code_text, json_dict)

        # PART3:extract
        if json_dict is not None:
            results_list = json_dict['results']
            reftype2mask = get_instance_mask_bbox(
                results_list, markline2codeline, line2artist_list,
                ref_type, return_only_union=return_only_union, diff_thresh=diff_thresh
            )
            return reftype2mask, ori_img
        else:
            # Extract all supported classes and instances without taking unions.
            # Group artists, axes, and error sampling by extractor; None selects all elements.
            buckets = defaultdict(lambda: {"artists": [], "elem": [], "ax": [], "err": []})

            for code_line, info in line2artist_list.items():
                api_name = info.get("api_name", None)
                ax = info.get("ax", None)
                errorevery = info.get("errorevery", None)

                inv_keys = sorted(k for k in info.keys() if isinstance(k, int))
                for inv in inv_keys:
                    artist_ret = info[inv]
                    ax = info['axes_by_invocation'][inv]
                    errorevery = info['errors_by_invocation'][inv]

                    bucket_name = api_name

                    # Normalize bar and barh into one bucket.
                    if api_name in ("bar", "barh"):
                        bucket_name = "bar"

                    # Normalize boxplot and bxp into one bucket.
                    if api_name in ("boxplot", "bxp"):
                        bucket_name = "bxp"

                    # Distinguish treemap returns from ordinary line plot returns.
                    if api_name == "plot" and _is_treemap_squarify_plot_return(artist_ret):
                        bucket_name = "treemap"

                    # Include pie wedges in full extraction.
                    buckets[bucket_name]["artists"].append(artist_ret)
                    buckets[bucket_name]["elem"].append(None)
                    buckets[bucket_name]["ax"].append(ax)
                    buckets[bucket_name]["err"].append(errorevery)

            # Run each extractor and collect key-to-instance-mask lists.
            key2instance_masks = defaultdict(list)

            def _accumulate(res_dict):
                if not res_dict:
                    return
                for k, v in res_dict.items():
                    if v is None:
                        continue
                    if isinstance(v, list):
                        key2instance_masks[k].extend([m for m in v if m is not None])
                    else:
                        # Accept a single mask if an extractor returns one instead of a list.
                        key2instance_masks[k].append(v)

            # scatter
            if buckets["scatter"]["artists"]:
                _accumulate(extract_scatter(
                    buckets["scatter"]["artists"],
                    buckets["scatter"]["elem"],
                    return_only_union=False,
                    diff_thresh=diff_thresh
                ))

            # Extract errorbar points and their associated errorbars.
            if buckets["errorbar"]["artists"]:
                _accumulate(extract_errorpoint_with_errorbar(
                    buckets["errorbar"]["artists"],
                    buckets["errorbar"]["elem"],
                    buckets["errorbar"]["ax"],
                    buckets["errorbar"]["err"],
                    return_only_union=False,
                    diff_thresh=diff_thresh
                ))

            # bar / barh
            if buckets["bar"]["artists"]:
                _accumulate(extract_bar(
                    buckets["bar"]["artists"],
                    buckets["bar"]["elem"],
                    buckets["bar"]["ax"],
                    buckets["bar"]["err"],
                    return_only_union=False,
                    diff_thresh=diff_thresh
                ))

            # Extract lines and their markers.
            if buckets["plot"]["artists"]:
                _accumulate(extract_plotpoint_with_plotline(
                    buckets["plot"]["artists"],
                    buckets["plot"]["elem"],
                    buckets["plot"]["ax"],
                    return_only_union=False,
                    diff_thresh=diff_thresh
                ))

            # The histogram extractor uses the return_union parameter.
            if buckets["hist"]["artists"]:
                _accumulate(extract_hist(
                    buckets["hist"]["artists"],
                    buckets["hist"]["elem"],
                    buckets["hist"]["ax"],
                    return_union=False,
                    diff_thresh=diff_thresh
                ))

            # box / bxp
            if buckets["bxp"]["artists"]:
                _accumulate(extract_bxp(
                    buckets["bxp"]["artists"],
                    buckets["bxp"]["elem"],
                    buckets["bxp"]["ax"],
                    return_only_union=False,
                    diff_thresh=diff_thresh
                ))

            # Extract individual pie wedges using return_union.
            if buckets["pie"]["artists"]:
                _accumulate(extract_pie_wedge(
                    buckets["pie"]["artists"],
                    buckets["pie"]["elem"],
                    buckets["pie"]["ax"],
                    return_union=False,
                    diff_thresh=diff_thresh
                ))

            # fill
            if buckets["fill"]["artists"]:
                _accumulate(extract_fill(
                    buckets["fill"]["artists"],
                    buckets["fill"]["elem"],
                    buckets["fill"]["ax"],
                    return_only_union=False,
                    diff_thresh=diff_thresh
                ))

            # fill_between
            if buckets["fill_between"]["artists"]:
                _accumulate(extract_fill_between(
                    buckets["fill_between"]["artists"],
                    buckets["fill_between"]["elem"],
                    buckets["fill_between"]["ax"],
                    return_only_union=False,
                    diff_thresh=diff_thresh
                ))

            # stackplot
            if buckets["stackplot"]["artists"]:
                _accumulate(extract_stackplot(
                    buckets["stackplot"]["artists"],
                    buckets["stackplot"]["elem"],
                    buckets["stackplot"]["ax"],
                    return_only_union=False,
                    diff_thresh=diff_thresh
                ))

            # treemap(squarify.plot -> (ax, bar_return))
            if buckets["treemap"]["artists"]:
                _accumulate(extract_treemap_squarify_plot(
                    buckets["treemap"]["artists"],
                    buckets["treemap"]["elem"],
                    buckets["treemap"]["ax"],
                    return_only_union=False,
                    diff_thresh=diff_thresh
                ))

            # Return individual masks without taking their union.
            if return_only_union:
                unions = {}
                for key, masks in key2instance_masks.items():
                    union = np.zeros(ori_img.shape[:2], dtype=bool)
                    for mask in masks:
                        union |= np.asarray(mask, dtype=bool)
                    unions[key] = union
                return unions, ori_img
            return dict(key2instance_masks), ori_img


        # The context manager closes figures on exit.



def save_masks(res_mask, ori_img, out_dir, return_only_union: bool):
    """Save raw binary masks, overlays, and inclusive xyxy pixel boxes."""
    import json
    from pathlib import Path
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ori_img).save(out / "ori.png")
    metadata = {"image": "ori.png", "height": int(ori_img.shape[0]),
                "width": int(ori_img.shape[1]), "bbox_format": "xyxy_inclusive",
                "instances": []}
    for label, value in res_mask.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError(f"Invalid mask label: {label!r}")
        masks = [value] if return_only_union else value
        if masks is None:
            continue
        for index, mask in enumerate(masks):
            if mask is None:
                continue
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != ori_img.shape[:2]:
                raise ValueError(f"Mask shape {mask.shape} does not match image {ori_img.shape[:2]}.")
            stem = f"union_{label}" if return_only_union else f"{label}_{index}"
            Image.fromarray(mask.astype(np.uint8) * 255).save(out / f"{stem}.png")
            Image.fromarray(overlay_mask(ori_img, mask)).save(out / f"{stem}_overlay.png")
            metadata["instances"].append({"label": label, "index": index,
                "mask": f"{stem}.png", "overlay": f"{stem}_overlay.png",
                "bbox": mask_bbox(mask), "area": int(mask.sum())})
    (out / "annotations.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    """Synthesize masks from a trusted Matplotlib script."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__ or "Synthesize chart instance masks from Matplotlib code.")
    parser.add_argument("code", type=Path, help="Input plotting script (executed as Python).")
    parser.add_argument("--output", type=Path, default=Path("extracted_instances"))
    parser.add_argument("--selection", type=Path, help="Optional JSON results with marked source lines.")
    parser.add_argument("--ref-type", help="English extractor family for selected extraction.")
    parser.add_argument("--union", action="store_true", help="Merge instances within each output class.")
    parser.add_argument("--diff-thresh", type=int, default=0)
    args = parser.parse_args()
    if args.selection and not args.ref_type:
        parser.error("--ref-type is required with --selection")
    if args.diff_thresh < 0:
        parser.error("--diff-thresh must be nonnegative")
    matplotlib.use("Agg", force=True)
    selection = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else None
    masks, image = get_seg_mask_bbox(args.code.read_text(encoding="utf-8"), selection,
        args.ref_type, return_only_union=args.union, diff_thresh=args.diff_thresh)
    save_masks(masks, image, args.output, args.union)
    print(f"Saved masks and annotations to {args.output}")


if __name__ == "__main__":
    main()
