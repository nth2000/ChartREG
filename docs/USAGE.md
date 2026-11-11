# Mask Synthesis Usage

## Python API

Run from the repository root:

```python
from pathlib import Path
from get_seg_mask_bbox_clean import get_seg_mask_bbox, save_masks

code = Path("examples/chart.py").read_text(encoding="utf-8")
masks, image = get_seg_mask_bbox(
    code_text=code,
    json_dict=None,
    ref_type=None,
    return_only_union=False,
    diff_thresh=0,
)
save_masks(masks, image, "extracted_instances", return_only_union=False)
```

`image` is a copied uint8 RGBA array of shape `(H, W, 4)`. In instance mode, `masks` maps English labels to lists of Boolean `(H, W)` arrays. Union mode maps labels to Boolean masks, computed as the OR of their instance masks. Coordinates use the original canvas: x increases rightward and y increases downward.

`diff_thresh` is the threshold on the sum of absolute RGBA channel differences. A pixel is included only when its difference is greater than the threshold. Masks describe contributions under the current rendering, clipping, transparency, and occlusion; fully invisible instances can produce empty masks.

## Selected Extraction

Mark a plotting call with a trailing integer comment, such as `#1`. Multiline calls can place the marker on their first or last line. Markers must be unique.

```json
{
  "results": [
    {"line": "#1", "invocation_count": 0, "element_indices": [0, 2]}
  ]
}
```

`invocation_count` is the zero-based execution count of the marked call, including calls inside loops. `element_indices` is a zero-based index list; `null` selects all elements in that invocation. `invocation_count: null` skips the entry. A reference type chooses an extractor family, which can return several related output labels.

| API / extractor family | Accepted reference types | Output labels |
| --- | --- | --- |
| `scatter` | `scatter` | `scatter` |
| `bar`, `barh` | `vertical_bar`, `horizontal_bar`, `polar_bar`, `vertical_bar_with_errorbar`, `horizontal_bar_with_errorbar`, `errorbar` | Same bar and errorbar labels |
| `errorbar` | `point_with_errorbar` | `point_with_errorbar`, `line_marker`, `scatter`, `errorbar_line`, `errorbar` |
| `plot` | `line`, `line_marker`, `polar_line`, `polar_line_marker` | These labels plus `scatter` and `polar_scatter` for marker-only series |
| `hist` | `histogram` | `histogram` |
| `boxplot`, `bxp` | `boxplot`, `box_patch`, `median_line` | `boxplot`, `box_patch`, `median_line` |
| `fill` | `filled_polygon` | `filled_polygon` |
| `fill_between` | `fill_between` | `fill_between` |
| `stackplot` | `stacked_area` | `stacked_area` |
| `pie` | `pie_wedge` | `pie_wedge` |
| `squarify.plot` | `treemap` | `treemap` |

Full boxplot instances include their box, whiskers, caps, median, and any drawn means; fliers are excluded. `box_patch` and `median_line` provide finer masks. Line element indices select markers; a complete line-series mask remains a single instance. Errorbar point indices refer to the original input data, before `errorevery` sampling.

## Files and Constraints

- Binary PNGs contain 0 for background and 255 for foreground. Overlays use a separate `_overlay.png` suffix.
- `annotations.json` stores image dimensions, labels, areas, filenames, and inclusive `[xmin, ymin, xmax, ymax]` boxes. Empty masks have a null box.
- One figure per synthesis call is supported, including multiple subplots. Keep the figure open and avoid `plt.show()` in input scripts. Use distinct source lines for different plotting APIs.
- The API temporarily wraps plotting methods, resets plotting defaults, and closes figures on exit. Run synthesis in a dedicated process, not concurrently in multiple threads.
- Standard `squarify.plot` works without modifying the installed package. All source comments, messages, and output labels are English; older annotation files with non-English type names must be updated.
- Discrete `markevery` and `errorevery` forms are covered by tests. Distance-based floating-point `markevery` is not supported.

`get_diff.py` is an optional Pillow-based visualization utility. It compares equal-sized images without resizing them:

```bash
python get_diff.py first.png second.png --output difference.png
```
