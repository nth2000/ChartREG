"""Regression tests for mask counts, unions, selection, capture, and exports."""
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from PIL import Image

from freeze_legend import _freeze_legends
from get_seg_mask_bbox_clean import get_seg_mask_bbox, save_masks
from utils_ import fig_rgba_u8, mask_bbox


PREFIX = "import matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(3, 2), dpi=70)\n"


class PipelineTests(unittest.TestCase):
    def synthesize(self, source, selection=None, family=None, union=False):
        return get_seg_mask_bbox(PREFIX + source, selection, family, union)

    def test_supported_chart_families_and_unions(self):
        cases = [
            ('ax.bar([0,1,2],[1,2,3])', {'vertical_bar': 3}),
            ('ax.barh([0,1,2],[1,2,3])', {'horizontal_bar': 3}),
            ('ax.bar([0,1,2],[1,2,3],yerr=.2,capsize=3)',
             {'vertical_bar': 3, 'vertical_bar_with_errorbar': 3, 'errorbar': 3}),
            ('ax.scatter([0,1,2],[1,2,3])', {'scatter': 3}),
            ('ax.plot([0,1,2],[1,2,3],marker="o")', {'line': 1, 'line_marker': 3}),
            ('ax.plot([0,1,2],[1,2,3],marker="o",linestyle="none")', {'scatter': 3}),
            ('ax.plot([0,1,2],[1,2,3],marker="o",markevery=[True,False,True])',
             {'line': 1, 'line_marker': 2}),
            ('ax.errorbar([0,1,2],[1,2,3],yerr=.2,fmt="o-",capsize=3)',
             {'errorbar_line': 1, 'line_marker': 3, 'errorbar': 3, 'point_with_errorbar': 3}),
            ('ax.errorbar([0,1,2],[1,2,3],yerr=.2,fmt="none",capsize=3)', {'errorbar': 3}),
            ('ax.errorbar([0,1,2],[1,2,3],yerr=.2,fmt="o",errorevery=(1,2),capsize=3)',
             {'scatter': 3, 'errorbar': 1, 'point_with_errorbar': 1}),
            ('ax.hist([1,2,2,3,3,3],bins=3)', {'histogram': 3}),
            ('ax.boxplot([[1,2,3],[2,4,6]],patch_artist=True)',
             {'boxplot': 2, 'box_patch': 2, 'median_line': 2}),
            ('ax.fill([0,1,2],[0,2,0])', {'filled_polygon': 1}),
            ('ax.fill_between([0,1,2],[1,2,1],[0,0,0])', {'fill_between': 1}),
            ('ax.stackplot([0,1,2],[[1,2,1],[2,1,2]])', {'stacked_area': 2}),
            ('ax.pie([1,2,3])', {'pie_wedge': 3}),
            ('import squarify\nsquarify.plot(sizes=[1,2,3],color=["red","green","blue"],ax=ax)', {'treemap': 3}),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                masks, image = self.synthesize(source)
                self.assertEqual({k: len(v) for k, v in masks.items() if v}, expected)
                union, same_image = self.synthesize(source, union=True)
                np.testing.assert_array_equal(image, same_image)
                for key, instances in masks.items():
                    merged = np.zeros(image.shape[:2], dtype=bool)
                    for mask in instances:
                        self.assertEqual(mask.shape, image.shape[:2])
                        self.assertEqual(mask.dtype, np.dtype(bool))
                        self.assertTrue(mask.any())
                        x0, y0, x1, y1 = mask_bbox(mask)
                        self.assertGreaterEqual(x0, 0)
                        self.assertLess(x1, image.shape[1])
                        self.assertGreaterEqual(y0, 0)
                        self.assertLess(y1, image.shape[0])
                        merged |= mask
                    if instances:
                        np.testing.assert_array_equal(union[key], merged)

    def test_multiline_pyplot_helper_and_loop_capture(self):
        cases = [
            ('ax.scatter (\n[0,1],\n[1,2]\n)', 2),
            ('plt.scatter([0,1],[1,2])', 2),
            ('def draw():\n    ax.scatter([0,1],[1,2])\ndraw()', 2),
            ('for i in range(2):\n    ax.scatter([i],[i])', 2),
            ('fig, axes = plt.subplots(1,2)\nfor ax in axes:\n    ax.scatter([0,1],[1,2])', 4),
        ]
        for source, count in cases:
            with self.subTest(source=source):
                masks, _ = self.synthesize(source)
                self.assertEqual(len(masks['scatter']), count)

    def test_polar_elements(self):
        for source, expected in [
            ('ax = fig.add_subplot(111,projection="polar")\nax.bar([0,1,2],[1,2,3])', {'polar_bar': 3}),
            ('ax = fig.add_subplot(111,projection="polar")\nax.plot([0,1,2],[1,2,3],marker="o")',
             {'polar_line': 1, 'polar_line_marker': 3}),
        ]:
            masks, _ = self.synthesize(source)
            self.assertEqual({k: len(v) for k,v in masks.items() if v}, expected)

    def test_multiline_selection_and_invocation_index(self):
        source = 'for i in range(2):\n    ax.scatter(\n        [0,1,2],\n        [1+i,2+i,3+i]\n    )  #0'
        selection = {'results': [{'line': '#0', 'invocation_count': 1, 'element_indices': [2]}]}
        masks, image = self.synthesize(source, selection, 'scatter')
        self.assertEqual(len(masks['scatter']), 1)
        self.assertTrue(masks['scatter'][0].any())
        unions, _ = self.synthesize(source, selection, 'scatter', union=True)
        np.testing.assert_array_equal(masks['scatter'][0], unions['scatter'])
        full, _ = self.synthesize(source)
        np.testing.assert_array_equal(full['scatter'][5], masks['scatter'][0])

    def test_selected_errorbar_without_markers(self):
        source = 'ax.errorbar([0,1,2,3],[1,2,3,4],yerr=.2,fmt="none",errorevery=(1,2),capsize=3)  #1'
        selection = {'results': [{'line': '#1', 'invocation_count': 0, 'element_indices': [3]}]}
        masks, _ = self.synthesize(source, selection, 'point_with_errorbar')
        self.assertEqual(len(masks['errorbar']), 1)
        self.assertTrue(masks['errorbar'][0].any())

    def test_legends_are_pixel_identical_and_restored(self):
        fig, ax = plt.subplots(figsize=(3, 2), dpi=70)
        ax.scatter([0,1,2], [1,2,3], marker='s', label='data')
        legend = ax.legend(loc='best', title='Series')
        before = fig_rgba_u8(fig)
        restore = _freeze_legends(fig)
        np.testing.assert_array_equal(before, fig_rgba_u8(fig))
        restore()
        restore()
        np.testing.assert_array_equal(before, fig_rgba_u8(fig))
        bbox = legend.get_window_extent().extents.astype(int)
        source = 'ax.scatter([0,1,2],[1,2,3],marker="s",label="data")\nax.legend(loc="best",title="Series")'
        masks, image = self.synthesize(source)
        union = np.logical_or.reduce(masks['scatter'])
        x0,y0,x1,y1=bbox
        self.assertFalse(union[image.shape[0]-y1:image.shape[0]-y0,x0:x1].any())

    def test_wrappers_and_rcparams_restored_after_exception(self):
        scatter = Axes.scatter
        font = plt.rcParams['font.size']
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            self.synthesize('plt.rcParams["font.size"]=31\nraise RuntimeError("test failure")')
        self.assertIs(Axes.scatter, scatter)
        self.assertEqual(plt.rcParams['font.size'], font)
        self.assertFalse(plt.get_fignums())

    def test_multiple_figures_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one figure'):
            self.synthesize('ax.scatter([0],[1])\nfig2,ax2=plt.subplots()\nax2.scatter([0],[1])')
        with self.assertRaisesRegex(ValueError, 'float markevery'):
            self.synthesize('ax.plot([0,1,2],[1,2,3],marker="o",markevery=0.2)')

    def test_binary_masks_boxes_and_overlays_exported(self):
        masks, image = self.synthesize('ax.bar([0,1,2],[1,2,3])')
        with tempfile.TemporaryDirectory() as folder:
            save_masks(masks, image, folder, False)
            out = Path(folder)
            metadata = json.loads((out/'annotations.json').read_text())
            self.assertEqual(len(metadata['instances']), 3)
            for record in metadata['instances']:
                with Image.open(out/record['mask']) as image_file:
                    raw = np.asarray(image_file)
                self.assertEqual(set(np.unique(raw)), {0,255})
                self.assertEqual(record['area'], int(np.count_nonzero(raw)))
                self.assertEqual(record['bbox'], list(mask_bbox(raw.astype(bool))))
                self.assertTrue((out/record['overlay']).is_file())

    def test_no_chinese_text_or_bytecode_in_source(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in {'.py','.md','.json','.txt'}:
                self.assertFalse(any('\u3400' <= char <= '\u9fff' for char in path.read_text()), str(path))


if __name__ == '__main__':
    unittest.main()
