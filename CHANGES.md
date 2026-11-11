# Changes in This Package

- Translated source comments, docstrings, messages, and semantic labels into English; removed stale notes and compiled bytecode.
- Included the supplied rendering and legend utilities at the import locations expected by the pipeline.
- Replaced fragile return tracing with temporary API wrappers so decorated scatter calls, multiline calls, pyplot, helper functions, and repeated calls are captured.
- Preserved each invocation's actual axes and errorbar sampling; added standard squarify support without a package patch.
- Fixed full-extraction union mode and kept selected-extraction union output consistent with instance output.
- Kept complete boxplots, median lines, and box patches under distinct labels.
- Added marker-free errorbar extraction and Boolean marker-sampling support.
- Fixed legend movement without changing its rendered appearance; restored scatter coordinates even when extraction raises an exception.
- Added a CLI, dependencies, working examples, binary mask export, overlays, inclusive bounding boxes, and regression tests.
- Updated image comparison to preserve input resolution and save the files it reports.

Validation: 10 unittest test methods, including parameterized coverage of 17 chart configurations, polar variants, selection, source capture, union equivalence, legend stability, cleanup on failure, and exported mask metadata. The benchmark data is not included in this package.
