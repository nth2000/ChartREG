"""Keep legend positions fixed while rendering artist differences."""


def _freeze_legends(fig):
    """Pin original legends in display space and return an idempotent restore callable.

    Existing legend artists preserve marker shapes, font styling, and spacing.
    Matplotlib legend handles are already separate from their source artists.
    """
    from matplotlib.legend import Legend

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    states = []
    for legend in fig.findobj(Legend):
        if not legend.get_visible():
            continue
        # Legend uses its parent's axes coordinates, or figure coordinates.
        parent = legend.axes if legend.axes is not None else fig
        transform = parent.transAxes if legend.axes is not None else fig.transFigure
        box = legend.get_window_extent(renderer)
        point = transform.inverted().transform((box.x0, box.y0))
        states.append((legend, legend._loc, legend._bbox_to_anchor, legend.stale))
        legend.set_bbox_to_anchor(None)
        legend._loc = tuple(point)

    restored = False

    def restore():
        nonlocal restored
        if restored:
            return
        for legend, loc, anchor, stale in states:
            legend._bbox_to_anchor = anchor
            legend._loc = loc
            legend.stale = stale
        restored = True

    return restore
