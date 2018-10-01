import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    'DegreeLocator',
]


class DegreeLocator(plt.Locator):
    """Locator for when the axis is angular (degree) scale.

    Examples
    --------
    >>> ax = plt.gca()
    >>> ax.xaxis.set_major_locator(DegreeLocator())
    >>> t = np.linspace(0, 360, 100)
    >>> _ = plt.plot(t, np.sin(np.deg2rad(t)))
    >>> plt.grid()
    >>> plt.show()
    """

    def __init__(self, bases=(360, 180, 90, 45, 30, 15), min_n_ticks=4):
        bases = np.array(bases)
        ranges = bases * min_n_ticks
        self._locators = [
            (
                plt.AutoLocator(),
                lambda ptp: not min(ranges) < ptp < max(ranges)
            )
        ]
        self._locators += [
            (
                plt.MultipleLocator(b),
                lambda ptp, r=r: ptp > r,
            )
            for b, r in zip(bases, ranges)
        ]
        self._curloc, _ = self._locators[0]

    def _update_curloc(self, vmin, vmax):
        for loc, cond in self._locators:
            if cond(abs(vmax - vmin)):
                self._curloc = loc
                return

    def set_axis(self, axis):
        for loc, _ in self._locators:
            loc.set_axis(axis)

    def create_dummy_axis(self, **kwargs):
        for loc, cond in self._locators:
            loc.create_dummy_axis(**kwargs)

    def set_view_interval(self, vmin, vmax):
        self._update_curloc(vmin, vmax)
        for loc, cond in self._locators:
            loc.set_view_interval(vmin, vmax)

    def set_data_interval(self, vmin, vmax):
        self._update_curloc(vmin, vmax)
        for loc, cond in self._locators:
            loc.set_data_interval(vmin, vmax)

    def set_bounds(self, vmin, vmax):
        self._update_curloc(vmin, vmax)
        for loc, cond in self._locators:
            loc.set_bounds(vmin, vmax)

    def tick_values(self, vmin, vmax):
        self._update_curloc(vmin, vmax)
        return self._curloc.tick_values(vmin, vmax)

    def set_params(self, **kwargs):
        for loc, _ in self._locators:
            loc.set_params(kwargs)

    def __call__(self):
        self._update_curloc(*self._curloc.axis.get_view_interval())
        return self._curloc()

    def raise_if_exceeds(self, locs):
        return self._curloc.raise_if_exceeds(locs)

    def view_limits(self, vmin, vmax):
        self._update_curloc(vmin, vmax)
        return self._curloc.view_limits(vmin, vmax)

    def autoscale(self):
        return self._curloc.autoscale()

    def pan(self, numsteps):
        return self._curloc.pan(numsteps)

    def zoom(self, direction):
        return self._curloc.zoom(direction)

    def refresh(self):
        return self._curloc.refresh()
