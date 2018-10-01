import itertools as it
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def animproc(
        func, fig=None, frames=None, daemon=True, blit=None, interval=30):
    """Create a process for animation plotting.

    For more details around the paratmeters, see the document of
    `matplotlib.animation.FuncAnimation()`.

    Examples
    --------
    >>> import time
    >>> def _fig():
    ...     fig = plt.figure()
    ...     fig.add_subplot(111).plot([], [], '+')
    ...     return fig
    ...
    >>> def _update(frame, fig):
    ...     ax = fig.axes[0]
    ...     line = ax.lines[0]
    ...     line.set_xdata(frame[0])
    ...     line.set_ydata(frame[1])
    ...     ax.relim()
    ...     ax.autoscale()
    ...
    >>> queue = mp.Queue(1)
    >>> process = animproc(_update, fig=_fig, frames=iter(queue.get, None))
    >>> process.start()
    >>> t0 = time.time()
    >>> t, x = [0], [0]
    >>> while process.is_alive() and t[-1] < 5:
    ...     t.append(time.time() - t0)
    ...     x.append(np.sin(5 * t[-1]))
    ...     time.sleep(0.01)
    ...     if queue.empty():
    ...         queue.put([t, x])
    """
    if fig is None:
        fig = plt.figure

    def target():
        fig_ = fig()
        _ = FuncAnimation(
            fig_, func,
            frames=frames,
            fargs=(fig_,),
            blit=blit,
            interval=interval)
        plt.show()

    proc = mp.Process(target=target)
    proc.daemon = daemon
    return proc


def plot_image_array(
        image_array, fig=None, title=None, row_titles=None, col_titles=None,
        **kwargs):
    rows = len(image_array)
    cols = max(map(len, image_array))
    if row_titles is None:
        row_titles = [''] * rows
    elif isinstance(row_titles, dict):
        row_titles = [row_titles.get(i, '') for i in range(rows)]
    else:
        row_titles = list(row_titles)
    if col_titles is None:
        col_titles = [''] * cols
    elif isinstance(row_titles, dict):
        col_titles = [col_titles.get(j, '') for j in range(cols)]

    if fig is None:
        fig = plt.gcf()
    fig.clear()

    for i, j in it.product(range(rows), range(cols)):
        ax = fig.add_subplot(rows, cols, 1 + i * cols + j)
        if len(image_array[i]) > j:
            image = image_array[i][j]
            if image.shape[-1] == 1:
                image = image[..., 0]
            ax.imshow(image, **kwargs)
        else:
            ax.imshow([[0]], **kwargs)
        ax.set_xticklabels((), visible=False)
        ax.set_yticklabels((), visible=False)
        ax.tick_params(axis='both', which='both', length=0)

        if i == 0:
            ax.set_title(str(col_titles[j]))
        if j == 0:
            ax.set_ylabel(str(row_titles[i]))

    fig.canvas.set_window_title(title)


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
        self._locators = [(
            plt.AutoLocator(),
            lambda ptp: not min(ranges) < ptp < max(ranges))]
        self._locators += [
            (plt.MultipleLocator(b), lambda ptp, r=r: ptp > r)
            for b, r in zip(bases, ranges)]
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
