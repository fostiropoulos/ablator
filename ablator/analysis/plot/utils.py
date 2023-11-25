import logging

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


logger = logging.getLogger(__name__)


def get_axes_fig(ax: Axes) -> Figure:
    """
    Gets the Figure from the axes or the currently active figure if it fails to
    find one attached to the axes, `ax`.

    Parameters
    ----------
    ax : Axes
        The axes that is attached to a Figure.

    Returns
    -------
    Figure
        The currently active figure or the figure to which the axes is attached.

    Raises
    ------
    RuntimeError
        When it is unable to find an active plot attached to the axes
        or the matplotlib environment.
    """
    if (fig := ax.get_figure()) is not None:
        return fig
    if len(plt.get_fignums()) > 0:
        return plt.gcf()
    raise RuntimeError("Can not find an active plot.")


def parse_name_remap(
    defaults: list[str] | None = None, name_map: dict[str, str] | None = None
) -> dict[str, str]:
    """
    Returns a dictionary mapping input attribute names to output attribute names,
    with optional remapping based on ``name_map``.

    Parameters
    ----------
    defaults : list[str] | None
        The default attribute names to use as keys in the output dictionary.
        If ``None``, the output dictionary will be based on ``name_map`` only.
    name_map : dict[str, str] | None
        A dictionary mapping input attribute names to output attribute names.
        If ``None``, the output dictionary will be based on ``defaults`` only.

    Returns
    -------
    dict[str, str]
        A dictionary mapping input attribute names to output attribute names.

    Raises
    ------
    NotImplementedError
        If ``defaults`` and ``name_map`` are both ``None``.

    Examples
    --------
    >>> defaults = ["attr1", "attr2", "attr3"]
    >>> name_map = {"attr2": "new_attr2", "attr4": "attr4_renamed"}
    >>> name_remap = parse_name_remap(defaults, name_map)
    >>> assert name_remap == {"attr1": "attr1", "attr2": "new_attr2", "attr3": "attr3"}
    >>> name_remap = parse_name_remap(defaults)
    >>> assert name_remap == {"attr1": "attr1", "attr2": "attr2", "attr3": "attr3"}
    >>> name_remap = parse_name_remap(name_map=name_map)
    >>> assert name_remap == {"attr2": "new_attr2", "attr4": "attr4_renamed"}
    """
    if name_map is not None and defaults is None:
        name_remap = name_map
    elif name_map is not None and defaults is not None:
        name_remap = {k: name_map[k] if k in name_map else k for k in defaults}
    elif defaults is not None:
        name_remap = dict(zip(defaults, defaults))
    else:
        raise NotImplementedError("`defaults` or `name_map` argument required.")
    return name_remap
