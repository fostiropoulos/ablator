import logging


logger = logging.getLogger(__name__)


def parse_name_remap(
    defaults: list[str] | None = None, name_map: dict[str, str] | None = None
) -> dict[str, str]:
    """
    Returns a dictionary mapping input attribute names to output attribute names,
    with optional remapping based on ``name_map``.

    Parameters
    ----------
    defaults : list of str or None, optional
        The default attribute names to use as keys in the output dictionary.
        If ``None``, the output dictionary will be based on ``name_map`` only.
    name_map : dict of str to str or None, optional
        A dictionary mapping input attribute names to output attribute names.
        If ``None``, the output dictionary will be based on ``defaults`` only.

    Returns
    -------
    dict of str to str
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
