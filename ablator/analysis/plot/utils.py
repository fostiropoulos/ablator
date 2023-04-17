import logging


logger = logging.getLogger(__name__)


def parse_name_remap(
    defaults: list[str] | None = None, name_map: dict[str, str] | None = None
) -> dict[str, str]:
    if name_map is not None and defaults is None:
        name_remap = name_map
    elif name_map is not None and defaults is not None:
        name_remap = {k: name_map[k] if k in name_map else k for k in defaults}
    elif defaults is not None:
        name_remap = dict(zip(defaults, defaults))
    else:
        raise NotImplementedError("`defaults` or `name_map` argument required.")
    return name_remap
