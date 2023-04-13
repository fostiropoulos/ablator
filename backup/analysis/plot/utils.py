import logging
from typing import Dict, List, Optional



logger = logging.getLogger(__name__)


def parse_name_remap(
    defaults: Optional[List[str]] = None, name_map: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """ """
    if name_map is not None and defaults is None:
        name_remap = name_map
    elif name_map is not None and defaults is not None:
        name_remap = {k: name_map[k] if k in name_map else k for k in defaults}
    elif defaults is not None:
        name_remap = dict(zip(defaults, defaults))
    else:
        raise NotImplementedError("`defaults` or `name_map` argument required.")
    return name_remap

