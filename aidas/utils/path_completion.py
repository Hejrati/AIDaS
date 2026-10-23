"""Non-recursive directory lookup for editable folder addresses."""

from __future__ import annotations

import os

from aidas.utils.filesystem import normalize_path_input


def directory_completions(text: str) -> list[tuple[str, str]]:
    """Return matching child-folder names and absolute, slash-terminated paths.

    An existing folder lists its immediate children. Otherwise, the final path
    component is a case-insensitive prefix within the preceding folder. The
    caller runs this filesystem lookup off the UI thread when necessary.
    """
    try:
        address = normalize_path_input(text)
        if not address:
            return []
        if os.path.isdir(address):
            parent, prefix = address, ""
        else:
            parent, prefix = os.path.split(address)
            parent = parent or os.curdir
        parent = os.path.abspath(parent)
        prefix = prefix.casefold()
        matches = []
        with os.scandir(parent) as entries:
            for entry in entries:
                if not entry.name.casefold().startswith(prefix):
                    continue
                try:
                    if not entry.is_dir():
                        continue
                    path = os.path.normpath(os.path.join(parent, entry.name))
                    if not path.endswith(os.sep):
                        path += os.sep
                    matches.append((entry.name, path))
                except (OSError, RuntimeError, ValueError):
                    continue
        return sorted(matches, key=lambda item: (item[0].casefold(), item[0]))
    except (OSError, RuntimeError, TypeError, ValueError):
        return []
