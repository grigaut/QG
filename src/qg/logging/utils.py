"""Message wrappers."""

from __future__ import annotations

from typing import Any, Literal


def assert_char(char: str) -> None:
    """Assert that a string is a character.

    Args:
        char (str): Character.

    Raises:
        ValueError: If char is not a single character.
    """
    if len(char) != 1:
        msg = "The fill character must be exactly one character long."
        raise ValueError(msg)


def pad(msg: str, char: str = "#", width: int = 1) -> str:
    """Horizontally pad a message.

    Args:
        msg (str): Message to pad.
        char (str, optional): Padding character. Defaults to "#".
        width (int, optional): Padding width. Defaults to 1.

    Returns:
        str: Padded message.
    """
    assert_char(char)
    msg_parts = msg.split("\n")
    max_len = max(len(p) for p in msg_parts)
    msg_ = [p.center(max_len, " ") for p in msg_parts]
    pad_seq = "".join([char] * width)
    msg = [pad_seq + p + pad_seq for p in msg_]
    return "\n".join(msg)


def banner(msg: str, char: str = "=") -> str:
    """Add a banner to a message.

    Args:
        msg (str): Message to banner.
        char (str, optional): Character for banner. Defaults to "=".

    Returns:
        str: Message within banner.
    """
    assert_char(char)
    msg_parts = msg.split("\n")
    max_len = max(len(p) for p in msg_parts)
    msg_ = [p.center(max_len, " ") for p in msg_parts]
    outer = "".join([char] * (max_len))
    return "\n".join([outer, *msg_, outer])


box_styles = {
    "=": ("═", "║", "╔", "╗", "╚", "╝", "╠", "╣"),
    "-": ("─", "│", "┌", "┐", "└", "┘", "├", "┤"),
    "round": ("─", "│", "╭", "╮", "╰", "╯", "├", "┤"),
    "bold": ("━", "┃", "┏", "┓", "┗", "┛", "┣", "┫"),
}

BoxStyles = Literal["-", "round", "=", "bold"]


def box(
    *msgs: str,
    style: BoxStyles = "=",
    char: str | None = None,
) -> str:
    """Draw a box around a message.

    Args:
        *msgs (str): Messages to draw a box around.
        style (BoxStyles, optional): Box style. Ignored if char is not None.
            Defaults to "=".
        char (str | None, optional): Char to use to draw the border.
            Take precedence over 'style'.Defaults to None.

    Raises:
        ValueError: If wrong style is passed.

    Returns:
        str: BOxed messages.
    """
    if char is None and style not in box_styles:
        msg = f"Available styles are {', '.join(list(box_styles.keys()))}"
        raise ValueError(msg)
    if char is not None:
        assert_char(char)
        h = v = tl = tr = bl = br = ml = mr = char
    else:
        h, v, tl, tr, bl, br, ml, mr = box_styles[style]
    max_lens = [max(len(m) for m in msg.split("\n")) for msg in msgs]
    max_len = max(max_lens)
    msgs_ljusted = [
        "\n".join(m_.ljust(m) for m_ in msg.split("\n"))
        for msg, m in zip(msgs, max_lens)
    ]
    msgs_ = [tl + "".join([h] * (max_len + 2)) + tr]
    for msg in msgs_ljusted:
        msg_c = "\n".join([m.center(max_len) for m in msg.split("\n")])
        blank_banner = banner(msg_c, " ")
        msg_pad = pad(blank_banner, " ")
        msg_pad_ = pad(msg_pad, v)
        msgs_ += msg_pad_.split("\n")
        msgs_ += [ml + "".join([h] * (max_len + 2)) + mr]
    msgs_[-1] = bl + "".join([h] * (max_len + 2)) + br
    return "\n".join(msgs_)


def step(current: int, total: int | None = None) -> str:
    """Create a string representing a step.

    Args:
        current (int): Current step.
        total (int | None, optional): Total steps. Defaults to None.

    Returns:
        str: current / total.
    """
    c_str = str(current)
    t_str = "?" * max(3, len(c_str)) if total is None else str(total)
    return c_str.zfill(len(t_str)) + "/" + t_str


def sec2text(time: float, *, squeeze_unit: bool = False) -> str:
    """Convert time in seconds to text.

    Args:
        time (float): Time in seconds.
        squeeze_unit (bool, optional): Whether to display only unit symbol.
            Defaults to False.

    Returns:
        str: Text.
    """
    if time < 60:
        s = "s" if time >= 2 else ""
        unit = "s" if squeeze_unit else f"second{s}"
        return f"{time:.1f} {unit}"
    return min2text(time / 60, squeeze_unit=squeeze_unit)


def min2text(time: float, *, squeeze_unit: bool = False) -> str:
    """Convert time in minutes to text.

    Args:
        time (float): Time in minutes.
        squeeze_unit (bool, optional): Whether to display only unit symbol.
            Defaults to False.

    Returns:
        str: Text.
    """
    if time < 60:
        s = "s" if time >= 2 else ""
        unit = "min" if squeeze_unit else f"minute{s}"
        return f"{time:.1f} {unit}"
    return hours2text(time / 60, squeeze_unit=squeeze_unit)


def hours2text(time: float, *, squeeze_unit: bool = False) -> str:
    """Convert time in hours to text.

    Args:
        time (float): Time in hours.
        squeeze_unit (bool, optional): Whether to display only unit symbol.
            Defaults to False.

    Returns:
        str: Text.
    """
    if time < 24:
        s = "s" if time >= 2 else ""
        unit = "h" if squeeze_unit else f"hour{s}"
        return f"{time:.1f} {unit}"
    return days2text(time / 24, squeeze_unit=squeeze_unit)


def days2text(time: float, *, squeeze_unit: bool = False) -> str:
    """Convert time in days to text.

    Args:
        time (float): Time in days.
        squeeze_unit (bool, optional): Whether to display only unit symbol.
            Defaults to False.

    Returns:
        str: Text.
    """
    s = "s" if time >= 2 else ""
    unit = "d" if squeeze_unit else f"day{s}"
    return f"{time:.1f} {unit}"


def meters2text(distance: float, *, squeeze_unit: bool = False) -> str:
    """Convert distance in meters to text.

    Args:
        distance (float): Distance in meters.
        squeeze_unit (bool, optional): Whether to display only unit symbol.
            Defaults to False.

    Returns:
        str: Text.
    """
    if distance < 1000:
        s = "s" if distance >= 2 else ""
        unit = "m" if squeeze_unit else f"meter{s}"
        return f"{distance:.1f} {unit}"
    return kilometers2text(distance / 1000, squeeze_unit=squeeze_unit)


def kilometers2text(distance: float, *, squeeze_unit: bool = False) -> str:
    """Convert distance in meters to text.

    Args:
        distance (float): Distance in meters.
        squeeze_unit (bool, optional): Whether to display only unit symbol.
            Defaults to False.

    Returns:
        str: Text.
    """
    s = "s" if distance >= 2 else ""
    unit = "km" if squeeze_unit else f"kilometer{s}"
    return f"{distance:.1f} {unit}"


def tree(*parts: str | list) -> str:
    """Generate a tree-like structure from nested strings and lists.

    The structure is interpreted as:
    - A string is a node
    - A list following a string contains that string's children
    - A list at the start has no parent

    Args:
        *parts: Strings or nested lists of strings.

    Returns:
        str: Tree representation of the structure.

    Examples:
        >>> print(tree("A", ["B", ["B1", "B2"], "C", ["C1"]]))
        A
        ├── B
        │   ├── B1
        │   └── B2
        └── C
            └── C1
    """
    if not parts:
        return ""

    lines = []
    _process_items(list(parts), lines, prefix="", is_root=True)
    return "\n".join(lines)


def _process_items(
    items: list[Any],
    lines: list[str],
    prefix: str = "",
    *,
    is_root: bool = False,
) -> None:
    """Process a list of items, handling strings and nested lists."""
    # Pre-process to identify which items are actual nodes (not children lists)
    nodes = []
    j = 0
    while j < len(items):
        if isinstance(items[j], str):
            children = (
                items[j + 1]
                if j + 1 < len(items) and isinstance(items[j + 1], list)
                else None
            )
            nodes.append((items[j], children))
            j += 2 if children is not None else 1
        else:
            # Orphaned list - process inline
            _process_items(items[j], lines, prefix=prefix, is_root=is_root)
            j += 1

    # Now render nodes
    for idx, (name, children) in enumerate(nodes):
        is_last = idx == len(nodes) - 1

        if is_root and idx == 0:
            lines.append(name)
        else:
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{name}")

        if children:
            extension = "    " if is_last else "│   "
            _process_items(
                children, lines, prefix=prefix + extension, is_root=False
            )
