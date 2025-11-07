"""Message wrappers."""

from __future__ import annotations

from typing import Literal


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
    max_len = max(max(len(m) for m in msg.split("\n")) for msg in msgs)
    msgs_ = [tl + "".join([h] * (max_len + 2)) + tr]
    for msg in msgs:
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


def sec2text(time: float) -> str:
    """Convert time in seconds to text.

    Args:
        time (float): Time in seconds.

    Returns:
        str: Text.
    """
    if time < 60:  # noqa: PLR2004
        s = "s" if time >= 2 else ""  # noqa: PLR2004
        return f"{time:.1f} second{s}"
    return min2text(time / 60)


def min2text(time: float) -> str:
    """Convert time in minutes to text.

    Args:
        time (float): Time in minutes.

    Returns:
        str: Text.
    """
    if time < 60:  # noqa: PLR2004
        s = "s" if time >= 2 else ""  # noqa: PLR2004
        return f"{time:.1f} minute{s}"
    return hours2text(time / 60)


def hours2text(time: float) -> str:
    """Convert time in hours to text.

    Args:
        time (float): Time in hours.

    Returns:
        str: Text.
    """
    if time < 24:  # noqa: PLR2004
        s = "s" if time >= 2 else ""  # noqa: PLR2004
        return f"{time:.1f} hour{s}"
    return days2text(time / 24)


def days2text(time: float) -> str:
    """Convert time in days to text.

    Args:
        time (float): Time in days.

    Returns:
        str: Text.
    """
    if time < 365:
        s = "s" if time >= 2 else ""  # noqa: PLR2004
        return f"{time:.1f} day{s}"
    return years2text(time / 365)


def years2text(time: float) -> str:
    """Convert time in years to text.

    Args:
        time (float): Time in years.

    Returns:
        str: Text.
    """
    s = "s" if time >= 2 else ""  # noqa: PLR2004
    return f"{time:.1f} year{s}"
