"""Shared style, names and colors for the paper's main-text figures.

Figures are drawn at their final printed size (ACL two-column: \\textwidth = 6.3 in, \\columnwidth = 3.03 in) so font sizes
in points match the page. Game and behavior names follow Tables 1 and 2 of the paper.
"""
from pathlib import Path

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

from analysis.config import REPO, out_dir, work_dir

ROOT = REPO
PAPER_FIGS = out_dir() / "figures"  # paper figures (PDF)
PAPER_TABLES = out_dir() / "tables"  # paper table bodies (LaTeX)
DATA = work_dir()  # intermediate files: inputs from analysis/inputs.py and *_values.csv outputs
TEXT_WIDTH = 6.3
COLUMN_WIDTH = 3.03

GAME_NAMES = {  # in the order of Table 1
    "buy_sell": "Buy-Sell",
    "ultimatum": "Ultimatum",
    "casino": "CaSiNo",
    "deal_or_no_deal": "Deal or No Deal",
    "pb_capit_anchored_01": "Service Contract",
    "pb_leak_price_01": "Car Sale",
    "pb_sanity_negot_01": "Bike Sale",
}
GAMES = list(GAME_NAMES)

MODEL_NAMES = {"claudehaiku45": "Claude Haiku 4.5", "claudesonnet5": "Claude Sonnet 5", "deepseek_v4_flash": "DeepSeek V4 Flash",
               "gpt56luna": "GPT-5.6 Luna", "gpt56terra": "GPT-5.6 Terra", "qwen36_plus": "Qwen3.6 Plus",
               "qwen37_plus": "Qwen3.7 Plus", "qwen38_flash": "Qwen3.8 Flash"}

USER_FACING = ["MB_OFFER_STATE_MISREPORTING", "MB_HALLUCINATED_ACTION", "MB_UNSUPPORTED_URGENCY"]
COUNTERPARTY_FACING = ["MB_OTHER_AGENT_OFFER_STATE_MISUNDERSTANDING", "MB_OTHER_AGENT_UNSUPPORTED_URGENCY",
                       "MB_EXTRA_PROTOCOL_CONTRACTING"]
MB_NAMES = {
    "MB_OFFER_STATE_MISREPORTING": "Misreporting",
    "MB_HALLUCINATED_ACTION": "Hallucinated action",
    "MB_UNSUPPORTED_URGENCY": "User-facing unsupported urgency",
    "MB_OTHER_AGENT_OFFER_STATE_MISUNDERSTANDING": "Misunderstanding",
    "MB_OTHER_AGENT_UNSUPPORTED_URGENCY": "Counterparty-facing unsupported urgency",
    "MB_EXTRA_PROTOCOL_CONTRACTING": "Out-of-scope terms",
}
MB_TICK_NAMES = {  # same names, line-broken for x tick labels
    "MB_OFFER_STATE_MISREPORTING": "Misreporting",
    "MB_HALLUCINATED_ACTION": "Hallucinated\naction",
    "MB_UNSUPPORTED_URGENCY": "User-facing\nunsupported\nurgency",
    "MB_OTHER_AGENT_OFFER_STATE_MISUNDERSTANDING": "Misunderstanding",
    "MB_OTHER_AGENT_UNSUPPORTED_URGENCY": "Counterparty-facing\nunsupported\nurgency",
    "MB_EXTRA_PROTOCOL_CONTRACTING": "Out-of-scope\nterms",
}
assert all(" ".join(MB_TICK_NAMES[b].split()) == MB_NAMES[b] for b in MB_NAMES)
MB_GROUP_NAMES = {"user_facing": "User-facing", "non_user_facing": "Counterparty-facing"}

# One color per condition, shared by every figure (validated: adjacent pairs clear CVD and normal-vision separation).
CONDITION_COLORS = {
    "no_user": "#A3A29B",
    "single_user": "#EB6834", "with_user_llm": "#EB6834",
    "user_on_agent_1": "#B5461A", "user_on_agent_2": "#F28C55",
    "two_users": "#2A78D6",
    "silent": "#F8CDB6",  # user-ablation ramp: one hue, darker = more user involvement
    "initial_consult_only": "#F2A07B",
}
CONDITION_NAMES = {
    "no_user": "No user", "single_user": "With user", "with_user_llm": "With user", "two_users": "Two users",
    "user_on_agent_1": "User on Agent 1", "user_on_agent_2": "User on Agent 2",
    "silent": "Passive user", "initial_consult_only": "Opening-only user",
}
ACCENT = CONDITION_COLORS["single_user"]
NS_COLOR = "#D3D2CC"   # not significant
INK, MUTED, GRID, AXIS = "#1F1F1E", "#52514E", "#E6E5DF", "#8F8E88"


def apply() -> None:
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 1.0, "legend.handleheight": 0.8,
        "legend.handletextpad": 0.4, "legend.columnspacing": 1.0, "legend.borderaxespad": 0.3,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 4, "axes.labelpad": 3,
        "axes.axisbelow": True, "axes.unicode_minus": True,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "grid.color": GRID, "grid.linewidth": 0.5, "hatch.linewidth": 0.5, "hatch.color": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def ygrid(ax) -> None:
    ax.grid(axis="y")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.tick_params(axis="x", length=0)


def save(fig, pdf_name: str | None, png_path: Path | None = None) -> None:
    """Write the paper PDF (deterministic: no creation date) and an optional PNG preview next to the analysis outputs."""
    if pdf_name:
        (PAPER_FIGS / f"{pdf_name}.pdf").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_FIGS / f"{pdf_name}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
        print(f"Saved {PAPER_FIGS / f'{pdf_name}.pdf'}")
    if png_path is not None:
        fig.savefig(png_path, dpi=300, facecolor="white")


def tex_num(value: float, digits: int = 1, sign: bool = False, int_width: int = 1) -> str:
    """Number for a LaTeX table: fixed decimals, optional explicit plus. Both signs are set in math mode so + and - have
    the same width; with sign=True a value that rounds to zero gets an invisible sign of that width. int_width pads the
    integer part with invisible digits (\\phantom{0}), so every entry of a centered column has the same width and the
    decimal points line up while the numbers stay centered under their header."""
    text = f"{abs(value):.{digits}f}"
    pad = r"\phantom{" + "0" * max(0, int_width - len(text.split(".")[0])) + "}" if int_width > len(text.split(".")[0]) else ""
    # the padding goes before the sign, so the sign stays attached to its digits
    if text.strip("0.") == "":  # rounds to zero: no sign
        return pad + (r"\phantom{$+$}" if sign else "") + text
    if value < 0:
        return pad + "$-$" + text
    return pad + ("$+$" if sign else "") + text


def tex_ci(low: float, high: float, digits: int = 1) -> str:
    return f"[{tex_num(low, digits)}, {tex_num(high, digits)}]"


def tex_sig(value: float, p: float | None, digits: int = 1, sign: bool = True, int_width: int = 1) -> str:
    """Table cell: the estimate in black when p < .05, in gray when not significant (p >= .05). p=None: never gray
    (e.g. a rate, which is not tested against zero). Bold is left for marking the best entry, by convention."""
    text = tex_num(value, digits, sign=sign, int_width=int_width)
    return text if p is None or p < 0.05 else rf"{{\color{{gray}}{text}}}"


def tex_game_header(game: str) -> str:
    """Game name for a narrow table column header, with a line break in the two long names."""
    breaks = {"Deal or No Deal": r"Deal or\\No Deal", "Service Contract": r"Service\\Contract"}
    name = GAME_NAMES[game]
    return "{" + breaks[name] + "}" if name in breaks else name


def tex_pm(half_width: float, digits: int = 1, int_width: int = 1, size: str = r"\footnotesize",
           hang: str | None = None) -> str:
    """95% CI half-width in parentheses, e.g. '(+-2.1)', for aggregates whose intervals are symmetric; int_width pads the
    integer part as in tex_num so a column of half-widths lines up.

    hang: for a CI on its own line under a number in the same column. "sign" -- the opening parenthesis hangs outside
    (zero width), so the +- takes the place of the number's sign and the digits line up with the digits above;
    "all" -- "(+-" hangs outside, for unsigned numbers; "bare" -- no parentheses, "+-" hangs outside (narrow columns). The closing parenthesis hangs too, so a CI line is no wider
    than the number above it."""
    text = f"{half_width:.{digits}f}"
    pad = "0" * max(0, int_width - len(text.split(".")[0]))
    pad = rf"\phantom{{{pad}}}" if pad else ""
    if hang == "sign":
        body = pad + r"\llap{(}$\pm$" + text + r"\rlap{)}"
    elif hang == "bare":  # no parentheses; "+-" hangs left, so the digits line up with the number above
        body = pad + r"\llap{$\pm$}" + text
    elif hang == "all":
        body = pad + r"\llap{($\pm$}" + text + r"\rlap{)}"
    else:
        body = pad + rf"($\pm${text})"  # padding outside the parentheses
    return rf"{{{size} {body}}}" if size else "{" + body + "}"


# ---- decimal-aligned table cells -------------------------------------------------------------------------------------
# Every numeric column is two tblr columns with no gap between them: the integer part with its sign, right-aligned
# (DEC_INT), and the decimal point with the fraction, left-aligned (DEC_FRAC). The decimal points of all entries in a
# column -- values and the CI lines under them -- then line up exactly, whatever the glyph widths, and each sign or +-
# stays attached to its own digits. A header or a "---" spans both parts (\SetCell[c=2]).
DEC_INT, DEC_FRAC = "Q[r,m,rightsep=0pt]", "Q[l,m,leftsep=0pt]"
DEC_COLS = DEC_INT + " " + DEC_FRAC


def _style(parts: tuple[str, str], gray: bool = False, italic: bool = False) -> str:
    """Join an (integer, fraction) pair into two tblr cells, optionally gray (not significant) or italic (reference)."""
    pre = (r"\color{gray}" if gray else "") + (r"\itshape " if italic else "")
    return " & ".join("{" + pre + p + "}" if pre else p for p in parts)


def dec_value(value: float, digits: int = 1, sign: bool = False, gray: bool = False, italic: bool = False) -> str:
    """Two cells for a value: '+35' & '.8'. A value that rounds to zero carries no sign."""
    text = f"{abs(value):.{digits}f}"
    whole, frac = text.split(".")
    mark = "" if text.strip("0.") == "" else ("$-$" if value < 0 else ("$+$" if sign else ""))
    return _style((mark + whole, "." + frac), gray, italic)


def dec_sig(value: float, p: float | None, digits: int = 1, sign: bool = True, italic: bool = False) -> str:
    """dec_value, gray when not significant (p >= .05); p=None means untested and never gray."""
    return dec_value(value, digits, sign, gray=p is not None and p >= 0.05, italic=italic)


def dec_ci(half_width: float, digits: int = 1, italic: bool = False, parens: bool = False) -> str:
    """Two cells for a 95% CI half-width, '(+-6' & '.3)' (or '+-6' & '.3' without parentheses): its decimal point lines
    up with the values in the column."""
    whole, frac = f"{half_width:.{digits}f}".split(".")
    if parens:
        return _style((rf"($\pm${whole}", f".{frac})"), italic=italic)
    return _style((rf"$\pm${whole}", f".{frac}"), italic=italic)


# tblr option for decimal-aligned tables whose numbers should be centered under a wider header: the extra width goes to
# all spanned columns evenly. Left-aligned tables omit it (tblr then gives the extra width to the fraction part).
DEC_HSPAN = "hspan=even,"


def tex_model_header(model: str) -> str:
    """Model name for a narrow column header, broken after the first word ('Claude\\\\Haiku 4.5')."""
    name = MODEL_NAMES[model]
    first, _, rest = name.partition(" ")
    return "{" + first + r"\\" + rest + "}" if rest else name


DEC_EMPTY = " & "                      # an empty decimal cell (both parts)
DEC_NA = r"\SetCell[c=2]{c} --- & "    # "not applicable", centered over both parts


def save_table(tex: str, name: str) -> None:
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    path = PAPER_TABLES / f"{name}.tex"
    path.write_text(tex, encoding="utf-8")
    print(f"Saved {path}")
