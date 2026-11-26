"""Visual identity of the Smart Gate console.

Design choices
--------------
* A light, cool-grey operator console with a deep navy header, similar to
  commercial access-control software. Colour is reserved for what the
  operator must notice: the verdict (green / red / amber) and the plate.
* The plate readout imitates the real plate (Mercosul: blue strip, old:
  grey) so the operator compares the reading with what he sees on camera.
* The simulated gate is built with plain CSS (no images, no JavaScript). The
  animation lasts exactly ``gate_open_seconds``: it rises, stays open and
  lowers again, and each run gets its own keyframes so it restarts every time
  a new vehicle is authorized.

Every function returns HTML on a single line: Streamlit's markdown parser
treats blank lines and indented lines as code blocks. Every dynamic text is
escaped with :func:`html.escape` because the HTML is rendered with
``unsafe_allow_html``.
"""

from __future__ import annotations

from html import escape

import streamlit as st

from database.database import AccessResult
from modules.plate_format import PlateFormat, detect_format, format_for_display

NAVY = "#14212E"
STEEL = "#EDF0F4"
GREEN = "#15803D"
RED = "#B91C1C"
AMBER = "#B45309"
AMBER_LIGHT = "#F59E0B"

# access result -> (css modifier, headline shown on the verdict banner)
VERDICTS: dict[AccessResult, tuple[str, str]] = {
    AccessResult.AUTHORIZED: ("ok", "Acesso autorizado"),
    AccessResult.DENIED: ("denied", "Acesso negado"),
    AccessResult.MANUAL_REVIEW: ("review", "Revisão manual"),
}

CSS = f"""
<style>
.stApp {{ background: {STEEL}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1rem; max-width: 1280px; }}
.stApp, .stApp button, .stApp input, .stApp textarea {{
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}}

/* ---- brand bar ---- */
.sg-brand {{
  background: {NAVY}; color: #fff; border-radius: 10px; padding: 14px 22px; margin-bottom: 18px;
  display: flex; align-items: center; gap: 16px;
}}
.sg-brand-mark {{
  width: 40px; height: 40px; border-radius: 8px; background: #fff; position: relative; flex: none;
}}
.sg-brand-mark::before {{ content: ""; position: absolute; left: 7px; top: 9px; width: 26px; height: 5px; background: {NAVY}; border-radius: 2px; }}
.sg-brand-mark::after {{ content: ""; position: absolute; left: 7px; top: 9px; width: 5px; height: 22px; background: {NAVY}; border-radius: 2px; }}
.sg-brand-title {{ font-size: 1.35rem; font-weight: 700; letter-spacing: .01em; line-height: 1.2; }}
.sg-brand-sub {{ font-size: .85rem; color: #AEBBCB; }}

/* ---- verdict banner ---- */
.sg-verdict {{
  display: flex; align-items: center; gap: 20px; border-radius: 10px; padding: 22px 26px; color: #fff;
  margin-bottom: 16px;
}}
.sg-verdict-ok {{ background: {GREEN}; }}
.sg-verdict-denied {{ background: {RED}; }}
.sg-verdict-review {{ background: {AMBER}; }}
.sg-verdict-light {{
  width: 58px; height: 58px; border-radius: 50%; flex: none;
  background: #fff; box-shadow: 0 0 0 6px rgba(255,255,255,.28);
}}
.sg-verdict-review .sg-verdict-light {{ background: {AMBER_LIGHT}; box-shadow: 0 0 0 6px rgba(255,255,255,.35); }}
.sg-verdict-title {{ font-size: 2.1rem; font-weight: 800; line-height: 1.1; }}
.sg-verdict-sub {{ font-size: 1.02rem; opacity: .95; margin-top: 4px; }}

/* ---- plate readout ---- */
.sg-plate {{
  display: inline-block; border: 3px solid #1b1b1b; border-radius: 10px; overflow: hidden;
  background: #fff; min-width: 270px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,.18);
}}
.sg-plate-strip {{
  background: #1F4FA3; color: #fff; font-size: .72rem; font-weight: 700; letter-spacing: .3em; padding: 2px 0;
}}
.sg-plate-chars {{
  font-family: Consolas, "DejaVu Sans Mono", "Courier New", monospace; font-weight: 800;
  font-size: 2.9rem; letter-spacing: .08em; color: #111; padding: 4px 18px 8px; line-height: 1.15;
}}
.sg-plate-old {{ background: #D8DADD; }}
.sg-plate-old .sg-plate-strip {{ background: #8A8D92; }}
.sg-plate-empty .sg-plate-chars {{ color: #8B939C; font-size: 1.6rem; padding: 14px 18px; }}
.sg-plate-caption {{ color: #5B6775; font-size: .85rem; margin-top: 6px; }}

/* ---- simulated gate ---- */
.sg-gate {{ position: relative; height: 230px; margin-top: 4px; overflow: hidden; }}
.sg-gate-road {{ position: absolute; left: 0; right: 0; bottom: 0; height: 12px; background: #3B4551; border-radius: 3px; }}
.sg-gate-post {{
  position: absolute; left: 16px; bottom: 12px; width: 34px; height: 76px; background: {NAVY};
  border-radius: 6px 6px 2px 2px;
}}
.sg-gate-lamp {{
  position: absolute; left: 10px; top: 9px; width: 14px; height: 14px; border-radius: 50%; background: var(--lamp, #9AA4AF);
}}
.sg-gate-arm {{
  position: absolute; left: 33px; bottom: 74px; width: 150px; height: 12px; border-radius: 6px;
  transform-origin: 0 50%; transform: rotate(0deg);
  background: repeating-linear-gradient(90deg, #F4F4F4 0 28px, #D62828 28px 56px);
  border: 1px solid #8C939B;
}}
.sg-gate-caption {{ color: #5B6775; font-size: .9rem; margin-top: 4px; }}
.sg-gate-open-label {{ color: {GREEN}; font-weight: 700; }}

/* ---- small details ---- */
.sg-chip {{
  display: inline-block; background: #E1E6ED; color: #33414F; border-radius: 999px; padding: 2px 12px;
  font-size: .82rem; font-weight: 600; margin-right: 6px;
}}
.sg-chip-manual {{ background: #DDE8FA; color: #1B4B97; }}

@media (prefers-reduced-motion: reduce) {{
  .sg-gate-arm {{ animation-timing-function: steps(1, end) !important; }}
}}
</style>
"""


def inject_css() -> None:
    """Apply the global stylesheet (call once per run, from ``app.py``)."""
    st.markdown(CSS, unsafe_allow_html=True)


def brand_bar_html(title: str = "Smart Gate", subtitle: str = "Controle de acesso por leitura de placas") -> str:
    return (
        '<div class="sg-brand"><div class="sg-brand-mark"></div><div>'
        f'<div class="sg-brand-title">{escape(title)}</div>'
        f'<div class="sg-brand-sub">{escape(subtitle)}</div>'
        "</div></div>"
    )


def verdict_html(result: AccessResult, subtitle: str) -> str:
    """Large green / red / amber banner with the outcome of the access attempt."""
    modifier, headline = VERDICTS[AccessResult(result)]
    return (
        f'<div class="sg-verdict sg-verdict-{modifier}"><div class="sg-verdict-light"></div><div>'
        f'<div class="sg-verdict-title">{escape(headline)}</div>'
        f'<div class="sg-verdict-sub">{escape(subtitle)}</div>'
        "</div></div>"
    )


def plate_html(plate: str | None, caption: str = "") -> str:
    """Plate-shaped readout; shows a placeholder when nothing valid was read."""
    cap = f'<div class="sg-plate-caption">{escape(caption)}</div>' if caption else ""
    if not plate:
        return (f'<div><div class="sg-plate sg-plate-empty"><div class="sg-plate-chars">Sem leitura</div></div>'
                f"{cap}</div>")
    old = detect_format(plate) is PlateFormat.OLD
    css = "sg-plate sg-plate-old" if old else "sg-plate"
    strip = "BRASIL"
    return (f'<div><div class="{css}"><div class="sg-plate-strip">{strip}</div>'
            f'<div class="sg-plate-chars">{escape(format_for_display(plate))}</div></div>{cap}</div>')


def gate_html(opening: bool, result: AccessResult, open_seconds: int, run_id: int = 0) -> str:
    """Simulated gate.

    Args:
        opening: play the opening animation (only for an authorized access
            that has not been animated yet). Otherwise the gate is shown closed.
        result: decides the colour of the lamp on the post when closed.
        open_seconds: total duration of the animation (rise, hold, lower).
        run_id: unique per animation, so the keyframes restart on every run.
    """
    seconds = max(1, int(open_seconds))
    closed_lamp = {AccessResult.DENIED: "#DC2626", AccessResult.MANUAL_REVIEW: AMBER_LIGHT}.get(result, "#9AA4AF")
    if opening:
        # rises in the first 12 % of the time, stays up, lowers in the last 12 %
        style = (
            f"<style>@keyframes sg-arm-{run_id}{{0%{{transform:rotate(0deg)}}12%{{transform:rotate(-70deg)}}"
            f"88%{{transform:rotate(-70deg)}}100%{{transform:rotate(0deg)}}}}"
            f"@keyframes sg-lamp-{run_id}{{0%{{background:#DC2626}}12%{{background:#22C55E}}"
            f"88%{{background:#22C55E}}100%{{background:#DC2626}}}}"
            f".sg-run-{run_id} .sg-gate-arm{{animation:sg-arm-{run_id} {seconds}s ease-in-out 1 forwards}}"
            f".sg-run-{run_id} .sg-gate-lamp{{animation:sg-lamp-{run_id} {seconds}s linear 1 forwards}}</style>"
        )
        caption = f'<span class="sg-gate-open-label">Cancela aberta</span> por {seconds} s e fecha em seguida.'
        return (f'{style}<div class="sg-run-{run_id}"><div class="sg-gate" style="--lamp:#DC2626">'
                '<div class="sg-gate-post"><span class="sg-gate-lamp"></span></div><div class="sg-gate-arm"></div>'
                f'<div class="sg-gate-road"></div></div><div class="sg-gate-caption">{caption}</div></div>')
    caption = "Cancela fechada."
    return (f'<div class="sg-gate" style="--lamp:{closed_lamp}"><div class="sg-gate-post">'
            '<span class="sg-gate-lamp"></span></div><div class="sg-gate-arm"></div>'
            f'<div class="sg-gate-road"></div></div><div class="sg-gate-caption">{caption}</div>')


def chip_html(text: str, manual: bool = False) -> str:
    css = "sg-chip sg-chip-manual" if manual else "sg-chip"
    return f'<span class="{css}">{escape(text)}</span>'
