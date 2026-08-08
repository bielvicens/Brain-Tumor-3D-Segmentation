from __future__ import annotations

"""Shared visual identity for the Brain Tumor 3D Segmentation app.

Design direction — "radiology workstation": a dark reading-room canvas (so
skull-stripped MRI slices, which are black outside the brain, sit flush
against the app instead of floating in a white square), a single restrained
cyan-teal accent for interactive chrome, and monospace type for every
numeric readout (shapes, voxel counts, slice index) the way a PACS/DICOM
viewer renders measurements. The tumor sub-region legend (red / amber /
violet) is deliberately a *different* chromatic family from the interface
accent, so at a glance "this is data" reads differently from "this is UI".

Every page should call `inject_theme()` once, right after `st.set_page_config`,
and then use the `render_*` helpers below instead of raw `st.markdown` HTML so
the whole app stays visually consistent.
"""

import streamlit as st
from matplotlib.colors import ListedColormap

# --------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#0a0f1a",
    "surface": "#121a29",
    "surface_2": "#0e1524",
    "border": "#22304a",
    "border_strong": "#324160",
    "text": "#e8edf6",
    "text_muted": "#8fa0bd",
    "text_faint": "#5b6c8a",
    "accent": "#34c9c2",
    "accent_strong": "#5eead4",
    "success": "#34c98f",
    "warning": "#e2a13d",
    "danger": "#ef5757",
}

# BraTS tumor sub-region legend. Class 0 (background) is intentionally
# excluded: it's masked out of the overlay entirely rather than tinted, so
# only the tumor sub-regions show. Colors are chosen from a different hue
# family than COLORS["accent"] so the data layer never gets mistaken for UI
# chrome, and deliberately avoid a red/green pairing.
CLASS_INFO = {
    1: ("NCR", "Necrotic core", "#ef4444"),
    2: ("ED", "Edema", "#eab308"),
    3: ("ET", "Enhancing tumor", "#a78bfa"),
}
OVERLAY_CMAP = ListedColormap([CLASS_INFO[c][2] for c in sorted(CLASS_INFO)])

FONT_LINKS = (
    "https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@500;600;700&"
    "family=Inter:wght@400;500;600&"
    "family=JetBrains+Mono:wght@500;600&display=swap"
)


def _css_vars() -> str:
    mapping = {
        "bg": COLORS["bg"],
        "surface": COLORS["surface"],
        "surface-2": COLORS["surface_2"],
        "border": COLORS["border"],
        "border-strong": COLORS["border_strong"],
        "text": COLORS["text"],
        "text-muted": COLORS["text_muted"],
        "text-faint": COLORS["text_faint"],
        "accent": COLORS["accent"],
        "accent-strong": COLORS["accent_strong"],
        "success": COLORS["success"],
        "warning": COLORS["warning"],
        "danger": COLORS["danger"],
    }
    return "\n".join(f"  --{key}: {value};" for key, value in mapping.items())


_CSS_TEMPLATE = """
<style>
@import url('{font_links}');

:root {{
{css_vars}
  --radius-lg: 16px;
  --radius-md: 12px;
  --radius-sm: 8px;
  --font-display: 'Space Grotesk', 'Inter', sans-serif;
  --font-body: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}}

.stApp {{
  background: var(--bg);
  font-family: var(--font-body);
}}
[data-testid="stSidebar"] {{
  background: var(--surface-2);
  border-right: 1px solid var(--border);
}}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}

.main .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }}

h1, h2, h3 {{ font-family: var(--font-display); color: var(--text); }}
p, li, span, label {{ color: var(--text); }}
hr {{ border-color: var(--border) !important; }}

:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border-strong); border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

/* Widget labels read like instrument-panel captions */
[data-testid="stWidgetLabel"] p {{
  font-family: var(--font-mono) !important;
  font-size: .72rem !important;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--text-muted) !important;
}}
[data-testid="stButton"] button {{
  border-radius: var(--radius-sm);
  font-family: var(--font-body);
  font-weight: 600;
  letter-spacing: .01em;
}}

/* Eyebrow / hero -------------------------------------------------- */
.eyebrow {{
  font-family: var(--font-mono);
  font-size: .72rem;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: .55rem;
}}
.hero-title {{
  font-family: var(--font-display);
  font-weight: 700;
  letter-spacing: -.02em;
  color: var(--text);
  margin: 0;
  line-height: 1.12;
}}
.hero-subtitle {{
  font-family: var(--font-body);
  font-size: 1.02rem;
  color: var(--text-muted);
  margin-top: .65rem;
  max-width: 660px;
  line-height: 1.55;
}}

/* Ruler signature — a slice-position ruler, echoed above the slice
   slider on the Predict page. */
.ruler {{
  height: 20px;
  margin: 1.9rem 0 2.3rem;
  border-top: 1px solid var(--border-strong);
  background-image:
    repeating-linear-gradient(to right, var(--border-strong) 0 1px, transparent 1px 40px),
    repeating-linear-gradient(to right, var(--border) 0 1px, transparent 1px 8px);
  background-position: top left, top left;
  background-size: 100% 13px, 100% 6px;
  background-repeat: no-repeat;
}}
.ruler.compact {{ margin: .9rem 0 1.4rem; height: 14px; }}

/* Feature grid ------------------------------------------------------ */
.feature-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }}
@media (max-width: 900px) {{ .feature-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
.feature-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.35rem 1.3rem;
  transition: border-color .15s ease, transform .15s ease;
}}
.feature-card:hover {{ border-color: var(--accent); transform: translateY(-2px); }}
@media (prefers-reduced-motion: reduce) {{ .feature-card {{ transition: none !important; }} }}
.feature-icon {{ font-size: 1.35rem; display: block; margin-bottom: .5rem; }}
.feature-tag {{
  font-family: var(--font-mono); font-size: .66rem; letter-spacing: .1em;
  color: var(--accent); text-transform: uppercase;
}}
.feature-card h4 {{
  font-family: var(--font-display); font-size: 1rem; font-weight: 600;
  color: var(--text); margin: .4rem 0 .4rem;
}}
.feature-card p {{ color: var(--text-muted); font-size: .87rem; line-height: 1.5; margin: 0; }}

/* Sidebar brand ------------------------------------------------------ */
.sidebar-brand {{
  display: flex; align-items: center; gap: .65rem;
  padding: .1rem 0 1.1rem; margin-bottom: 1.1rem;
  border-bottom: 1px solid var(--border);
}}
.sidebar-brand-icon {{ font-size: 1.35rem; line-height: 1; }}
.sidebar-brand-name {{ font-family: var(--font-display); font-weight: 700; font-size: .95rem; color: var(--text); line-height: 1.2; }}
.sidebar-brand-sub {{ font-family: var(--font-mono); font-size: .64rem; letter-spacing: .08em; color: var(--text-faint); text-transform: uppercase; }}
.sidebar-section-label {{
  font-family: var(--font-mono); font-size: .68rem; letter-spacing: .12em;
  text-transform: uppercase; color: var(--text-faint); margin: 0 0 .5rem;
}}

/* Metric readouts ----------------------------------------------------- */
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; }}
.metric-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 1.05rem 1.15rem; }}
.metric-label {{ font-family: var(--font-mono); font-size: .66rem; letter-spacing: .1em; text-transform: uppercase; color: var(--text-faint); margin-bottom: .4rem; }}
.metric-value {{ font-family: var(--font-mono); font-size: 1.45rem; font-weight: 600; color: var(--text); }}

/* Stat grid — like metric-card, but with an optional class-color accent
   bar and caption, for per-class breakdowns (e.g. segmentation stats). */
.stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; }}
.stat-card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
  padding: 1.05rem 1.15rem 1.1rem; border-left: 3px solid var(--border-strong);
}}
.stat-label {{
  font-family: var(--font-mono); font-size: .66rem; letter-spacing: .1em; text-transform: uppercase;
  color: var(--text-faint); margin-bottom: .4rem; display: flex; align-items: center; gap: .4rem;
}}
.stat-value {{ font-family: var(--font-mono); font-size: 1.4rem; font-weight: 600; color: var(--text); }}
.stat-caption {{ font-size: .78rem; color: var(--text-muted); margin-top: .3rem; }}

/* Section heading — an underlined mono label for a page section, with an
   optional right-aligned meta value (e.g. the active slice index). */
.section-heading {{
  font-family: var(--font-mono); font-size: .78rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--text-muted); margin: 0 0 .9rem; padding-bottom: .55rem; border-bottom: 1px solid var(--border);
  display: flex; align-items: baseline; justify-content: space-between; gap: 1rem;
}}
.section-heading .meta {{ color: var(--text-faint); letter-spacing: .08em; white-space: nowrap; }}

/* Status banner --------------------------------------------------------- */
.status-banner {{ padding: .8rem 1.05rem; border-radius: var(--radius-md); font-size: .92rem; border: 1px solid; margin-bottom: 1.15rem; }}
.status-banner.success {{ background: rgba(52,201,143,.08); border-color: rgba(52,201,143,.35); color: var(--success); }}
.status-banner.info {{ background: rgba(52,201,194,.08); border-color: rgba(52,201,194,.35); color: var(--accent-strong); }}
.status-banner strong {{ color: var(--text); font-family: var(--font-mono); font-weight: 600; }}

/* Viewer panel labels ----------------------------------------------------- */
.panel-label {{
  font-family: var(--font-mono); font-size: .72rem; letter-spacing: .1em;
  text-transform: uppercase; color: var(--text-muted); margin: 0 0 .65rem;
  display: flex; align-items: center; gap: .45rem;
}}
.panel-label .dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--accent); display: inline-block; flex-shrink: 0; }}

/* Legend chips ----------------------------------------------------------- */
.legend-row {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; }}
.legend-chip {{
  display: inline-flex; align-items: center; gap: .4rem; font-size: .78rem;
  color: var(--text-muted); background: var(--surface-2); border: 1px solid var(--border);
  padding: .3rem .62rem; border-radius: 999px;
}}
.legend-swatch {{ width: .6rem; height: .6rem; border-radius: 2px; display: inline-block; flex-shrink: 0; }}
.legend-code {{ font-family: var(--font-mono); color: var(--text); font-weight: 600; }}

/* Empty state -------------------------------------------------------------- */
.empty-state {{ text-align: center; padding: 3.4rem 1.5rem; border: 1px dashed var(--border-strong); border-radius: var(--radius-lg); }}
.empty-state .icon {{ font-size: 1.9rem; margin-bottom: .6rem; }}
.empty-state h4 {{ font-family: var(--font-display); color: var(--text); margin: .1rem 0 .4rem; }}
.empty-state p {{ color: var(--text-muted); font-size: .9rem; margin: 0; }}
</style>
"""


def inject_theme() -> None:
    """Inject fonts + the shared CSS system. Call once per page, right after
    `st.set_page_config`.

    Uses `st.html()` rather than `st.html(...)`:
    the latter runs content through Streamlit's Markdown parser, which can
    treat a blank line inside a long `<style>` block as "end of raw HTML"
    and start rendering the rest as literal text. `st.html()` inserts the
    HTML directly with no Markdown parsing, so this can't happen.
    """
    st.html(_CSS_TEMPLATE.format(font_links=FONT_LINKS, css_vars=_css_vars()))


# --------------------------------------------------------------------------
# Reusable components
# --------------------------------------------------------------------------
def render_hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.html(f'<p class="eyebrow">{eyebrow}</p>'
        f'<h1 class="hero-title" style="font-size:2.55rem;">{title}</h1>'
        f'<p class="hero-subtitle">{subtitle}</p>')


def render_page_header(eyebrow: str, icon: str, title: str, subtitle: str) -> None:
    st.html(f'<p class="eyebrow">{eyebrow}</p>'
        f'<h1 class="hero-title" style="font-size:2rem;">{icon} {title}</h1>'
        f'<p class="hero-subtitle" style="margin-bottom:.4rem;">{subtitle}</p>')


def render_ruler(compact: bool = False) -> None:
    css_class = "ruler compact" if compact else "ruler"
    st.html(f'<div class="{css_class}"></div>')


def render_feature_grid(features: list[tuple[str, str, str, str]]) -> None:
    """`features` is a list of (tag, icon, title, description)."""
    cards = "".join(
        f'<div class="feature-card">'
        f'<span class="feature-icon">{icon}</span>'
        f'<span class="feature-tag">{tag}</span>'
        f"<h4>{title}</h4>"
        f"<p>{description}</p>"
        f"</div>"
        for tag, icon, title, description in features
    )
    st.html(f'<div class="feature-grid">{cards}</div>')


def render_sidebar_brand(subtitle: str = "") -> None:
    st.html('<div class="sidebar-brand">'
        '<span class="sidebar-brand-icon">🧠</span>'
        "<div>"
        '<div class="sidebar-brand-name">BraTS Segmentation</div>'
        f'<div class="sidebar-brand-sub">{subtitle}</div>'
        "</div>"
        "</div>")


def render_section_label(text: str) -> None:
    st.html(f'<p class="sidebar-section-label">{text}</p>')


def render_metric_row(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>'
        for label, value in items
    )
    st.html(f'<div class="metric-grid">{cards}</div>')


def render_stat_grid(entries: list[tuple[str, str, str | None, str | None]]) -> None:
    """`entries` is a list of (label, value, color, caption). `color` tints
    the card's left edge (pass None for a neutral card); `caption` is an
    optional short line under the value (pass None to omit)."""
    cards = []
    for label, value, color, caption in entries:
        border_style = f' style="border-left-color:{color};"' if color else ""
        dot = f'<span class="legend-swatch" style="background:{color}"></span>' if color else ""
        caption_html = f'<div class="stat-caption">{caption}</div>' if caption else ""
        cards.append(
            f'<div class="stat-card"{border_style}>'
            f'<div class="stat-label">{dot}{label}</div>'
            f'<div class="stat-value">{value}</div>'
            f"{caption_html}"
            f"</div>"
        )
    st.html(f'<div class="stat-grid">{"".join(cards)}</div>')


def render_section_heading(text: str, meta: str | None = None) -> None:
    meta_html = f'<span class="meta">{meta}</span>' if meta else ""
    st.html(f'<div class="section-heading"><span>{text}</span>{meta_html}</div>')


def render_status_banner(kind: str, message_html: str) -> None:
    st.html(f'<div class="status-banner {kind}">{message_html}</div>')


def render_panel_label(text: str) -> None:
    st.html(f'<p class="panel-label"><span class="dot"></span>{text}</p>')


def render_legend() -> None:
    chips = "".join(
        f'<span class="legend-chip"><span class="legend-swatch" style="background:{color}"></span>'
        f'<span class="legend-code">{code}</span> {label}</span>'
        for code, label, color in CLASS_INFO.values()
    )
    st.html(f'<div class="legend-row">{chips}</div>')


def render_empty_state(icon: str, title: str, body: str) -> None:
    st.html(f'<div class="empty-state"><div class="icon">{icon}</div>'
        f"<h4>{title}</h4><p>{body}</p></div>")


def style_dark_figure(fig, ax) -> None:
    """Match a matplotlib figure's background to the app's dark surface
    color so slices/plots sit flush inside the viewer panels instead of
    showing a jarring white rectangle."""
    fig.patch.set_facecolor(COLORS["surface"])
    ax.set_facecolor(COLORS["surface"])


def style_colorbar(cbar) -> None:
    cbar.outline.set_edgecolor(COLORS["border_strong"])
    cbar.ax.yaxis.set_tick_params(color=COLORS["text_muted"], labelcolor=COLORS["text_muted"])