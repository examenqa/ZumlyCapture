"""Zumly design tokens layered on compact Fluent 2 component metrics.

Centralized design constants aligned with Windows 11 / Fluent 2.
All spacing uses a 4px base grid.  Corner radii use 4px (controls) / 8px
(containers).  Colors follow Fluent 2 neutral palette (grey ramp) and
semantic color roles.  Elevation system uses Fluent 2 shadow spec.

Reference:
- https://fluent2.microsoft.design/color
- https://fluent2.microsoft.design/color-tokens/
- https://fluent2.microsoft.design/elevation
"""

# ── Spacing (Fluent 2 Spacer Tokens) ──────────────────────────────────
# Based on https://fluent2.microsoft.design/layout
# Uses 2px base unit with Fluent 2 spacer values (size20, size40, size80, etc.)
SPACE_XXS: int = 2   # size20 — tight compact spacing
SPACE_XS: int = 4    # size40 — minimal padding
SPACE_SM: int = 8    # size80 — small controls, icons
SPACE_MD: int = 12   # size120 — default control padding
SPACE_LG: int = 16   # size160 — section spacing
SPACE_XL: int = 24   # size240 — panel spacing
SPACE_XXL: int = 32  # size320 — large gaps, hero sections
SPACE_XXXL: int = 48 # size480 — major layout divisions

# Additional Fluent 2 spacer tokens for fine control
SPACE_6: int = 6     # size60 — icon-text gap
SPACE_10: int = 10   # size100 — list item padding
SPACE_20: int = 20   # size200 — medium sections
SPACE_40: int = 40   # size400 — large containers
SPACE_64: int = 64   # size640 — extra-large divisions

# ── Corner Radius (Fluent 2 Shapes) ───────────────────────────────────
# Based on https://fluent2.microsoft.design/shapes
# Global-Corner-Radius tokens
RADIUS_NONE: int = 0      # Navigation bars, tabs, edge-aligned elements
RADIUS_SMALL: int = 4     # Global-Corner-Radius-40 — buttons, inputs, small controls
RADIUS_MEDIUM: int = 8    # Global-Corner-Radius-80 — large buttons, cards
RADIUS_LARGE: int = 12    # Global-Corner-Radius-120 — sheets, popovers, large surfaces
RADIUS_XLARGE: int = 16   # Global-Corner-Radius-160 — extra-large containers
RADIUS_CIRCULAR: int = 9999  # Global-Corner-Radius-Circular — avatars, status dots (50% effective)

# ── Zumly brand palette ────────────────────────────────────────────────
BRAND_NAVY: str = "#0F1738"
BRAND_CYAN: str = "#08AFC0"
BRAND_CYAN_DARK: str = "#079EB0"
BRAND_PURPLE: str = "#6D2BD6"
BRAND_VIOLET: str = "#6940DA"
WHITE: str = "#FFFFFF"

# ── Brand-derived dark surfaces ────────────────────────────────────────

# Background layers (darkest → lightest)
# Aligned with Fluent 2 colorNeutralBackground hierarchy
BG_SOLID: str = "#050816"            # deepest media/canvas well
BG_LAYER_1: str = "#080D1C"          # application shell
BG_LAYER_2: str = "#10172A"          # panels and timeline tracks
BG_LAYER_3: str = "#172039"          # controls and raised surfaces
BG_LAYER_4: str = "#1E2947"          # hover/elevated surfaces
BG_LAYER_5: str = "#253252"          # menus and tooltips

# Interactive surface states
BG_SUBTLE: str = "transparent"       # colorSubtleBackground rest (hover: grey[22])
BG_SUBTLE_HOVER: str = "#1A2440"
BG_SUBTLE_PRESSED: str = "#111A30"
BG_SUBTLE_SELECTED: str = "#172C46"

# Card surfaces (from colorNeutralCardBackground)
BG_CARD: str = "#172039"
BG_CARD_HOVER: str = "#1E2947"
BG_CARD_PRESSED: str = "#111A30"
BG_CARD_SELECTED: str = "#172C46"

# ── Fluent 2 Foreground (Text & Icons) ─────────────────────────────────
FG_1: str = WHITE
FG_2: str = "#B4BED2"
FG_3: str = "#8E9AB4"
FG_4: str = "#7F8CA8"
FG_DISABLED: str = "#55627D"
FG_INVERTED: str = BRAND_NAVY

# ── Fluent 2 Borders / Strokes ─────────────────────────────────────────
STROKE_ACCESSIBLE: str = "#526789"
STROKE_1: str = "#3D5277"
STROKE_2: str = "#293653"
STROKE_SUBTLE: str = "#18243C"

# ── Brand / Accent ─────────────────────────────────────────────────────
# Purple is the primary command/synthetic-effect color. Cyan is the focus,
# active-tool, progress, and interaction-emphasis color.
BRAND: str = BRAND_PURPLE
BRAND_HOVER: str = "#7C3BE0"
BRAND_ACTIVE: str = "#5720B8"
BRAND_TRANSLUCENT: str = "rgba(109, 43, 214, 0.18)"
BRAND_TRANSLUCENT_HOVER: str = "rgba(109, 43, 214, 0.26)"
BRAND_TRANSLUCENT_STRONG: str = "rgba(109, 43, 214, 0.34)"
BRAND_DISABLED: str = "#3D2A69"

ACCENT: str = BRAND_CYAN
ACCENT_HOVER: str = "#16C4D5"
ACCENT_ACTIVE: str = BRAND_CYAN_DARK
ACCENT_TRANSLUCENT: str = "rgba(8, 175, 192, 0.16)"
ACCENT_TRANSLUCENT_HOVER: str = "rgba(8, 175, 192, 0.24)"
ACCENT_TRANSLUCENT_STRONG: str = "rgba(8, 175, 192, 0.32)"

# ── Fluent 2 Semantic Status Colors ───────────────────────────────────
# Success (green ramp) — colorStatusSuccess*
SUCCESS: str = "#10b981"             # green primary (Fluent 2 spec)
SUCCESS_BG_1: str = "#0c5239"        # green shade40 — success background
SUCCESS_BG_2: str = "#0f6b4a"        # green shade30 — success content bg
SUCCESS_FG: str = "#6ee7b7"          # green tint30 — success foreground on dark

# Danger (cranberry/red ramp) — colorStatusDanger*
DANGER: str = "#ef4444"              # cranberry primary (Fluent 2 spec)
DANGER_HOVER: str = "#f87171"        # tint variation
DANGER_DARK: str = "#dc2626"         # record button background (shade10)
DANGER_BG_1: str = "#5c1010"         # cranberry shade40 — danger background
DANGER_BG_2: str = "#7a1818"         # cranberry shade30 — danger content bg
DANGER_FG: str = "#fca5a5"           # cranberry tint30 — danger foreground on dark
DANGER_TRANSLUCENT: str = "rgba(239, 68, 68, 0.12)"
DANGER_TRANSLUCENT_STRONG: str = "rgba(239, 68, 68, 0.15)"

# Warning (orange ramp) — colorStatusWarning*
WARNING: str = "#f59e0b"             # orange primary (Fluent 2 spec)
WARNING_HOVER: str = "#fbbf24"       # tint variation
WARNING_BG_1: str = "#5c3a10"        # orange shade40 — warning background
WARNING_BG_2: str = "#7a4d18"        # orange shade30 — warning content bg
WARNING_FG: str = "#fed7aa"          # orange tint30 — warning foreground on dark

# Info (blue ramp) — colorStatusInfo* (using generic blue palette)
INFO: str = "#3b82f6"                # blue primary
INFO_HOVER: str = "#60a5fa"          # blue tint
INFO_BG_1: str = "#18395c"           # blue shade40 — info background
INFO_BG_2: str = "#204d7a"           # blue shade30 — info content bg
INFO_FG: str = "#93c5fd"             # blue tint30 — info foreground on dark

# ── Special surfaces ──────────────────────────────────────────────────
CLOSE_HOVER_BG: str = "#c42b1c"       # Windows-standard close-button red
DISCARD_BG: str = "#474747"           # grey[28] — discard button base
DISCARD_HOVER_BG: str = "#525252"     # grey[32] — discard hover
DIALOG_BG: str = BG_LAYER_2
OVERLAY_BG: str = "rgba(16, 23, 42, 0.94)"

# ── Typography (Fluent 2 Type Ramp) ───────────────────────────────────
# Based on https://fluent2.microsoft.design/typography
# Font family: Segoe UI Variable with fallback to Segoe UI
# Windows 10 may not have Segoe UI Variable — fallback ensures compatibility
FONT_FAMILY: str = '"Segoe UI", "San Francisco", -apple-system, BlinkMacSystemFont, sans-serif'
FONT_FAMILY_MONO: str = '"Consolas", "Courier New", monospace'

# Font Weights (Fluent 2 conventions)
FONT_WEIGHT_REGULAR: int = 400   # fontWeightRegular — default body text
FONT_WEIGHT_MEDIUM: int = 500    # fontWeightMedium — slight emphasis
FONT_WEIGHT_SEMIBOLD: int = 600  # fontWeightSemibold — headings, buttons
FONT_WEIGHT_BOLD: int = 700      # fontWeightBold — strong emphasis, display

# Type Ramp (font size / line height)
# Caption 2: 10px / 14px (Regular 400)
FONT_SIZE_CAPTION_2: int = 10
FONT_LINE_HEIGHT_CAPTION_2: int = 14

# Caption 1: 12px / 16px (Regular 400 or Semibold 600)
FONT_SIZE_CAPTION_1: int = 12
FONT_LINE_HEIGHT_CAPTION_1: int = 16

# Body 1: 14px / 20px (Regular 400 or Semibold 600) — default UI text
FONT_SIZE_BODY_1: int = 14
FONT_LINE_HEIGHT_BODY_1: int = 20

# Body 2: 16px / 22px (Regular 400)
FONT_SIZE_BODY_2: int = 16
FONT_LINE_HEIGHT_BODY_2: int = 22

# Subtitle 2: 20px / 28px (Semibold 600)
FONT_SIZE_SUBTITLE_2: int = 20
FONT_LINE_HEIGHT_SUBTITLE_2: int = 28

# Subtitle 1: 24px / 32px (Semibold 600)
FONT_SIZE_SUBTITLE_1: int = 24
FONT_LINE_HEIGHT_SUBTITLE_1: int = 32

# Title 3: 28px / 36px (Semibold 600)
FONT_SIZE_TITLE_3: int = 28
FONT_LINE_HEIGHT_TITLE_3: int = 36

# Title 2: 32px / 40px (Semibold 600)
FONT_SIZE_TITLE_2: int = 32
FONT_LINE_HEIGHT_TITLE_2: int = 40

# Title 1: 40px / 52px (Semibold 600)
FONT_SIZE_TITLE_1: int = 40
FONT_LINE_HEIGHT_TITLE_1: int = 52

# Display: 68px / 92px (Bold 700)
FONT_SIZE_DISPLAY: int = 68
FONT_LINE_HEIGHT_DISPLAY: int = 92

# Legacy aliases — bumped to improve readability on modern displays
# Use new Fluent 2 tokens (FONT_SIZE_CAPTION_1, FONT_SIZE_BODY_1, etc.) for new code
FONT_SIZE_CAPTION: int = 12    # bumped from 11
FONT_SIZE_BODY: int = 14       # bumped from 13
FONT_SIZE_SUBTITLE: int = 16   # bumped from 15
FONT_SIZE_TITLE: int = 18      # bumped from 16
FONT_SIZE_HEADER: int = 22     # bumped from 20

# ── Animation (Fluent 2 Motion Tokens) ────────────────────────────────
# Based on https://fluent2.microsoft.design/motion
# Duration tokens
DURATION_ULTRA_FAST: int = 50   # durationUltraFast — focus rings, micro feedback
DURATION_FASTER: int = 100      # durationFaster — icon state swaps, badge updates
DURATION_FAST: int = 150        # durationFast — button press, toggle, checkbox
DURATION_NORMAL: int = 200      # durationNormal — default hover/focus/state transitions
DURATION_GENTLE: int = 250      # durationGentle — content revealing, tooltip appear
DURATION_SLOW: int = 300        # durationSlow — drawer open, panel slide-in
DURATION_SLOWER: int = 400      # durationSlower — complex orchestrated sequences
DURATION_ULTRA_SLOW: int = 500  # durationUltraSlow — full-page transitions, hero reveals

# Easing curves (CSS cubic-bezier values for reference — Qt uses QEasingCurve enums)
# Qt mappings:
#   curveEasyEase → QEasingCurve.Type.OutCubic (default, smooth)
#   curveDecelerate → QEasingCurve.Type.OutQuad (entering elements)
#   curveAccelerate → QEasingCurve.Type.InQuad (exiting elements)
#   curveLinear → QEasingCurve.Type.Linear (loops, progress)
CURVE_EASY_EASE: str = "cubic-bezier(0.33, 0, 0.67, 1)"    # Elements moving within viewport
CURVE_EASY_EASE_MAX: str = "cubic-bezier(0.8, 0, 0.2, 1)"  # Emphasis, dramatic repositioning
CURVE_ACCELERATE: str = "cubic-bezier(0.9, 0.1, 1, 0.2)"   # Exiting screen (ease in, fast out)
CURVE_DECELERATE: str = "cubic-bezier(0.1, 0.9, 0.2, 1)"   # Entering screen (ease out, slow in)
CURVE_LINEAR: str = "linear"                                # Continuous repetitive motion

# ── Fluent 2 Elevation / Shadow System ─────────────────────────────────
# Based on official Fluent 2 elevation spec (dark theme)
# Reference: https://fluent2.microsoft.design/elevation
# Fluent shadows combine key + ambient shadows for depth perception

# Shadow Layer 0 — No elevation (flat surfaces)
SHADOW_LAYER_0_BLUR: int = 0
SHADOW_LAYER_0_OFFSET_Y: int = 0
SHADOW_LAYER_0_KEY: str = "rgba(0, 0, 0, 0)"
SHADOW_LAYER_0_AMBIENT: str = "rgba(0, 0, 0, 0)"

# Shadow Layer 1 (Shadow2) — Minimal depth (buttons at rest, subtle cards)
SHADOW_LAYER_1_BLUR: int = 2
SHADOW_LAYER_1_OFFSET_Y: int = 1
SHADOW_LAYER_1_KEY: str = "rgba(0, 0, 0, 0.28)"        # dark theme: 28% key
SHADOW_LAYER_1_AMBIENT: str = "rgba(0, 0, 0, 0.24)"    # dark theme: 24% ambient

# Shadow Layer 2 (Shadow4) — Cards, list items
SHADOW_LAYER_2_BLUR: int = 4
SHADOW_LAYER_2_OFFSET_Y: int = 2
SHADOW_LAYER_2_KEY: str = "rgba(0, 0, 0, 0.28)"
# NOTE: Qt's QGraphicsDropShadowEffect only supports a single shadow. The ambient
# shadow tokens below are defined for Fluent 2 spec reference but are not used.
# Future custom renderers (QPainter compositing, WebView overlays) may use both.
SHADOW_LAYER_2_AMBIENT: str = "rgba(0, 0, 0, 0.24)"

# Shadow Layer 3 (Shadow8) — Command bars, tooltips, dropdowns
SHADOW_LAYER_3_BLUR: int = 8
SHADOW_LAYER_3_OFFSET_Y: int = 4
SHADOW_LAYER_3_KEY: str = "rgba(0, 0, 0, 0.28)"
SHADOW_LAYER_3_AMBIENT: str = "rgba(0, 0, 0, 0.24)"  # unused (see SHADOW_LAYER_2_AMBIENT note)

# Shadow Layer 4 (Shadow16) — Dialogs, callouts, flyouts
SHADOW_LAYER_4_BLUR: int = 16
SHADOW_LAYER_4_OFFSET_Y: int = 8
SHADOW_LAYER_4_KEY: str = "rgba(0, 0, 0, 0.28)"
SHADOW_LAYER_4_AMBIENT: str = "rgba(0, 0, 0, 0.24)"  # unused (see SHADOW_LAYER_2_AMBIENT note)

# Material effects — subtle transparency for overlays (acrylic-inspired)
MATERIAL_OVERLAY_ALPHA: float = 0.92  # backdrop opacity
MATERIAL_CARD_ALPHA: float = 0.98     # card/dialog opacity

# ── Focus ring ─────────────────────────────────────────────────────────
FOCUS_RING_WIDTH: int = 2
FOCUS_RING_OFFSET: int = 2

# ── Scrollbar ──────────────────────────────────────────────────────────
SCROLLBAR_THIN: int = 6         # default narrow width
SCROLLBAR_WIDE: int = 12        # expanded width on hover
SCROLLBAR_MIN_HEIGHT: int = 24  # minimum handle length

# ── Legacy compatibility aliases ───────────────────────────────────────
# Keep backward compat with existing QSS selectors during transition
BG_CANVAS = BG_LAYER_1          # maps to grey[4]
BG_PANEL = BG_LAYER_2           # maps to grey[8]
BG_SURFACE = BG_LAYER_3         # maps to grey[12]
BG_ELEVATED = BG_LAYER_4        # maps to grey[16]
BG_INTERACTIVE = BG_LAYER_4     # maps to grey[16] (interactive defaults to elevated)
BG_HOVER = BG_CARD_HOVER        # maps to grey[24]
BG_HOVER_STRONG = BG_LAYER_5

BORDER_SUBTLE = STROKE_SUBTLE   # maps to grey[4]
BORDER_MEDIUM = STROKE_2        # maps to grey[32]
BORDER_STRONG = STROKE_1        # maps to grey[40]

FG_PRIMARY = FG_1               # white
FG_SECONDARY = FG_2             # grey[84]
FG_MUTED = FG_3                 # grey[68]
FG_DIM = FG_4                   # grey[60]
FG_WHITE = FG_1                 # white

CARD_BG = BG_CARD               # grey[20]
CARD_BORDER = STROKE_1
CARD_HOVER_BG = BG_CARD_HOVER   # grey[24]

DANGER_LIGHT = DANGER_FG        # cranberry tint30
DANGER_TEXT = DANGER            # cranberry primary

# Legacy shadow tokens (map to Fluent 2 Layer 2 and 3)
SHADOW_SUBTLE_BLUR = SHADOW_LAYER_2_BLUR
SHADOW_SUBTLE_OFFSET = SHADOW_LAYER_2_OFFSET_Y
SHADOW_SUBTLE_COLOR = SHADOW_LAYER_2_KEY
SHADOW_MEDIUM_BLUR = SHADOW_LAYER_3_BLUR
SHADOW_MEDIUM_OFFSET = SHADOW_LAYER_3_OFFSET_Y
SHADOW_MEDIUM_COLOR = SHADOW_LAYER_3_KEY

# ── Fluent 2 Light Theme Colors ───────────────────────────────────────────
# Official Fluent 2 light mode palette
# Reference: https://fluent2.microsoft.design/color
LIGHT_BG_1: str = "#ffffff"        # colorNeutralBackground1 — canvas
LIGHT_BG_2: str = "#fafafa"        # colorNeutralBackground2
LIGHT_BG_3: str = "#f5f5f5"        # colorNeutralBackground3
LIGHT_BG_4: str = "#f0f0f0"        # colorNeutralBackground4
LIGHT_BG_5: str = "#ebebeb"        # colorNeutralBackground5
LIGHT_FG_1: str = "#242424"        # colorNeutralForeground1 — primary text
LIGHT_FG_2: str = "#424242"        # colorNeutralForeground2 — secondary text
LIGHT_FG_3: str = "#616161"        # colorNeutralForeground3 — tertiary text
LIGHT_FG_4: str = "#707070"        # colorNeutralForeground4 — quaternary
LIGHT_STROKE_1: str = "#d1d1d1"    # colorNeutralStroke1 — default borders
LIGHT_STROKE_2: str = "#e0e0e0"    # colorNeutralStroke2 — dividers
LIGHT_STROKE_ACCESSIBLE: str = "#616161"  # colorNeutralStrokeAccessible (3:1 contrast)
# Brand tokens — keep the original zumly purple, adjusted for light backgrounds
LIGHT_BRAND_FG: str = "#5720B8"
LIGHT_BRAND_BG: str = BRAND_PURPLE
LIGHT_BRAND_BG_HOVER: str = "#5E24BE"
LIGHT_BRAND_BG_PRESSED: str = "#4D1B9E"

# Light theme subtle interactive states
LIGHT_BG_SUBTLE: str = "transparent"       # colorSubtleBackground rest (same as dark)
LIGHT_BG_SUBTLE_HOVER: str = "#f5f5f5"     # subtle hover
LIGHT_BG_SUBTLE_PRESSED: str = "#ebebeb"   # subtle pressed
LIGHT_BG_CARD_HOVER: str = "#f0f0f0"       # card hover

# Light theme brand translucent surfaces (Issue #112)
LIGHT_BRAND_TRANSLUCENT: str = "rgba(109, 43, 214, 0.10)"
LIGHT_BRAND_TRANSLUCENT_HOVER: str = "rgba(109, 43, 214, 0.15)"
LIGHT_BRAND_TRANSLUCENT_STRONG: str = "rgba(109, 43, 214, 0.20)"

# Light theme focus ring
LIGHT_FOCUS_RING: str = BRAND_CYAN_DARK


# ── Theme-Aware Color Getters ─────────────────────────────────────────────
# Use these helpers in custom-painted widgets that need to respond to theme changes.

def bg_canvas(dark: bool = True) -> str:
    """Timeline/control background."""
    return BG_LAYER_1 if dark else LIGHT_BG_2


def bg_track(dark: bool = True) -> str:
    """Track background inside timeline."""
    return BG_LAYER_2 if dark else LIGHT_BG_3


def bg_track_border(dark: bool = True) -> str:
    """Track border stroke."""
    return STROKE_2 if dark else LIGHT_STROKE_2


def fg_primary(dark: bool = True) -> str:
    """Primary foreground (text, icons)."""
    return FG_1 if dark else LIGHT_FG_1


def fg_muted(dark: bool = True) -> str:
    """Muted/secondary foreground."""
    return FG_3 if dark else LIGHT_FG_3


def fg_dim(dark: bool = True) -> str:
    """Dim foreground (track labels, tick marks)."""
    return FG_4 if dark else LIGHT_FG_4


def dialog_stylesheet(dark: bool = True) -> str:
    """Return QMessageBox stylesheet for the current theme.
    
    Use this for all modal dialogs to ensure consistent theming.
    """
    if dark:
        return f"""
            QMessageBox {{
                background-color: {DIALOG_BG};
                color: {FG_PRIMARY};
            }}
            QMessageBox QLabel {{
                color: {FG_PRIMARY};
                font-size: 13px;
            }}
            QPushButton {{
                height: 32px;
                min-width: 80px;
                padding: 0 18px;
                border-radius: 6px;
                border: 1px solid {STROKE_1};
                background-color: {BG_LAYER_3};
                color: {FG_PRIMARY};
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {BG_LAYER_4};
                border-color: {ACCENT};
            }}
            QPushButton:default {{
                background-color: {BRAND};
                border: none;
                color: {WHITE};
                font-weight: 600;
            }}
            QPushButton:default:hover {{
                background-color: {BRAND_HOVER};
            }}
        """
    else:
        return f"""
            QMessageBox {{
                background-color: {LIGHT_BG_1};
                color: {LIGHT_FG_1};
            }}
            QMessageBox QLabel {{
                color: {LIGHT_FG_1};
                font-size: 13px;
            }}
            QPushButton {{
                height: 32px;
                min-width: 80px;
                padding: 0 18px;
                border-radius: 6px;
                border: 1px solid {LIGHT_STROKE_1};
                background-color: {LIGHT_BG_3};
                color: {LIGHT_FG_1};
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {LIGHT_BG_4};
                border-color: {LIGHT_STROKE_ACCESSIBLE};
            }}
            QPushButton:default {{
                background-color: {BRAND};
                border: none;
                color: white;
                font-weight: 600;
            }}
            QPushButton:default:hover {{
                background-color: {LIGHT_BRAND_BG_HOVER};
            }}
        """

# Compatibility names retained for the editor shell and older widgets.
SURFACE_BASE = BG_SOLID
SURFACE_ELEVATED = BG_LAYER_2
BRAND_ACCENT = ACCENT
TEXT_PRIMARY = FG_PRIMARY
TEXT_MUTED = FG_MUTED
DIVIDER = STROKE_2
STATE_RECORDING = DANGER
STATE_SUCCESS = SUCCESS

# Timeline semantics. These are visual-only and never participate in geometry.
TIMELINE_ZOOM = BRAND_CYAN
TIMELINE_ZOOM_AUTO = "#67E8F9"
TIMELINE_HIGHLIGHT = "#FBBF24"
TIMELINE_SYNTHETIC = BRAND_PURPLE
TIMELINE_SYNTHETIC_SELECTED = "#A78BFA"
TIMELINE_TRANSITION = "#14B8A6"
TIMELINE_TRANSITION_SELECTED = "#5EEAD4"
TIMELINE_LAYOUT = "#6366F1"
TIMELINE_LAYOUT_SELECTED = "#A5B4FC"
TIMELINE_MASK = "#F59E0B"
TIMELINE_ANNOTATION = "#EC4899"
TIMELINE_DRAWING = "#22C55E"
TIMELINE_TEXT = "#3B82F6"
TIMELINE_SELECTED_OUTLINE = WHITE
TIMELINE_SNAP = WARNING_HOVER
TIMELINE_TRACK_LABEL = FG_MUTED
TIMELINE_BLOCK_TEXT = FG_PRIMARY
TIMELINE_BLOCK_TEXT_MUTED = FG_SECONDARY
TIMELINE_RANGE = BRAND_VIOLET
TIMELINE_RANGE_SELECTED = TIMELINE_SYNTHETIC_SELECTED
TIMELINE_RANGE_TEXT = "#F5F3FF"
TIMELINE_VIDEO = TIMELINE_TRANSITION
TIMELINE_VIDEO_SELECTED = TIMELINE_TRANSITION_SELECTED
TIMELINE_VIDEO_EDGE = "#0F766E"
TIMELINE_VIDEO_TEXT = "#ECFEFF"
