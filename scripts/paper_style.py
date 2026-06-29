"""Canonical color palette and style for GORGO paper figures.

All paper figure scripts should import from here to ensure consistency.
"""

# GORGO variants — dark-to-light navy gradient
POLICY_COLORS = {
    "gorgo-hillclimb": "#1b3a5c",
    "gorgo-hillclimb-p95": "#1b3a5c",
    "gorgo-static": "#2d6a9f",
    "gorgo-static-p95": "#2d6a9f",
    "gorgo-autotune": "#4a86c7",
}

# Baselines
BASELINE_COLOR = "#b0c4d8"

# Winner highlight
WINNER_OUTLINE = "#e8c840"
WINNER_LW = 2.5

# RTT regions — same blue gradient, darkest = farthest
REGION_COLORS = {
    "Seoul": "#1b3a5c",
    "Frankfurt": "#4a86c7",
    "Ashburn": "#a8cce8",
}

# Dataset comparison
DATASET_COLORS = {
    "GLM-5.1": "#1b3a5c",
    "WildChat-4.8M": "#6a9fd8",
    "LMSYS-Chat-1M": "#a8cce8",
}

GORGO_POLICIES = {
    "gorgo-hillclimb",
    "gorgo-hillclimb-p95",
    "gorgo-static",
    "gorgo-static-p95",
    "gorgo-autotune",
}

POLICY_DISPLAY = {
    "gorgo-hillclimb": "gorgo (online)",
    "gorgo-hillclimb-p95": "gorgo (online)",
    "gorgo-static": "gorgo (fixed)",
    "gorgo-static-p95": "gorgo (fixed)",
    "simple-session-affinity": "simple-session-affinity",
    "least-request": "least-request",
    "least-load": "least-load",
    "prefix-cache": "prefix-cache",
    "random": "random",
}


def get_color(label: str) -> str:
    return POLICY_COLORS.get(
        label, POLICY_COLORS.get(POLICY_DISPLAY.get(label, ""), BASELINE_COLOR)
    )


def is_gorgo(label: str) -> bool:
    return label in GORGO_POLICIES or any(g in label for g in ("gorgo",))


def display_name(label: str) -> str:
    return POLICY_DISPLAY.get(label, label)


def apply_paper_style(ax):
    """Remove top/right spines and lighten grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.5)


def classify_region(median_rtt_ms: float) -> str:
    if median_rtt_ms < 100:
        return "Ashburn"
    elif median_rtt_ms < 500:
        return "Frankfurt"
    return "Seoul"
