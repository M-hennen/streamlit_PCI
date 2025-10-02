import plotly.graph_objects as go

# Define classification thresholds and colors
SPEI_CLASSES = [
    (-2.0, -1.5, "Severe drought", "darkred"),
    (-1.5, -1.0, "Moderate drought", "#DE643C"),
    (-1.0,  0.0, "Mild drought", "orange"),
    (0.0,   1.0, "Near normal", "lightgreen"),
    (1.0,   1.5, "Abnormally wet", "green"),
    (1.5,   float("inf"), "Extremely wet", "darkgreen")
]

# Adjusted monthly rainfall severity for Wales (ERA5, mm)
RAINFALL_CLASSES = [
    (0, 20, "Extreme drought", "#8B4513"),   # dark brown
    (20, 40, "Severe drought", "#D2691E"),   # chocolate / lighter brown
    (40, 70, "Normal", "#F4A460"), # sandy brown
    (70, 120, "Near normal", "#87CEFA"),     # light sky blue
    (120, 160, "Wet", "#4682B4"),            # steel blue
    (160, float("inf"), "Extremely wet", "#00008B") # dark blue
]

NDVI_CLASSES = [
    (0.0, 0.2, "Very low vegetation", "#8B4513"),   # dark brown
    (0.2, 0.4, "Low vegetation", "#D2691E"),   # chocolate / lighter brown
    (0.4, 0.6, "Moderate vegetation", "#F4A460"), # sandy brown
    (0.6, 0.8, "High vegetation", "#32CD32"),     # lime green
    (0.8, 1.0, "Very high vegetation", "#006400")            # dark green
]

def classify_rainfall(mm):
    """Classify monthly rainfall (mm) into severity and water-like color."""
    for lower, upper, label, color in RAINFALL_CLASSES:
        if lower <= mm < upper:
            return label, color
    return "Unknown", "grey"

def classify_spei(val):
    """Classify a SPEI value into category and color."""
    for lower, upper, label, color in SPEI_CLASSES:
        if lower <= val < upper:
            return label, color
    # fallback for very low values
    return "Extreme drought", "darkred"

def classify_ndvi(val):
    """Classify an NDVI value into category and color."""
    for lower, upper, label, color in NDVI_CLASSES:
        if lower <= val < upper:
            return label, color
    return "Unknown", "grey"

def plot_spei_gauge(val, previous_val=None, title="SPEI-3"):
    """Create a Plotly gauge chart for a SPEI value."""
    category, bar_colour = classify_spei(val)

    fig = go.Figure(go.Indicator(
        domain={'x': [0, 0.9], 'y': [0, 0.9]},
        value=val,
        mode="gauge+number+delta",
        title={
            'text': f"<b>{title}</b><br><sub>{category}</sub>",
            'font': {
                'size': 24,       # bigger title
                'color': 'black', # default black instead of grey
                'family': "Arial, sans-serif"
            }
        },
        delta={'reference': previous_val} if previous_val is not None else None,
        gauge={
            'axis': {'range': [2, -2]},  # flipped axis
            'bar': {'color': bar_colour},
            'steps': [
                {'range': [2, 0],   'color': "#B6E388"},  # light green
                {'range': [0, -1],  'color': "#F7DC6F"},  # stronger yellow
                {'range': [-1, -1.5], 'color': "#F5B041"}, # orange
                {'range': [-1.5, -2], 'color': "#CD6155"}  # red-brown
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': previous_val if previous_val is not None else val
            }
        }
    ))
    return fig


def plot_rainfall_gauge(val, previous_val=None, title=None):
    """Create a Plotly gauge chart for a SPEI value."""
    category, bar_colour = classify_rainfall(val)

    fig = go.Figure(go.Indicator(
        domain={'x': [0, 0.9], 'y': [0, 0.9]},
        value=val,
        mode="gauge+number+delta",
        title={
            'text': f"<sub>{category}</sub>",#f"<b>{title}</b><br><sub>{category}</sub>",
            'font': {
                'size': 24,       # bigger title
                'color': 'black', # default black instead of grey
                'family': "Arial, sans-serif"
            }
        },
        delta={'reference': previous_val} if previous_val is not None else None,
        gauge={
            'axis': {'range': [0, 160]},  # flipped axis
            'bar': {'color': bar_colour},
            'steps': [
                {'range': [0, 20], 'color': "#D8B4A6"},    # extreme drought - pastel brown
                {'range': [20, 40], 'color': "#E6AC8B"},   # severe drought - lighter pastel brown
                {'range': [40, 70], 'color': "#F4CBA6"},   # moderate drought - pastel sandy
                {'range': [70, 120], 'color': "#A6D8F4"},  # near normal - pastel light blue
                {'range': [120, 160], 'color': "#7FB3D5"}, # wet - pastel steel blue
                {'range': [160, 200], 'color': "#5A85B1"}            # pastel steel blue
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': previous_val if previous_val is not None else val
            }
        }
    ))
    return fig


def plot_ndvi_gauge(val, previous_val=None, seasonal_val=None, title="NDVI"):
    """Create a Plotly gauge chart for an NDVI value."""
    category, bar_colour = classify_ndvi(val)

    fig = go.Figure(go.Indicator(
        domain={'x': [0, 0.9], 'y': [0, 0.9]},
        value=round(val, 2),
        mode="gauge+number+delta",
        title={
            'text': f"<b>{title}</b><br><sub>{category}</sub>",
            'font': {
                'size': 24,       # bigger title
                'color': 'black', # default black instead of grey
                'family': "Arial, sans-serif"
            }
        },
        delta={'reference': previous_val} if previous_val is not None else None,
        gauge={
            'axis': {'range': [-1, 1]},  # flipped axis
            'bar': {'color': bar_colour},
            'steps': [
                {'range': [-1.0, 0.0],  'color': "#D8B4A6"},  # bare soil / non-veg (pastel brown)
                {'range': [0.0, 0.2],   'color': "#F4E1A6"},  # barren / sparse (pastel yellow)
                {'range': [0.2, 0.4],   'color': "#D9F4A6"},  # low vegetation (pastel yellow-green)
                {'range': [0.4, 0.6],   'color': "#A6E6A6"},  # moderate vegetation (pastel green)
                {'range': [0.6, 0.8],   'color': "#7FD5A6"},  # healthy vegetation (pastel teal-green)
                {'range': [0.8, 1.0],   'color': "#5AB178"}   # very dense vegetation (pastel deep green)
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': seasonal_val if seasonal_val is not None else val
            }
        }
    ))
    return fig