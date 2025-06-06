import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

figsize_default = 7 # inches
element_scaling = 0.5


# Function for setting the default figure size
def figsize(rel_width, rel_height):
    return (rel_width * figsize_default, rel_height * figsize_default)


# Function for labelling figures with letters
def label_figure(fig, pos, label, size=13):
    if isinstance(pos, str):
        if pos == "top left":
            pos = (0, 1.01)
        elif pos == "top right":
            pos = (0.98, 1.01)
    
    if isinstance(pos, tuple):
        fig.text(*pos, label, size=size, weight="bold")

def label_axis(ax, pos, label, size=13, ha=None, va="top"):
    if isinstance(pos, str):
        if pos == "top left":
            pos = (0.05, 0.95)
            ha="left"
        elif pos == "top right":
            pos = (0.95, 0.95)
            ha="right"
    
    if isinstance(pos, tuple):
        ax.text(*pos, label, size=size, transform=ax.transAxes, ha=ha, va=va, weight="bold")

def label_axes(axes, pos, start_label_int=0, size=13, ha=None):
    for i,ax in enumerate(axes.flatten()):
        label_axis(ax, pos, chr(i+65+start_label_int), size=13, ha=ha)

# Get a factor of the original rcParam as value
def rcfac(rckwarg: str, factor: float):
    old = plt.rcParams.get(rckwarg)
    if isinstance(old, int):
        return int(np.round(old * factor))
    elif isinstance(old, float):
        return old * factor
    elif old is None:
        raise ValueError(f"not an rcParam: {rckwarg}")
    else:
        raise ValueError(f"unsupported type of rcParam: {type(old)}")

# Set the default parameters for plots

# Generally halve the size and width of fonts, lines, etc.
# plt.rcParams.update(
#     {k:v*element_scaling for k,v in plt.rcParams.items() if (k.endswith("size") or k.endswith("width") or k.endswith("pad")) and isinstance(v, (int, float))}
# )

plt.rcParams["figure.max_open_warning"] = 30
plt.rcParams['figure.dpi'] = 150  # 600 DPI

# Set the dpi for exporting
export_dpi = 300
export_formats = ["png", "pdf", "svg"]

# Set the default line colors

cmaps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']

# 4
default_cmap_index=4
default_cmap = plt.get_cmap(cmaps[4])
default_cmap_colors = default_cmap.colors
plt.rcParams['axes.prop_cycle'] = cycler('color', default_cmap_colors)


# Define the figures
thesis_figures = {}

# Create figure 1
thesis_figures["umap_targets_all"] = {"main": plt.figure(figsize=figsize(0.8, 1))}

# Create figure 2
thesis_figures["umap_targets_ojip"] = {"main": plt.figure(figsize=figsize(0.8, 1))}

# Create figure 3
thesis_figures["ojip_doublenorm"] = {"main": plt.figure(figsize=figsize(1, 1))}

# Create figure 4
thesis_figures["dense_loss"] = {"main": plt.figure(figsize=figsize(1, 0.8))}

# Create figure 4
thesis_figures["dense_loss_supp"] = {"main": plt.figure(figsize=figsize(1, 0.8))}

# Create figure 5
thesis_figures["dense_metrics"] = {"main": plt.figure(figsize=figsize(1, 1.2))}

# Create figure 4
thesis_figures["lstm_loss"] = {"main": plt.figure(figsize=figsize(1, 0.8))}

# Create figure 4
thesis_figures["lstm_loss_supp"] = {"main": plt.figure(figsize=figsize(1, 0.8))}

# Create figure 5
thesis_figures["lstm_metrics"] = {"main": plt.figure(figsize=figsize(1, 1.2))}

# Create figure 5
thesis_figures["lstm_metrics_over_time"] = {"main": plt.figure(figsize=figsize(1, 1.2))}