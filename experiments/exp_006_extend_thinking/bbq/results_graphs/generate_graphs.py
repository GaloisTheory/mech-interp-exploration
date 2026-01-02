#!/usr/bin/env python3
"""Generate BBQ graphs for MATS write-up.

Usage:
    python generate_graphs.py
    
Or run cells interactively with #%% markers in VS Code/Cursor.
"""

#%% Imports
from graph_utils import plot_blank_static_accuracy, plot_summary_table

#%% 8B Model - All Categories
#plot_blank_static_accuracy(model='32B', save=True)
plot_summary_table(model='32B', save=True)

#%% 8B Model - Age & Appearance Only
plot_blank_static_accuracy(model='32B', save=True)
#plot_summary_table(model='8B', categories=['age', 'appearance'], save=True)

#%% 32B Model - All Categories
plot_blank_static_accuracy(model='32B', save=True)
plot_summary_table(model='32B', save=True)

# %%
from graph_utils import plot_blank_static_accuracy, plot_incorrect_static_accuracy, plot_summary_table

# Incorrect answer graphs
plot_incorrect_static_accuracy(model='8B', save=True)
plot_incorrect_static_accuracy(model='8B', categories=['age', 'appearance'], save=True)
plot_incorrect_static_accuracy(model='32B', save=True)
# %%
