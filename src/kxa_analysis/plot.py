import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from .utils_analysis import get_2d
import matplotlib.cm as cm


def plot_hist(df, column, title = None, xlabel = None, is_log = False):
    if title is None:
        title = column
    if xlabel is None:
        xlabel = column
    if is_log:
        is_log = "log"
    else:
        is_log = None
    fig = px.histogram(df, x=column, nbins=20)  # Adjust nbins as needed
    fig.update_layout(
        title = title,
        xaxis_title = xlabel,
        yaxis_title = "Count",
        yaxis_type = is_log  # Set y-axis to log scale
    )
    fig.show()

def generate_shades(base_rgb, n):
    """
    Generate n progressively darker shades from the base_rgb color.
    base_rgb should be in the form 'rgb(r, g, b)'
    """
    import re
    import numpy as np

    # Extract the RGB components
    r, g, b = map(int, re.findall(r'\d+', base_rgb))

    # Convert to numpy array for vector math
    base_color = np.array([r, g, b])

    # Slightly darken the color instead of lightening
    shades = []
    for i in range(n):
        factor = 1 - (i * 0.05)  # Each step darkens by 15%
        new_color = np.clip(base_color * factor, 0, 255).astype(int)
        shades.append(f"rgb({new_color[0]}, {new_color[1]}, {new_color[2]})")
    return shades


# Define the base colors
base_colors = {
    '1xKXA+SAFIT2_4h-F': 'rgb(100, 255, 100)',
    '1xKXA+SAFIT2_4h-M': 'rgb(0, 100, 0)',
    '1xKXA_4h-F': 'rgb(255, 50, 255)',
    'Ketamine_4h-F': 'rgb(255, 50, 255)',
    '1xKXA_4h-M': 'rgb(50, 255, 255)',
    'Ketamine_4h-M': 'rgb(50, 255, 255)',

    '1xSaline+SAFIT2_4h-F': 'rgb(255, 255, 100)',
    '1xSaline+SAFIT2_4h-M': 'rgb(150, 150, 0)',
    '1xSaline_4h-F': 'rgb(130, 130, 130)',
    'Control_4h-F': 'rgb(130, 130, 130)',
    '1xSaline_4h-M': 'rgb(20, 20, 20)',
    'Control_4h-M': 'rgb(20, 20, 20)',
    '1xKXA+FKBP5KO_4h-F': 'rgb(255, 150, 150)',  # Add color for this case
    '1xKXA+FKBP5KO_4h-M': 'rgb(150, 0, 0)',      # Add color for this case
    '1xSaline+FKBP5KO_4h-F': 'rgb(180, 180, 255)',  # Add color for this case
    '1xSaline+FKBP5KO_4h-M': 'rgb(80, 80, 150)',    # Add color for this case

}

# Prefixes for different shades
prefixes = ['L1-', 'L2_3-', 'L4-', 'L5_6-']

# Create the extended color dictionary
extended_colors = {
    f"{prefix}{key}": shade
    for key, base_rgb in base_colors.items()
    for prefix, shade in zip(prefixes, generate_shades(base_rgb, 4))
}

# Merge base and extended colors
merged_dict = dict(base_colors, **extended_colors)


def plot_2d(df, feature, title = None, conditions = ['Model', 'Sex'], 
            colors= merged_dict, name = None, extension = 'pdf', show = True,
            ax_labels = ['dim_1', 'dim_2']):

    # Extract feature values (each row is already [x, y])
    point_cloud = np.array(df[feature].tolist())[:,[0,1]]  # Convert list of lists to a NumPy array
    # Create DataFrame with dimensions and labels
    dim1 = ax_labels[0]
    dim2 = ax_labels[1]
    plot_frame = pd.DataFrame(point_cloud, columns=[dim1, dim2])

    # Create conditions column as a list
    conditions_list = df[conditions].apply(lambda x: '-'.join(x), axis=1).tolist()
    # Add labels (same length as df)
    plot_frame["Label"] = conditions_list
    if 'Layer' in conditions:
        plot_frame['Layer'] = plot_frame['Label'].str.extract(r'^(L1|L2_3|L4|L5_6)')

        layer_symbols = {
            'L1': 'square',
            'L2_3': 'circle',
            'L4': 'diamond',
            'L5_6': 'triangle-up'
        }

        plot_frame['Layer'] = plot_frame['Layer'].astype('category')


    # Plot using plotly.express
    # Adjust sizes: larger for 'interpolation'
    plot_frame['size'] = plot_frame['Label'].apply(lambda x: 10 if x == 'interpolation' else 8)
# Start figure
    fig = go.Figure()

    # Plot each group manually to fully control legend and shape
    for label in plot_frame['Label'].unique():
        subset = plot_frame[plot_frame['Label'] == label]

        # Skip empty subsets just in case
        if subset.empty:
            continue

        if 'Layer' in conditions:
            layer = subset['Layer'].iloc[0]
            symbol = layer_symbols.get(layer, 'circle')
        else:
            symbol = 'circle'

        color = colors.get(label, 'gray')

        fig.add_trace(go.Scattergl(
            x=subset[dim1],
            y=subset[dim2],
            mode='markers',
            marker=dict(
                size=subset['size'],
                color=color,
                symbol=symbol,
                line=dict(width=0.5, color='black')
            ),
            name=label,
            hoverinfo='text',
            text=label
        ))
    # Final layout
    fig.update_layout(
        title=title,
        width=800,
        height=800,
        showlegend=True,
        legend_title_text='Condition',
        xaxis_title=dim1,
        yaxis_title=dim2
    )
    # Calculate medians
    median_vectors = plot_frame.groupby('Label')[[dim1, dim2]].median().reset_index()
    median_vectors[dim1] = median_vectors[dim1].astype(float)
    median_vectors[dim2] = median_vectors[dim2].astype(float)
    
    if 'Layer' in conditions:
        # Extract Layer from Label for shape mapping
        median_vectors['Layer'] = median_vectors['Label'].str.extract(r'^(L1|L2_3|L4|L5_6)')
        median_vectors['symbol'] = median_vectors['Layer'].map(layer_symbols)

    # Filter valid colors
    valid_colors = {key: colors[key] for key in colors if any(key == label for label in median_vectors['Label'].values)}


    # Add median markers
    # Add median markers with hover labels
    fig.add_trace(
        go.Scattergl(
            x=median_vectors[dim1],
            y=median_vectors[dim2],
            mode='markers',
            marker=dict(
                size=14,
                color=median_vectors['Label'].map(valid_colors),
                symbol=median_vectors['symbol'],  # Apply the shape!
                line=dict(width=2, color='black')
            ),
            text=median_vectors['Label'],
            hoverinfo='text',
            name='Median',
            showlegend=True
        )
    )
    # Update layout
    fig.update_layout(
        showlegend=True,
        width=800,  
        height=800  
    )



    if name is not None:
            save_filepath = name
            os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
            if extension == 'pdf':
                fig.write_image(save_filepath + '2d.pdf', engine="kaleido", format = 'pdf')
            elif extension == 'html':
                fig.write_html(save_filepath + '2d.html')
    
    if show:
        fig.show()
    del fig  # Delete the figure

def plot_pi(pi, title = None, name = None, is_log=False, cmap=None, show=True, norm = None, scale = 'Loading Scale'):
    if len(pi.shape) == 1:  
        pi_2d = get_2d(pi)
    else:
        pi_2d = pi  # Convert to 2D if needed
    
    fig, ax = plt.subplots(figsize=(12, 6))

    vmin, vmax = pi_2d.min(), pi_2d.max()
    if cmap is None:
        # Handle different cases based on data range
        if vmin < 0 and vmax > 0:
            cmap = plt.cm.seismic  # Diverging colormap for both negative and positive values
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        elif vmin >= 0:
            cmap = plt.cm.Reds  # Positive values only
            if is_log:
                vmin = max(vmin, 1e-8)  # Avoid zero issues
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        else:
            cmap = plt.cm.Blues  # Negative values only
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot the heatmap
    cax1 = ax.imshow(pi_2d, cmap=cmap, origin='lower', norm=norm)
    ax.set_title(title)

    # Create a colorbar
    cbar = fig.colorbar(cax1, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Log Color Scale' if is_log else scale)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    if name is not None:
        save_dir = os.path.dirname(name)
        if save_dir:  # Only create directory if there is one
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(name + '.pdf', format='pdf')

    if show:
        fig.show()
    del fig  # Delete the figure

def plot_dist_matrix(mf_sorted, dist_matrix):
    # Assuming distance_matrix is your data and condition_ranges is defined as in your code
    condition_ranges = mf_sorted.groupby('Condition').apply(lambda g: (g.index.min(), g.index.max()))

    # Create the figure and axes
    fig, ax = plt.subplots()

    # Display the image with a logarithmic color scale
    cax = ax.imshow(dist_matrix)#, vmax=0.09)
    # Extract the ranges and labels
    x_ticks = [start for (start, _) in condition_ranges]
    y_ticks = [end for (_, end) in condition_ranges]
    x_labels = condition_ranges.index

    # Set the ticks and labels for x and y axes
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_yticklabels(x_labels)

    # Invert the y-axis to have 0,0 at the bottom-left
    ax.invert_yaxis()

    # Add color bar with logarithmic scale
    fig.colorbar(cax, ax=ax, orientation='vertical', label='Distance')

    plt.show()


# def convert_rgb_to_tuple(rgb_str):
#     """Convert 'rgb(r, g, b)' string to an (r, g, b) tuple normalized to [0, 1]."""
#     rgb_values = rgb_str.strip('rgb()').split(',')
#     return tuple([int(val)/255.0 for val in rgb_values])


# def plot_2d_plt(df, feature, title, conditions=['Model', 'Sex'], colors=merged_dict, ax=None):
#     # Create conditions column as a list
#     conditions_list = df[conditions].apply(lambda x: '-'.join(x), axis=1).tolist()
    
#     # Extract feature values (each row is already [x, y])
#     point_cloud = np.array(df[feature].tolist())[:, [0, 1]]  # Convert list of lists to a NumPy array
    
#     # Create DataFrame with dimensions and labels
#     dim1 = "dim_1"
#     dim2 = "dim_2"
#     plot_frame = pd.DataFrame(point_cloud, columns=[dim1, dim2])

#     # Add labels (same length as df)
#     plot_frame["Label"] = conditions_list

#     # Use provided axis, otherwise create a new figure and axis
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 8))

#     # Get unique labels (conditions)
#     unique_labels = plot_frame["Label"].unique()
    
#     # Convert the color values to valid RGB tuples or hex (for 'rgb(r, g, b)' format)
#     color_map = {label: convert_rgb_to_tuple(colors.get(label, 'rgb(128, 128, 128)')) for label in unique_labels}

#     # Plot the scatter plot with smaller points
#     for label in unique_labels:
#         label_data = plot_frame[plot_frame["Label"] == label]
#         ax.scatter(label_data[dim1], label_data[dim2], 
#                    label=label, 
#                    color=color_map[label], 
#                    edgecolors='black', 
#                    s=30,  # Smaller marker size
#                    alpha=0.7)  # Transparency for better visualization

#     # Customize the plot
#     ax.set_title(title)
#     ax.set_xlabel("PCA VAE X")
#     ax.set_ylabel("PCA VAE Y")
    
#     # Smaller legend, moved to a less intrusive position
#     ax.legend(loc='lower right', fontsize=8)  

#     # Optional: Set grid
#     ax.grid(True)

#     return ax  # Return the axis instead of showing the plot


def plot_vae_dist(mf, points, dist, vmin=None, vmax=None):
    """
    Plots VAE latent space with points colored by distance in PI space.

    Args:
        mf: DataFrame containing latent space and distance values.
        points: Column name in `mf` containing 2D coordinates (tuples or lists).
        dist: Column name in `mf` containing distance values.
        vmin: Minimum value for color normalization (default: min of dist).
        vmax: Maximum value for color normalization (default: max of dist).
    """
    x, y = zip(*mf[points])  # Unpack 2D points
    distances = mf[dist]

    # Set fixed colormap limits
    if vmin is None:
        vmin = distances.min()
    if vmax is None:
        vmax = distances.max()
        
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("magma")  # Use 'magma' colormap

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=distances, cmap=cmap, norm=norm, edgecolors='k', linewidth=0.5)

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Distance in PI space')

    # Labels and title
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.title('VAE Latent Space with PI Distance-based Coloring')

    plt.show()
# # from plot import plot_vae_dist_plt

# def plot_vae_dist_plt(ax, df, points, dist, vmin, vmax):
#     """
#     Plot a scatter plot on the given ax.

#     Parameters
#     ----------
#     ax : matplotlib.axes.Axes
#         The axis to plot on.
#     df : pandas.DataFrame
#         Dataframe that must contain:
#           - a column with coordinate pairs named by `points`
#           - a column with distances named by `dist`
#     points : str
#         Name of the column containing coordinate pairs (assumed to be (x,y)).
#     dist : str
#         Name of the column containing distances to color by.
#     vmin, vmax : float
#         Color limits for the scatter.
#     """
#     ax.clear()
#     # Extract x and y coordinates
#     x_vals = df[points].apply(lambda p: p[0])
#     y_vals = df[points].apply(lambda p: p[1])
#     sc = ax.scatter(x_vals, y_vals, c=df[dist], vmin=vmin, vmax=vmax, cmap='magma')
#     ax.set_title(f"Scatter: {dist}")
#     return sc
