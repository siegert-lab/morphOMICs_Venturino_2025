import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from morphomics.io.io import save_obj, load_obj
from utils import inverse_function, mask_pi, get_base
import torch as th
from morphomics.nn_models import train_test

def plot_vae_dist_plt(ax, df, points, dist, vmin, vmax):
    """
    Plot a scatter plot on the given axis.
    
    Parameters:
      ax: Matplotlib Axes to draw on.
      df: DataFrame that contains:
            - a column named by `points` with 2D coordinate pairs.
            - a column named by `dist` with scalar values for coloring.
      points: Column name with 2D coordinates.
      dist: Column name with distance values.
      vmin, vmax: Color normalization limits.
      
    Returns:
      The PathCollection from scatter.
    """
    ax.clear()
    # Extract x and y coordinates
    x_vals = df[points].apply(lambda p: p[0])
    y_vals = df[points].apply(lambda p: p[1])
    sc = ax.scatter(x_vals, y_vals, c=df[dist], vmin=vmin, vmax=vmax, cmap='magma')
    ax.set_title(f"Scatter: {dist}")
    return sc

def plot_heatmap(ax, Z, extent, vmin, vmax):
    """
    Plot a heatmap on the given axis.
    
    Parameters:
      ax: Matplotlib Axes to draw on.
      Z: 2D numpy array with scalar values.
      extent: List [x_min, x_max, y_min, y_max] for the imshow extent.
      vmin, vmax: Color normalization limits.
      
    Returns:
      The AxesImage from imshow.
    """
    ax.clear()
    hm = ax.imshow(Z, extent=extent, origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    ax.set_title("Heatmap: Distance")
    return hm

# Load your trained VAE pipeline object
path = 'results/vae/trained_vae'
my_pip = load_obj(path)
vae_pip = my_pip.metadata

pixes_tokeep = vae_pip['pixes_tokeep']
standardizer = vae_pip['standardizer']
pca, vae = vae_pip['fitted_pca_vae']

# Get the morphoframe and reset its index.
mf = my_pip.morphoframe['v1_pi']
mf = mf.reset_index()  # The old index becomes a column.
mf.rename(columns={'index': 'old_idcs'}, inplace=True)

# ------------------------------------------------------------
# STEP 1. Create a grid over the latent space.
# ------------------------------------------------------------
# Here we assume that mf has a column 'pca_vae' with 2D coordinates.
x_min = mf['pca_vae'].apply(lambda p: p[0]).min()
x_max = mf['pca_vae'].apply(lambda p: p[0]).max()
y_min = mf['pca_vae'].apply(lambda p: p[1]).min()
y_max = mf['pca_vae'].apply(lambda p: p[1]).max()

grid_res = 50  # You can adjust this for finer/coarser grids.
x = np.linspace(x_min, x_max, grid_res)
y = np.linspace(y_min, y_max, grid_res)
X, Y = np.meshgrid(x, y)
# Each row in point_grid is a 2D point.
point_grid = np.column_stack([X.ravel(), Y.ravel()])

# ------------------------------------------------------------
# STEP 2. Precompute the predicted pi for each grid point.
# ------------------------------------------------------------
pi_pred_list = []
for point in point_grid:
    pi_pred = inverse_function(point, model=vae, pca=pca,
                               scaler=standardizer, filter=pixes_tokeep)
    pi_pred_list.append(pi_pred)

# ------------------------------------------------------------
# STEP 3. Create the figure and initial plots.
# ------------------------------------------------------------
fig, (ax_scatter, ax_image) = plt.subplots(1, 2, figsize=(12, 6))

# For the scatter plot we want to color by a distance value.
# Here we initialize a new column 'dist_pi' (it will be updated by mouse moves).
mf['dist_pi'] = 0.0

# Initial scatter plot on the left.
sc_scatter = plot_vae_dist_plt(ax_scatter, mf, points='pca_vae', 
                               dist='dist_pi', vmin=0.0, vmax=0.07)

# Initial heatmap on the right: use a dummy grid (all zeros) for now.
Z_initial = np.zeros((grid_res, grid_res))
extent = [x_min, x_max, y_min, y_max]
hm = plot_heatmap(ax_image, Z_initial, extent, vmin=0.0, vmax=0.07)

# Create colorbars for both plots.
cbar_scatter = fig.colorbar(sc_scatter, ax=ax_scatter)
cbar_heatmap = fig.colorbar(hm, ax=ax_image)

# ------------------------------------------------------------
# STEP 4. Define the mouse-motion callback.
# ------------------------------------------------------------
def on_mouse_move(event):
    # Only update if the mouse is over the scatter plot.
    if event.inaxes == ax_scatter:
        mouse_x, mouse_y = event.xdata, event.ydata
        if mouse_x is None or mouse_y is None:
            return
        
        print(f"Mouse position: ({mouse_x:.2f}, {mouse_y:.2f})")
        mouse_pos = np.array([mouse_x, mouse_y])
        
        # Compute the corresponding reference point in the original space.
        pi_origin = inverse_function(mouse_pos, model=vae, pca=pca,
                                      scaler=standardizer, filter=pixes_tokeep)
        # For each grid point, compute the Euclidean distance between its
        # precomputed predicted pi and the new pi_origin.
        dist_pi_pred = [np.linalg.norm(pi - pi_origin) for pi in pi_pred_list]
        # Reshape into a 2D grid.
        Z = np.array(dist_pi_pred).reshape(grid_res, grid_res)
        
        # Also update the scatter plot by computing distances for each data point.
        mf['dist_pi'] = mf['pi'].apply(lambda pi: np.linalg.norm(pi - pi_origin))
        sc_new = plot_vae_dist_plt(ax_scatter, mf, points='pca_vae', 
                                   dist='dist_pi', vmin=0.0, vmax=0.07)
        cbar_scatter.update_normal(sc_new)
        
        # Update the heatmap.
        hm_new = plot_heatmap(ax_image, Z, extent, vmin=0.0, vmax=0.07)
        cbar_heatmap.update_normal(hm_new)
        
        # Redraw the figure.
        fig.canvas.draw_idle()

# Connect the mouse motion event.
fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

plt.show()
