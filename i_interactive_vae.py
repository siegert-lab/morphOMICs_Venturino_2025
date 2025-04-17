import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from morphomics.io.io import save_obj, load_obj
from utils import get_2d, inverse_function
from src.kxa_analysis.plot import plot_2d, plot_2d_plt
path = 'results/vae/trained_vae'
my_pip = load_obj(path)

vae_pip = my_pip.metadata

pixes_tokeep = vae_pip['pixes_tokeep']
standardizer = vae_pip['standardizer']
pca, vae = vae_pip['fitted_pca_vae']

mf = my_pip.morphoframe['v1_pi']
# Reset index and store the old index in a new column
mf = mf.reset_index()  # Resets the index and adds the old index as a column
# Rename the old index column to 'old_idcs'
mf.rename(columns={'index': 'old_idcs'}, inplace=True)

mf_vae_kxa = mf[mf['Model'].isin(['1xSaline_4h', '1xKXA_4h'])]

# Extract x, y from 'pca_vae'
x, y = zip(*mf_vae_kxa['pca_vae'])

# Initialize figure with 2 subplots: scatter plot + image
fig, (ax_scatter, ax_image) = plt.subplots(1, 2, figsize=(10, 5))

# Pass ax_scatter to the function
plot_2d_plt(df=mf_vae_kxa,
             feature='pca_vae', 
             title =  'VAE Latent Space',
               ax=ax_scatter)

# Initial empty image
img_placeholder = np.zeros((100, 100))  # Adjust size if needed
image_display = ax_image.imshow(img_placeholder, cmap='hot')
ax_image.set_title("Generated Image")

# Invert the Y-axis of the image
ax_image.invert_yaxis()

# Function to update image based on mouse hover
def on_mouse_move(event):
    if event.inaxes == ax_scatter:
        mouse_x, mouse_y = event.xdata, event.ydata
        if mouse_x is None or mouse_y is None:
            return
        
        print(f"Mouse position: ({mouse_x:.2f}, {mouse_y:.2f})")  # Debugging

        mouse_pos = np.array([mouse_x, mouse_y])
        img = inverse_function(mouse_pos, 
                               model=vae,
                               pca=pca,
                               scaler=standardizer,
                               filter=pixes_tokeep)

        print(f"Generated image min: {np.min(img)}, max: {np.max(img)}")  # Debugging
        
        img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
        img_2d = get_2d(img_normalized)

        image_display.set_data(img_2d)
        image_display.set_clim(0, 1)
        ax_image.set_title(f"Point: ({mouse_pos[0]:.2f}, {mouse_pos[1]:.2f})")
        fig.canvas.draw_idle()


# Connect mouse motion event
fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

plt.show()
