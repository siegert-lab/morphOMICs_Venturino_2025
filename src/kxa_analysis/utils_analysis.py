import numpy as np
import torch as th
def get_2d(pi):
    size = int(np.sqrt(pi.shape[0]))
    pi_2d = pi.reshape(size, size)
    return pi_2d

def get_2d(pi):
    size = int(np.sqrt(pi.shape[0]))
    pi_2d = pi.reshape(size, size)
    return pi_2d

def mask_pi(pi, pixes_tokeep):    
    # Create a mask of zeros
    pi_threshold = np.zeros_like(pi)

    # Assign original values only at the specified indices
    pi_filtered = pi[pixes_tokeep]
    pi_threshold[pixes_tokeep] = pi_filtered
    return pi_threshold, pi_filtered

def get_base(pi, pixes_tokeep):
    pi_full = np.zeros((10000))
    pi_full[pixes_tokeep] = pi
    return pi_full

def inverse_function(point, model, pca, scaler, filter):
    model.eval()
    with th.no_grad():
        # Pass the data through the model
        point_tensor = th.tensor(point, dtype=th.float32)
        out = model.decoder(point_tensor)
        pred_processed_pi = out.cpu().detach().numpy().reshape(1, -1)
        pred_scaled_pi = pca.inverse_transform(pred_processed_pi)
        pi_scaled = get_base(pred_scaled_pi, filter)
        pred_filter_pi = scaler.inverse_transform(pred_scaled_pi)
        pi = get_base(pred_filter_pi, filter)
        return pi, pi_scaled
