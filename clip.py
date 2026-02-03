import numpy as np

csi = np.load("csi_3min_1min2min/csi_3min_1min2min.npy")

action_clip = csi[650:692, :, :, :]
np.save("csi_3min_1min2min/action_2.npy", action_clip)