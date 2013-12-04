from __init__ import spice
import healpy as hp
import numpy as np

lmax = 512
input_cl = np.ones(lmax + 1)
np.random.seed(1000)
input_map = hp.synfast(input_cl, nside=256, pixwin=True)

mask = np.ones_like(input_map)
hp.write_map("total_mask.fits", mask)

cl = spice(bin=False, norm=False, mapfile=input_map, nlmax=lmax, polarization="NO", beam_file="NO", beam_file2="NO")

import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.plot(cl)
plt.plot(input_cl, color="red", linewidth=3)
plt.xlim([0, lmax+1])
plt.xlabel("$\ell$")
plt.ylabel("$C_\ell$")
