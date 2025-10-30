# FFT → spectrum visualization → notch filtering → IFFT (on your image)
# You can copy this into your own notebook/script; I've kept the code modular and commented.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max

# 1) Load grayscale image as float32 in [0,1]
path = "source.png"
img_u8 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = img_u8.astype(np.float32) / 255.0
H, W = img.shape

# Show original
plt.figure(figsize=(8,4))
plt.imshow(img, cmap="gray")
plt.title("Original image")
plt.axis("off")
plt.show()

# 2) 2D FFT with zero-frequency shifted to center
F = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)

# 3) Visualize magnitude spectrum (log scale to enhance contrast)
mag = np.log1p(np.abs(Fshift))  # log(1+|F|)

plt.figure(figsize=(8,4))
plt.imshow(mag, cmap="gray")
plt.title("Log magnitude spectrum (centered)")
plt.axis("off")
plt.show()

# 4) Build a notch-reject mask automatically:
#    - find bright peaks in spectrum (excluding the DC center region)
#    - place symmetric Gaussian notches at those peaks
def gaussian_notch_mask(shape, centers, sigma):
    """Return a [0..1] notch-reject mask with Gaussian dips at given centers.
       centers are (row, col) indices in the shifted spectrum coordinates."""
    H, W = shape
    rr, cc = np.ogrid[:H, :W]
    mask = np.ones((H, W), dtype=np.float32)
    for (r, c) in centers:
        d2 = (rr - r)**2 + (cc - c)**2
        mask *= 1.0 - np.exp(-0.5 * d2 / (sigma**2))
    return mask

# Detect peaks (candidate grid harmonics). Tune threshold and min_distance if needed.
# Ignore a small window around DC to avoid selecting the huge central peak.
mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
dc_margin = 25
mask_dc = np.ones_like(mag_norm, dtype=bool)
mask_dc[H//2 - dc_margin:H//2 + dc_margin + 1, W//2 - dc_margin:W//2 + dc_margin + 1] = False
mag_for_peaks = mag_norm.copy()
mag_for_peaks[~mask_dc] = 0.0

# peak_local_max returns coordinates (row, col)
peaks = peak_local_max(mag_for_peaks, min_distance=12, threshold_rel=0.45, num_peaks=20)

# Only keep peaks roughly aligned with horizontal/vertical axes (grid artifact).
# (i.e., row ≈ center or col ≈ center, within a band)
band = 30
sel = []
for r, c in peaks:
    if abs(r - H//2) < band or abs(c - W//2) < band:
        sel.append((int(r), int(c)))

# Enforce symmetry: for each (r,c) include its opposite (H-1-r, W-1-c)
sel_sym = set()
for r, c in sel:
    sel_sym.add((r, c))
    sel_sym.add((H - 1 - r, W - 1 - c))
sel = sorted(list(sel_sym))

# Build the notch mask
sigma = 6  # Gaussian width of each notch (pixels)
notch_mask = gaussian_notch_mask((H, W), sel, sigma)

# Visualize the mask over spectrum (just the mask itself here)
plt.figure(figsize=(8,4))
plt.imshow(notch_mask, cmap="gray")
plt.title("Gaussian notch-reject mask (white=pass, black=reject)")
plt.axis("off")
plt.show()

# 5) Apply mask in frequency domain and invert FFT
Fshift_filt = Fshift * notch_mask
img_filt = np.fft.ifft2(np.fft.ifftshift(Fshift_filt))
img_filt = np.real(img_filt)

# Normalize to [0,1] for display
img_filt_n = (img_filt - img_filt.min()) / (img_filt.max() - img_filt.min() + 1e-8)

plt.figure(figsize=(8,4))
plt.imshow(img_filt_n, cmap="gray")
plt.title("Result after notch filtering + IFFT")
plt.axis("off")
plt.show()

# For reference, print the selected peak coordinates (in shifted spectrum indices)
print("Selected notch centers (row, col):", sel)
