import cv2
import matplotlib.pyplot as plt
from remove_sky import remove_sky

# --- Config ---
image_path = "panorama.jpg"  # <-- sostituisci con il path della tua immagine
method = "deeplab"  # oppure "hsv"

# --- Load image ---
print("Caricamento immagine...")
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# --- Process ---
print("Avvio rimozione...")
sky_removed = remove_sky(image_rgb, method=method)

# --- Plot ---
print("Creazione plots...")
fig, ax = plt.subplots(2, 1, figsize=(5, 5))
ax[0].imshow(image_rgb)
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(sky_removed)
ax[1].set_title(f"Sky Removed ({method})")
ax[1].axis('off')

plt.tight_layout()
plt.show()