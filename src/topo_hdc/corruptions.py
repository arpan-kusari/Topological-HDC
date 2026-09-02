import numpy as np
from skimage.transform import resize
import cv2
from .config import CorruptionConfig

def apply_rotation(X_test, angle, scale=1):
    X_test_rot = []
    for img in X_test:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)  # Calculate the image center

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # Apply the rotation
        rot_img = cv2.warpAffine(img, M, (w, h))
        X_test_rot.append(rot_img)
    return X_test_rot
    
def aplly_gaussian_noise(X_test, sigma):
    """Add N(0, sigma^2) noise and clip to [0,1]."""
    X_test_g = []
    noise = np.random.normal(loc=0.0, scale=sigma, size=X_test[0].shape)
    for img in X_test:
        img_g = np.clip(img + noise, 0.0, 1.0)
        X_test_g.append(img_g)    
    return X_test_g

def apply_salt_pepper_noise(X_test, p, xmin=0, xmax=255):
    """
    Salt-and-pepper noise: corrupt fraction p of pixels.
    Half become 0 ("pepper"), half become 1 ("salt").
    """
    X_test_salt = []
    for img in X_test:
        r = np.random.rand(*img.shape)
        img_s = img.copy()
        img_s[r < (p / 2)] = xmin
        img_s[(r >= (p / 2)) & (r < p)] = xmax
        X_test_salt.append(img_s)
    return X_test_salt
    
def apply_cutout(X_test, size, value=0):
    """
    Random square occlusion (cutout).
    size: side length of the square mask.
    value: fill value (0.0 for black, 1.0 for white, or e.g. img.mean()).
    """
    X_test_cut = []
    for img in X_test:
        h, w = img.shape[:2]
        img_c = img.copy()

        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)
        half = size // 2

        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)

        img_c[y1:y2, x1:x2] = value
        X_test_cut.append(img_c)
    return X_test_cut
    
def zoom_to_canvas(img, zoom=1.2, out_size=28, order=1):
    """
    zoom>1: zoom in (crop center, then resize back)
    zoom<1: zoom out (shrink, then pad into canvas)
    img: HxW
    returns out_size x out_size
    """
    x = img.astype(np.float32)
    H, W = x.shape
    assert H == W, "expected square image"

    if zoom == 1.0:
        return x.copy()

    if zoom > 1.0:
        # crop center
        crop_size = int(round(H / zoom))
        crop_size = max(1, min(H, crop_size))
        y0 = (H - crop_size) // 2
        x0 = (W - crop_size) // 2
        crop = x[y0:y0+crop_size, x0:x0+crop_size]
        z = resize(crop, (out_size, out_size), order=order,
                   anti_aliasing=True, preserve_range=True).astype(np.float32)
        return z

    # zoom < 1.0: shrink then pad
    small_size = int(round(H * zoom))
    small_size = max(1, small_size)
    small = resize(x, (small_size, small_size), order=order,
                   anti_aliasing=True, preserve_range=True).astype(np.float32)

    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    y0 = int((out_size - small_size) // 2)
    x0 = int((out_size - small_size) // 2)
    canvas[y0:y0+small_size, x0:x0+small_size] = small
    return canvas
    
def apply_zoom(X_test, zoom, out_size):
    X_zoom = []
    for img in X_test:
        out_size = img.shape[0]
        img_z = zoom_to_canvas(img, zoom=zoom, out_size=out_size)
        X_zoom.append(img_z)
    return X_zoom
   
def apply_corruption(X: np.ndarray, cfg: CorruptionConfig):
    if cfg.kind == "rotation":
        X = apply_rotation(X_test=X, angle=np.deg2rad(cfg.angle_deg), scale=1)
    elif cfg.kind == "gaussian":
        X = apply_gaussian_noise(X_test=X, sigma=cfg.sigma)
    elif cfg.kind == "saltpepper":
        X = apply_salt_peppe_noise(X_test=X, p=cfg.p)
    elif cfg.kind == "cutout":
        X = apply_cutout(X_test=X, size=cfg.cutout_size)
    elif cfg.kind == "zoom":
        X = apply_zoom(X_test=X, zoom=cfg.zoom)
    return X
        

