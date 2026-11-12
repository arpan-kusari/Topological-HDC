from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, center_of_mass
from skimage.filters import threshold_otsu, threshold_local, gaussian
from skimage.morphology import closing, opening, remove_small_objects, disk, closing
from skimage.measure import label, regionprops, find_contours
from mahotas.features import zernike_moments
from skimage.transform import resize
from skimage.feature import hog
import numpy as np
from dataclasses import dataclass
from gtda.homology import CubicalPersistence
from gtda.images import HeightFiltration
from gtda.diagrams import PersistenceImage
import cripser
from topo_hdc.config import AblationConfig

class ProcessImage:
    def __init__(self, img):
        self.img = img
        self.bw = None
        self.lab = None
        self.contours = None
        self.zm = None
        self.hog = None
        self.points = None

    def preprocess(self, block_size=15, offset=0.0, max_size=5, keep_largest=True):
        """
        Adaptive thresholding for MNIST-like digits.
        img: 28x28, float or uint8
        Returns: boolean mask
        """
        x = self.img.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0  # -> [0,1]

        # Ensure odd block size
        if block_size % 2 == 0:
            block_size += 1

        T = threshold_local(x, block_size=block_size, method="gaussian", offset=offset)
        self.bw = x > T

        # Very gentle cleanup: remove tiny specks only
        self.bw = remove_small_objects(self.bw, max_size=max_size)

        if not keep_largest:
            return self.bw

        self.lab = label(self.bw)
        if self.lab.max() == 0:
            return self.bw
        largest = max(regionprops(self.lab), key=lambda r: r.area).label
        return self.lab == largest
        
    def bbox_from_intensity(self, pad=2):
        x = self.img.astype(np.float32)
        if x.max() > 1.5:
            x /= 255.0
            
        t = threshold_otsu(x)
        # FIX: Change to x > t since MNIST/EMNIST letters are bright on a dark background
        bw = x > t  

        lab = label(bw)
        if lab.max() == 0:
            return None

        r = max(regionprops(lab), key=lambda rr: rr.area)
        y0, x0, y1, x1 = r.bbox 
        y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
        y1 = min(x.shape[0], y1 + pad); x1 = min(x.shape[1], x1 + pad)
        return y0, y1, x0, x1


    def crop_and_normalize_gray_bbox(self, bbox, out_size=64):
        x = self.img.astype(np.float32)
        if x.max() > 1.5:
            x /= 255.0

        if bbox is None:
            return np.zeros((out_size, out_size), dtype=np.float32)

        y0, y1, x0, x1 = bbox
        crop = x[y0:y1, x0:x1]

        h, w = crop.shape
        s = max(h, w)
        sq = np.zeros((s, s), dtype=np.float32)
        oy = (s - h) // 2
        ox = (s - w) // 2
        sq[oy:oy+h, ox:ox+w] = crop

        return resize(sq, (out_size, out_size), order=1, anti_aliasing=True,
                  preserve_range=True).astype(np.float32)
                  
    def normalize_glyph_gray(self, out_size=64, pad=2):
        bbox = self.bbox_from_intensity(pad=pad)
        norm = self.crop_and_normalize_gray_bbox(bbox, out_size=out_size)
        # keep values in [0,1]
        return np.clip(norm, 0.0, 1.0).astype(np.float32)
        
    def fourier_mellin_descriptor(self, out_size=64, num_rho=32, num_theta=64):
        """
        Computes a Rotation, Translation, and Scale (RTS) invariant 
        Fourier-Mellin descriptor directly from the normalized gray glyph.
        Returns a flat 1D float64 feature vector.
        """
        # 1. Normalize glyph to handle baseline centering/scale
        norm = self.normalize_glyph_gray(out_size=out_size)
        # 2. Build Log-Polar Coordinate Grid around the image center
        h, w = norm.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        max_r = np.hypot(cy, cx)
        # Logarithmic mapping for radius, linear for angle
        rhos = np.logspace(0, np.log10(max_r), num_rho)
        thetas = np.linspace(0, 2 * np.pi, num_theta, endpoint=False)
        
        rho_grid, theta_grid = np.meshgrid(rhos, thetas, indexing='ij')
        # Map back to Cartesian coordinates to sample the patch
        ys = cy + rho_grid * np.sin(theta_grid)
        xs = cx + rho_grid * np.cos(theta_grid)
        
        # Clip coordinates safely inside image bounds
        ys = np.clip(ys, 0, h - 1).astype(int)
        xs = np.clip(xs, 0, w - 1).astype(int)
        
        log_polar_img = norm[ys, xs]
        # 3. Compute 2D FFT Magnitude to achieve shift invariance 
        # (which maps back to Scale/Rotation invariance in log-polar space)
        f_transform = np.fft.fft2(log_polar_img)
        f_magnitude = np.abs(f_transform)
        
        # Flatten into a continuous descriptor vector for HDC encoding
        return f_magnitude.flatten().astype(np.float64)
        
#    def persistent_image_descriptor(self, cfg: RobustHoleConfig, grid_size=20, sigma=None, num_angles=8):
#        """
#        Extracts directional spoke-wheel topological features across closing scales 
#        and converts them into a concatenated Persistent Image vector.
#        """
#        foreground = np.asarray(self.bw, dtype=bool)
#        if not foreground.any():
#            return np.zeros(num_angles * grid_size * grid_size, dtype=np.float64)
#            
#        # 1. Calculate Center of Mass to serve as the spoke wheel axle
#        cy, cx = center_of_mass(foreground)
#        height, width = foreground.shape
#        
#        # 2. Define angles for the spokes (e.g., 0, 45, 90, 135 degrees)
#        angles = np.linspace(0, np.pi, num_angles, endpoint=False)
#        
#        # Pre-calculate relative pixel coordinates from the center of mass
#        y_indices, x_indices = np.indices((height, width))
#        rel_y = y_indices - cy
#        rel_x = x_indices - cx
#        
#        radii = sorted(cfg.closing_radii)
#        max_r = max(radii) if radii else 3 #TODO
#        
#        # We will accumulate flattened persistent images for each direction
#        directional_vectors = []
#        
#        # 3. Perform a directional spoke wheel sweep for each angle
#        for theta in angles:
#            # Project relative coordinates onto the current spoke vector orientation
#            # This gives a 1D spatial "filtration coordinate" for every pixel
#            projected_coords = rel_x * np.cos(theta) + rel_y * np.sin(theta)
#            
#            # Capture the ink intensity paired with its spatial location
#            ink_pixels = foregound > 0.1
#            
#            if not ink_pixels.any():
#                directional_vectors.append(np.zeros(grid_size * grid_size, dtype=np.float64))
#                
#            # physical locations of ink along this specific route
#            spatial_locations = projected_coords[ink_pixels]
#            ink_intensities = foreground[ink_pixels]
#            
#            # 4. Rasterize onto a fixed 2D grid
#            # Set spatial boundary limits based on the maximum possible distance from center of mass
#            max_dist = np.sqrt((height/2)**2 + (width/2)**2)
#            b_bins = np.linspace(-max_dist, max_dist, grid_size)
#            p_bins = np.linspace(0.0, 1.0, grid_size)
#            
#            # Auto-tune sigma if not provided, ensuring it maps to grid cell step size
#            if sigma is None:
#                b_step = (2 * max_dist) / grid_size
#                current_sigma = b_step * 0.5  # Smoothly influences immediate neighbors
#            else:
#                current_sigma = sigma
#                
#            pi_matrix = np.zeros((grid_size, grid_size), dtype=np.float64)
#            
#            for b, p in zip(spatial_locations, ink_intensities):
#                for i in range(grid_size):
#                    for j in range(grid_size):
#                        dist_sq = (b_bins[i] - b)**2 + (p_bins[j] - p)**2
#                        weight = p # Weight linearly by morphological persistence scale
#                        pi_matrix[i, j] += weight * np.exp(-dist_sq / (2.0 * current_sigma**2))
#                        
#            directional_vectors.append(pi_matrix.flatten())
#            
#        # Concatenate all directions into a single spatial-topological descriptor vector
#        return np.concatenate(directional_vectors).astype(np.float64)

#    def persistent_image_descriptor(self, cfg: RobustHoleConfig):
#        # Apply a directional Height Filtration to create a topological scalar field
#        # This extracts spatial shape information along a specific axis
#        # 1. Define multiple directions to capture all angles
#        direction_list = [
#            np.array([1, 0]), 
#            np.array([0, 1]), 
#            np.array([1, 1]), 
#            np.array([1, -1])
#        ]

#        single_image = self.bw.reshape((1, 28, 28))
#        # We will collect the diagrams for each direction
#        diagrams_for_this_image = []
#        cp = CubicalPersistence(homology_dimensions=[0, 1], n_jobs=-1)

#        # 3. Loop through directions to avoid the list TypeError
#        for d in direction_list:
#            hf = HeightFiltration(direction=d)
#            filtered_digit = hf.fit_transform(single_image)
#            persistence_diagrams = cp.fit_transform(filtered_digit)
#            
#            # persistence_diagrams has shape (1, n_points, 3). 
#            # We take [0] to extract the diagram for this specific image sample.
#            diagrams_for_this_image.append(persistence_diagrams[0])
#            
#        # Return the raw diagrams list so we can apply a global coordinate grid outside
#        return diagrams_for_this_image


    def persistent_image_descriptor(
        self,
        cfg: AblationConfig,
    ) -> list[np.ndarray]:
        """
        Compute directional persistence diagrams from a binary foreground mask.

        Returns one diagram per selected direction, in cfg.pi_directions order.
        Each row is [birth, persistence, homology_dimension].

        Weighting and smoothing are applied later during PI vectorization.
        """
        if not cfg.use_pi:
            return []

        if self.bw is None:
            raise ValueError(
                "self.bw is not initialized. Set it to the intended binary "
                "foreground mask before computing persistence diagrams."
            )

        foreground = np.asarray(self.bw)

        if foreground.ndim != 2:
            raise ValueError(
                "Expected a two-dimensional foreground mask; "
                f"got shape {foreground.shape}."
            )

        if foreground.size == 0:
            raise ValueError("The foreground mask must not be empty.")

        # Require an explicit binary mask rather than silently interpreting
        # all nonzero grayscale pixels as foreground.
        if not np.isin(foreground, [0, 1]).all():
            raise ValueError(
                "self.bw must be binary, with True/1 representing foreground."
            )

        foreground = foreground.astype(bool, copy=False)

        if not foreground.any():
            return [
                np.empty((0, 3), dtype=np.float64)
                for _ in cfg.pi_directions
            ]

        single_image = foreground[np.newaxis, ...]
        diagrams_for_this_image = []

        for direction in cfg.pi_directions:
            filtration = HeightFiltration(
                direction=np.asarray(direction, dtype=np.float64)
            )
            filtered_digit = filtration.fit_transform(single_image)

            # Remove sample axis: (1, H, W) -> (H, W).
            grid_2d = np.ascontiguousarray(
                filtered_digit[0],
                dtype=np.float64,
            )

            raw_ph = np.asarray(
                cripser.compute_ph(
                    grid_2d,
                    filtration="V",
                    maxdim=max(cfg.pi_homology_dims),
                ),
                dtype=np.float64,
            )

            if raw_ph.size == 0:
                diagrams_for_this_image.append(
                    np.empty((0, 3), dtype=np.float64)
                )
                continue

            if raw_ph.ndim != 2 or raw_ph.shape[1] < 3:
                raise ValueError(
                    f"Unexpected cripser output shape: {raw_ph.shape}."
                )

            dimensions = raw_ph[:, 0]
            births = raw_ph[:, 1]
            deaths = raw_ph[:, 2]

            # Exclude nonfinite pairs and essential classes represented
            # by the largest float64 value.
            valid = (
                np.isfinite(births)
                & np.isfinite(deaths)
                & (deaths < np.finfo(np.float64).max)
                & (deaths > births)
                & np.isin(dimensions, cfg.pi_homology_dims)
            )

            births = births[valid]
            deaths = deaths[valid]
            dimensions = dimensions[valid]

            with np.errstate(over="ignore", invalid="ignore"):
                persistences = deaths - births

            valid_lifetimes = (
                np.isfinite(persistences)
                & (persistences > 0)
            )

            formatted_diagram = np.column_stack(
                (
                    births[valid_lifetimes],
                    persistences[valid_lifetimes],
                    dimensions[valid_lifetimes],
                )
            )

            diagrams_for_this_image.append(formatted_diagram)

        return diagrams_for_this_image

        
        
    def svhn_soft_crop(self, img_gray, out_size=32, pad=4, gamma=2.0):
        """
        Segmentation-free normalization:
        - build a soft 'ink' map from darkness
        - compute weighted center of mass
        - crop a fixed window around COM (with padding), then resize
        """
        x = img_gray.astype(np.float32)
        if x.max() > 1.5:
            x /= 255.0
        x = np.clip(x, 0.0, 1.0)

        # "ink" = darkness (digits are usually darker than background)
        ink = (1.0 - x)
        ink = np.power(np.clip(ink, 0.0, 1.0), gamma)

        s = ink.sum()
        if s < 1e-6:
            return resize(x, (out_size, out_size), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)

        H, W = x.shape
        ys, xs = np.mgrid[0:H, 0:W]
        cy = (ink * ys).sum() / s
        cx = (ink * xs).sum() / s

        # crop a window around the COM
        win = min(H, W) - 2 * pad
        win = max(24, int(win))  # keep a reasonable min
        y0 = int(round(cy - win / 2)); x0 = int(round(cx - win / 2))
        y0 = np.clip(y0, 0, H - win); x0 = np.clip(x0, 0, W - win)

        crop = x[y0:y0+win, x0:x0+win]
        norm = resize(crop, (out_size, out_size), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
        return np.clip(norm, 0.0, 1.0)
        
    def svhn_contrast_norm(self, x):
        m = x.mean()
        sd = x.std() + 1e-6
        x = (x - m) / sd
        # clip instead of sigmoid to preserve edges
        x = np.clip(x, -3.0, 3.0)
        # scale to [0,1]
        return ((x + 3.0) / 6.0).astype(np.float32)


    def hog_descriptor(self, norm, data='mnist'):
        if data == 'mnist' or data == 'emnist':
            x = resize(norm, (28, 28), order=1, anti_aliasing=True, preserve_range=True).astype(np.float32)
        elif data == 'svhn':
            x = self.svhn_soft_crop(self.img, out_size=32, pad=6, gamma=2.0) 
            x = self.svhn_contrast_norm(x)
        self.hog = hog(
            x,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return self.hog.astype(np.float64)

    
