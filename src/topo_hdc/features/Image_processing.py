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
from topo_hdc.config import RobustHoleConfig
from gtda.homology import CubicalPersistence
from gtda.images import HeightFiltration
from gtda.diagrams import PersistenceImage

@dataclass
class HoleCandidate:
    mask: np.ndarray
    centroid_yx: np.ndarray
    seed_yx: np.ndarray
    area: int
    enclosure_score: float
    missing_fraction: float
    scale_count: int = 1

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
        # foreground = darker pixels
        t = threshold_otsu(x)
        bw = x < t  # letters are dark on light background

        lab = label(bw)
        if lab.max() == 0:
            return None

        r = max(regionprops(lab), key=lambda rr: rr.area)
        y0, x0, y1, x1 = r.bbox  # y1/x1 are exclusive
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

    def persistent_image_descriptor(self, cfg: RobustHoleConfig):
        # Apply a directional Height Filtration to create a topological scalar field
        # This extracts spatial shape information along a specific axis
        # 1. Define multiple directions to capture all angles
        direction_list = [
            np.array([1, 0]), 
            np.array([0, 1]), 
            np.array([1, 1]), 
            np.array([1, -1])
        ]

        single_image = self.bw.reshape((1, 28, 28))
        # We will collect the diagrams for each direction
        diagrams_for_this_image = []
        cp = CubicalPersistence(homology_dimensions=[0, 1], n_jobs=-1)

        # 3. Loop through directions to avoid the list TypeError
        for d in direction_list:
            hf = HeightFiltration(direction=d)
            filtered_digit = hf.fit_transform(single_image)
            persistence_diagrams = cp.fit_transform(filtered_digit)
            
            # persistence_diagrams has shape (1, n_points, 3). 
            # We take [0] to extract the diagram for this specific image sample.
            diagrams_for_this_image.append(persistence_diagrams[0])
            
        # Return the raw diagrams list so we can apply a global coordinate grid outside
        return diagrams_for_this_image
        

    # ----------------------------
    # Step 2. Extract contours (outer + holes)
    # ----------------------------
    def extract_contours(self):
        """
        Returns a list of contours. Each contour is an (N,2) array of (row, col) coordinates.
        find_contours finds iso-valued contours; on binary images use level=0.5.
        """
        self.contours = find_contours(self.bw.astype(float), level=0.5)
        # Sort by length descending
        self.contours = sorted(self.contours, key=lambda c: c.shape[0], reverse=True)
        return self.contours

    def zernike_descriptor_gray(self, norm, degree=6, radius=20, out_size=64):
        self.zm = zernike_moments(norm, radius=radius, degree=degree)
        return np.abs(self.zm).astype(np.float64)
        
    def spatial_pyramid_zernike_gray(self, norm, out_size=64, degree=6, radius=None, grid=(2,2), pad=2):
        """
        Spatial pyramid Zernike:
          - full image moments
          - grid moments (e.g. 2x2)
        Returns 1D float64 feature vector.

        img: 28x28 EMNIST image (float [0,1] or uint8)
        """

        if radius is None:
            radius = out_size // 2 - 1  # conservative to stay inside disk

        feats = []

        # 1) full region
        zm_full = zernike_moments(norm, radius=radius, degree=degree)
        feats.append(np.abs(zm_full))

        # 2) grid regions
        gy, gx = grid
        h, w = norm.shape
        ys = np.linspace(0, h, gy + 1, dtype=int)
        xs = np.linspace(0, w, gx + 1, dtype=int)

        # Use a smaller radius per cell
        cell_h = ys[1] - ys[0]
        cell_w = xs[1] - xs[0]
        r_cell = min(cell_h, cell_w) // 2 - 1
        r_cell = max(3, int(r_cell))

        for iy in range(gy):
            for ix in range(gx):
                patch = norm[ys[iy]:ys[iy+1], xs[ix]:xs[ix+1]]
                # Ensure patch is square for zernike_moments; pad if needed
                ph, pw = patch.shape
                s = max(ph, pw)
                sq = np.zeros((s, s), dtype=np.float32)
                oy = (s - ph) // 2
                ox = (s - pw) // 2
                sq[oy:oy+ph, ox:ox+pw] = patch

                # Resize patch to a fixed size so r_cell is consistent
                patch_size = 32
                sq = resize(sq, (patch_size, patch_size), order=1,
                        anti_aliasing=True, preserve_range=True).astype(np.float32)
                sq = np.clip(sq, 0.0, 1.0)

                zm = zernike_moments(sq, radius=min(r_cell, patch_size//2 - 1), degree=degree)
                feats.append(np.abs(zm))

        return np.concatenate(feats, axis=0).astype(np.float64)
        
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

    # ----------------------------
    # Step 3. Resample a contour to fixed number of points
    # ----------------------------
    def resample_polyline(self, points, n=256, closed=True):
        """
        points: (N,2) in (row,col). We'll treat as (y,x).
        Returns (n,2) resampled.
        """
        p = points.astype(np.float64)
        if closed:
            # ensure closure
            if np.linalg.norm(p[0] - p[-1]) > 1e-6:
                p = np.vstack([p, p[0]])

        # cumulative arc-length
        d = np.sqrt(((p[1:] - p[:-1]) ** 2).sum(axis=1))
        s = np.hstack([[0.0], np.cumsum(d)])
        if s[-1] < 1e-8:
            return np.repeat(p[:1], n, axis=0)

        # target arc-length positions
        t = np.linspace(0, s[-1], n, endpoint=not closed)
        # interpolate each coordinate
        y = np.interp(t, s, p[:, 0])
        x = np.interp(t, s, p[:, 1])
        return np.stack([y, x], axis=1)


    # ----------------------------
    # Step 4. Periodic spline smoothing
    # ----------------------------
    def fit_periodic_spline(self, points, s=1.0, per=True):
        """
        Fit a parametric spline x(u), y(u).
        points: (N,2) with (y,x).
        Returns tck for splev.
        """
        y = points[:, 0]
        x = points[:, 1]
        # splprep expects [x,y] or [y,x]; we use [x,y] for convenience
        tck, _ = splprep([x, y], s=s, per=per)
        return tck


    def sample_spline(self, tck, n=256):
        u = np.linspace(0, 1, n, endpoint=False)
        x, y = splev(u, tck)
        return np.stack([y, x], axis=1)  # back to (y,x)


    # ----------------------------
    # Step 5. RTS-invariant descriptor:
    #   radial signature r(theta) -> Fourier magnitudes
    # ----------------------------
    def centroid_from_mask(self):
        ys, xs = np.nonzero(self.bw)
        if len(xs) == 0:
            return np.array([14.0, 14.0])
        return np.array([ys.mean(), xs.mean()])
        
    @staticmethod
    def _maximum_circular_false_run(values: np.ndarray) -> int:
        values = np.asarray(values, dtype=bool)
        if values.size == 0 or values.all():
            return 0
        if (~values).all():
            return len(values)
        doubled = np.concatenate([values, values])
        longest = 0
        current = 0
        for value in doubled:
            if value:
                current = 0
            else:
                current += 1
                longest = max(longest, current)
        return min(longest, len(values))
        
    @staticmethod
    def _interior_seed(region_mask: np.ndarray) -> np.ndarray:
        distance = distance_transform_edt(region_mask)
        
        if distance.max() <= 0:
            coordds = np.argwhere(region_mask)
            return coords.mean(axis=0).astype(np.floaat64)
            
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        return np.asarray([y, x], dtype=np.float64)
        
    @staticmethod
    def _candidate_match(
        first: HoleCandidate,
        second: HoleCandidate,
        distance_threshold: bool,
        ) -> bool:
        distance = np.linalg.norm(first.centroid_yx - second.centroid_yx)
        intersection = np.logical_and(first.mask, second.mask).sum()
        union = np.logical_or(first.mask, second.mask).sum()
        iou = intersection/max(union, 1)
        
        return distance <= distance_threshold or iou >= 0.2


    # ----------------------------
    # Holes: contours -> RTS-invariant feature vector
    # ----------------------------
    def spoke_enclosure(self,
                        foreground: np.ndarray,
                        seed_yx: np.ndarray,
                        num_spokes: int = 48,
                        ray_step: float = 0.5,
                        ) -> tuple[float, float]:
        """
        Returns: 
            enclosure_score: Fraction of rays that encounter foreground
            missing_fraction: Longest contiguous angular interval without foreground, normalized by number of spokes
        """
        foreground = np.asarray(foreground, dtype=bool)
        H, W = foreground.shape
        seed_y, seed_x = seed_yx
        
        max_radius = float(np.hypot(H, W))
        radii = np.arange(ray_step, max_radius + ray_step, ray_step)
        
        hits = np.zeros(num_spokes, dtype=bool)
        
        for j in range(num_spokes):
            theta = 2.0 * np.pi * j / num_spokes
            
            ys = np.rint(seed_y + radii * np.sin(theta)).astype(np.int64)
            xs = np.rint(seed_x + radii * np.cos(theta)).astype(np.int64)
            
            valid = (
                (ys >= 0)
                & (ys < H)
                & (xs >= 0)
                & (xs < W)
            )
            ys = ys[valid]
            xs = xs[valid]
            
            if len(xs) > 0:
                hits[j] = np.any(foreground[ys, xs])
        enclosure_score = float(hits.mean())
        missing_run = self._maximum_circular_false_run(hits)
        missing_fraction = float(missing_run / num_spokes)
        
        return enclosure_score, missing_fraction
        
    def hole_candidates_at_scale(
        self,
        foreground: np.ndarray,
        radius: int,
        cfg: RobustHoleConfig,
        ) -> list[HoleCandidate]:
        """
        Generate bounded-background candidates after closing, then validate
        them against the original foreground using the spoke wheel
        """
        foreground = np.asarray(foreground, dtype=bool)
        H, W = foreground.shape
        if radius == 0:
            closed = foreground.copy()
        else:
            closed = closing(
                foreground, footprint=disk(radius),
            )
        inverse = ~closed
        labels = label(inverse, connectivity=2)
        
        border_labels = set(
            np.unique(
                np.concatenate(
                    [
                        labels[0, :],
                        labels[-1, :],
                        labels[:, 0],
                        labels[:, -1],
                    ]
                )
            ).tolist()
        )
        
        max_area = int(cfg.max_area_fraction * H * W)
        candidates = []
        
        for region in regionprops(labels):
            if region.label in border_labels:
                continue
                
            if region.area < cfg.min_area or region.area > max_area:
                continue
                
            region_mask = labels == region.label
            seed_yx = self._interior_seed(region_mask)
            
            enclosure, missing_fraction = self.spoke_enclosure(
                foreground=foreground, 
                seed_yx=seed_yx, 
                num_spokes=cfg.num_spokes, 
                ray_step = cfg.ray_step,
            )
            if enclosure < cfg.enclosure_threshold:
                continue
            if missing_fraction > cfg.max_missing_fraction:
                continue
            
            centroid_yx = np.asarray(region.centroid, dtype=np.float64)
            
            candidates.append(
                HoleCandidate(
                    mask=region_mask, 
                    centroid_yx=centroid_yx,
                    seed_yx=seed_yx,
                    area=int(region.area),
                    enclosure_score=enclosure,
                    missing_fraction=missing_fraction,
                )
            )
        return candidates
        
    def merge_hole_candidates(
        self,
        candidate_groups: list[list[HoleCandidate]],
        cfg: RobustHoleConfig,
        ) -> list[HoleCandidate]:
        merged: list[list[tuple[int, HoleCandidate]]] = []
        
        for scale_index, candidates in enumerate(candidate_groups):
            for candidate in candidates:
                matching_group = None
                for group in merged:
                    if any(
                        self._candidate_match(
                            candidate, 
                            existing, 
                            cfg.merge_distance,
                            )
                            for _, existing in group
                    ):
                        matching_group = group
                        break
                            
                if matching_group is None:
                    merged.append([(scale_index, candidate)])
                else:
                    matching_group.append((scale_index, candidate))
        accepted = []
        for group in merged:
            scale_count = len({scale for scale, _ in group})
            if scale_count < cfg.min_scale_count:
                continue
            best = max(
                (candidate for _, candidate in group),
                key = lambda candidate: (
                    candidate.enclosure_score,
                    -candidate.missing_fraction,
                ),
            )
            best.scale_count = scale_count
            accepted.append(best)
        return sorted(
            accepted, 
            key=lambda candidate: candidate.area,
            reverse=True,
        )
            
    def robust_hole_masks(
        self,
        cfg: RobustHoleConfig | None = None,
    ) -> list[HoleCandidate]:
        if cfg is None:
            cfg = RobustHoleConfig()
        foreground = np.asarray(self.bw, dtype=bool)
        if not foreground.any():
            return []
        candidates_by_scale = [
            self.hole_candidates_at_scale(
                foreground=foreground,
                radius=radius,
                cfg=cfg,
            )
            for radius in cfg.closing_radii
        ]
        return self.merge_hole_candidates(
            candidate_groups=candidates_by_scale,
            cfg=cfg,
        )

    def extract_hole_contours(
        self,
        cfg: RobustHoleConfig | None = None,
    ) -> list[tuple[np.ndarray, HoleCandidate]]:
        hole_cs = []
        for hm in self.robust_hole_masks(cfg):
            cs = find_contours(hm.mask.astype(np.float32), level=0.5)
            if not cs:
                continue
                
            contour = max(cs, key = lambda c:c.shape[0])
            hole_cs.append((contour, hm))
        return hole_cs

    def hole_relative_to_outer(self, outer_pts_yx, hole_centroid_yx):
        # nearest outer boundary point
        d2 = np.sum((outer_pts_yx - hole_centroid_yx[None,:])**2, axis=1)
        i = int(np.argmin(d2))
        dist = np.sqrt(d2[i])
        t = i / outer_pts_yx.shape[0]  # normalized index ~ normalized arc-length (if uniformly sampled)
        return t, dist

    def canonical_frame_with_signfix(self, outer_pts_yx):
        P = np.asarray(outer_pts_yx, dtype=np.float64)
        c = P.mean(axis=0)
        P0 = P - c

        s = np.sqrt(np.mean(np.sum(P0**2, axis=1))) + 1e-12

        XY = np.stack([P0[:,1], P0[:,0]], axis=1)  # (x,y)
        cov = (XY.T @ XY) / max(1, XY.shape[0])
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, np.argmax(vals)]  # principal axis in (x,y), unit

        # sign-fix: choose direction so projection skewness is positive
        proj = XY @ v
        skew = np.mean(proj**3)
        if skew < 0:
            v = -v

        theta = np.arctan2(v[1], v[0])
        ct, st = np.cos(-theta), np.sin(-theta)
        R_xy = np.array([[ct, -st],
                         [st,  ct]], dtype=np.float64)
        return c, s, R_xy

    def canonicalize_point_yx(self, p_yx, c_yx, s, R_xy):
        """
        p_yx: (2,) point in (y,x) (e.g., hole centroid)
        Returns canonized (py, px) in (y,x), normalized by outer scale.
        """
        dy, dx = (np.asarray(p_yx, dtype=np.float64) - c_yx) / s

        # rotate in (x,y)
        x, y = dx, dy
        xr, yr = R_xy @ np.array([x, y], dtype=np.float64)

        return np.array([yr, xr], dtype=np.float64)  # (y,x)

    def radial_signature_from_points(self, points_yx, center_yx, n_theta=128):
        """
        Build r(theta) by mapping points to polar around center and taking
        max radius per angle bin (a star-shaped approximation).
        This avoids ray/segment intersection code.

        Works well for MNIST-like shapes; for concave shapes it’s an approximation.
        """
        p = points_yx - center_yx[None, :]
        y, x = p[:, 0], p[:, 1]
        r = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)  # [-pi, pi]
        # map to [0, 2pi)
        theta = (theta + 2 * np.pi) % (2 * np.pi)

        bins = (theta / (2 * np.pi) * n_theta).astype(int)
        bins = np.clip(bins, 0, n_theta - 1)

        sig = np.zeros(n_theta, dtype=np.float64)
        # max radius per bin
        for b, rv in zip(bins, r):
            if rv > sig[b]:
                sig[b] = rv

        # Fill empty bins by simple circular interpolation
        if np.any(sig == 0):
            idx = np.arange(n_theta)
            good = sig > 0
            if good.sum() >= 2:
                # circular interpolation: duplicate domain
                idx2 = np.hstack([idx[good], idx[good] + n_theta])
                sig2 = np.hstack([sig[good], sig[good]])
                sig_interp = np.interp(idx, idx2, sig2)
                sig = np.maximum(sig, sig_interp)
            else:
                # degenerate
                sig[:] = sig.max()

        return sig


    def fourier_magnitude_features(self, r_theta, k=24):
        """
        Rotation invariance by DFT magnitude.
        Drop DC (0). Return k magnitudes from 1..k.
        """
        r = r_theta.astype(np.float64)
        # scale normalize (RMS)
        rms = np.sqrt(np.mean(r ** 2)) + 1e-12
        r = r / rms

        F = np.fft.rfft(r)  # length n_theta//2 + 1
        mags = np.abs(F)
        # mags[0] is DC; take next k bins
        k = min(k, mags.shape[0] - 1)
        return mags[1:k + 1]


    def hole_feature_vector(self, contour_yx, n_resample=128, spline_s=0.5, n_theta=64, k=12):
        c = self.resample_polyline(contour_yx, n=n_resample, closed=True)
        tck = self.fit_periodic_spline(c, s=spline_s, per=True)
        pts = self.sample_spline(tck, n=n_resample)
        center = pts.mean(axis=0)
        rtheta = self.radial_signature_from_points(pts, center, n_theta=n_theta)
        return self.fourier_magnitude_features(rtheta, k=k)

    def foreground_com_canonical(self, c_outer_yx, s_outer, R_xy):
        """
        Returns center of mass of foreground pixels in canonical coords (y,x).
        """
        ys, xs = np.nonzero(self.bw)
        if len(xs) == 0:
            return np.array([0.0, 0.0], dtype=np.float64)

        pts = np.stack([ys, xs], axis=1).astype(np.float64)        # (y,x)
        P0 = (pts - c_outer_yx[None,:]) / s_outer                  # translate+scale in (y,x)

        # rotate in (x,y)
        x = P0[:,1]; y = P0[:,0]
        xr, yr = R_xy @ np.vstack([x, y])
        # mean in canonical frame
        return np.array([yr.mean(), xr.mean()], dtype=np.float64)  # (y,x)
        
    def spatial_pyramid_membership(
        self,
        point_yx: np.ndarray,
        outer_points_yx: np.ndarray,
        levels: tuple[int, ...] = (1, 2, 4),
    ) -> np.ndarray:
        """
        Encode a canonical hole position using one-hot cell membership at
        multiple spatial-pyramid levels.

        For levels (1, 2, 4), the output length is 1 + 4 + 16 = 21.

        Parameters
        ----------
        point_yx:
            Canonical hole centroid, shape (2,), in (y, x).
        outer_points_yx:
            Canonical outer-contour points, shape (N, 2).
        levels:
            Number of cells per dimension at each pyramid level.

        Returns
        -------
        np.ndarray:
            Concatenated one-hot cell memberships.
        """
        point_yx = np.asarray(point_yx, dtype=np.float64)
        outer_points_yx = np.asarray(
            outer_points_yx,
            dtype=np.float64,
        )

        if outer_points_yx.ndim != 2 or outer_points_yx.shape[1] != 2:
            raise ValueError(
                "outer_points_yx must have shape (N, 2)."
            )

        min_y, min_x = outer_points_yx.min(axis=0)
        max_y, max_x = outer_points_yx.max(axis=0)

        height = max(max_y - min_y, 1e-12)
        width = max(max_x - min_x, 1e-12)

        py, px = point_yx

        # Normalize relative to the canonical outer-shape bounding box.
        y01 = np.clip(
            (py - min_y) / height,
            0.0,
            1.0 - 1e-12,
        )
        x01 = np.clip(
            (px - min_x) / width,
            0.0,
            1.0 - 1e-12,
        )

        memberships = []

        for level in levels:
            if level <= 0:
                raise ValueError(
                    f"Pyramid level must be positive; received {level}."
                )

            row = min(int(y01 * level), level - 1)
            col = min(int(x01 * level), level - 1)

            one_hot = np.zeros(level * level, dtype=np.float64)
            cell_index = row * level + col
            one_hot[cell_index] = 1.0

            memberships.append(one_hot)

        return np.concatenate(memberships)
    
        
    def hole_feature_length(
        self,
        hole_k_shape: int,
        ablation_cfg: RobustHoleConfig,
    ) -> int:
        base_length = hole_k_shape + 6

        pyramid_length = sum(
            level * level
            for level in ablation_cfg.pyramid_levels
        )

        return base_length + pyramid_length

    def compute_hole_feats_for_image(
        self, 
        hole_k_shape: int =12,
        hole_cfg: RobustHoleConfig | None = None,
    ):
        self.bw = self.preprocess()

        # outer spline
        contours = self.extract_contours()
        if len(contours) == 0:
            return []  # no digit, no holes

        outer_contour = contours[0]
        outer_res = self.resample_polyline(
            outer_contour, 
            n=256, 
            closed=True,
        )
        outer_tck = self.fit_periodic_spline(
            outer_res, 
            s=1.0, 
            per=True,
        )
        outer_pts = self.sample_spline(
            outer_tck, 
            n=256,
        )
        
        c_outer, s_outer, R_xy = self.canonical_frame_with_signfix(outer_pts)
        
        com_y, com_x = self.foreground_com_canonical(
            c_outer, 
            s_outer, 
            R_xy,
        )
        
        detected_holes = self.extract_hole_contours(hole_cfg)
        
        expected_shape_length = hole_k_shape
        expected_pyramid_length = sum(
            level * level for level in hole_cfg.pyramid_levels
        )
        expected_geometry_length = 6

        features = []
        for hc, candidate in detected_holes:
            # centroid
            h_res = self.resample_polyline(
                hc, 
                n=128, 
                closed=True,
            )
            h_tck = self.fit_periodic_spline(
                h_res, 
                s=0.5, 
                per=True,
            )
            h_pts = self.sample_spline(
                h_tck, 
                n=128,
            )
            hole_centroid = candidate.centroid_yx

            canon = self.canonicalize_point_yx(
                hole_centroid, 
                c_outer, 
                s_outer, 
                R_xy,
            )
            py, px = canon  # RTS-invariant coordinates of the hole centroid
            
            hole_feature_len = self.hole_feature_length(
                hole_k_shape=hole_k_shape,
                ablation_cfg=hole_cfg,
            )

            f_shape = self.hole_feature_vector(
                hc, 
                k=hole_k_shape,
            )  # 12
            
            pers = (
                candidate.scale_count/max(len(hole_cfg.closing_radii), 1)
                if hole_cfg is not None
                else 1.0
            )
            
            if hole_cfg.use_reliability_features:
                enclosure = candidate.enclosure_score
                persistence = pers
            else:
                enclosure = 0.0
                persistence = 0.0
                
            # Spatial-pyramid membership.
            pyramid_length = sum(
                level * level
                for level in hole_cfg.pyramid_levels
            )

            if hole_cfg.use_spatial_pyramid:
                pyramid_features = self.spatial_pyramid_membership(
                    point_yx=hole_centroid,
                    outer_points_yx=outer_pts,
                    levels=hole_cfg.pyramid_levels,
                )
            else:
                # Keep dimensions identical across ablation variants.
                pyramid_features = np.zeros(
                    pyramid_length,
                    dtype=np.float64,
                )
            f_shape = np.asarray(f_shape, dtype=np.float32).reshape(-1)
        
            geometry_features = np.asarray(
                [
                    py,
                    px,
                    com_y,
                    com_x,
                    enclosure,
                    persistence,
                ],
                dtype=np.float32,
            )
            pyramid_features = np.asarray(
                pyramid_features,
                dtype=np.float32,
            ).reshape(-1)
            feature = np.concatenate(
                [f_shape, geometry_features, pyramid_features]
            ).astype(np.float32)

            
            if len(f_shape) != expected_shape_length:
                raise ValueError(
                    f"Shape descriptor: expected {expected_shape_length}, "
                    f"got {len(f_shape)}."
                )

            if len(geometry_features) != expected_geometry_length:
                raise ValueError(
                    f"Geometry descriptor: expected {expected_geometry_length}, "
                    f"got {len(geometry_features)}."
                )

            if len(pyramid_features) != expected_pyramid_length:
                raise ValueError(
                    f"Pyramid descriptor: expected {expected_pyramid_length}, "
                    f"got {len(pyramid_features)} for levels "
                    f"{hole_cfg.pyramid_levels}."
                )
            features.append(feature)

        return features
        
    
