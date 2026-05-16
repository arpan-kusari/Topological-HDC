from scipy.interpolate import splprep, splev
from skimage.filters import threshold_otsu, threshold_local, gaussian
from skimage.morphology import closing, opening, remove_small_objects, disk
from skimage.measure import label, regionprops, find_contours
from mahotas.features import zernike_moments
from skimage.transform import resize
from skimage.feature import hog
import numpy as np

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

    # ----------------------------
    # Holes: contours -> RTS-invariant feature vector
    # ----------------------------
    def hole_masks(self):
        inv = ~self.bw
        lab = label(inv)
        H, W = self.bw.shape
        holes = []
        for r in regionprops(lab):
            coords = r.coords
            touches_border = np.any(
                (coords[:, 0] == 0) | (coords[:, 0] == H-1) |
                (coords[:, 1] == 0) | (coords[:, 1] == W-1)
            )
            if not touches_border:
                m = np.zeros((H, W), dtype=bool)
                m[coords[:, 0], coords[:, 1]] = True
                holes.append(m)
        return holes

    def extract_hole_contours(self):
        hole_cs = []
        for hm in self.hole_masks():
            cs = find_contours(hm.astype(float), level=0.5)
            if len(cs) > 0:
                cs = sorted(cs, key=lambda c: c.shape[0], reverse=True)
                hole_cs.append(cs[0])
        hole_cs = sorted(hole_cs, key=lambda c: c.shape[0], reverse=True)
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

    def compute_hole_feats_for_image(self, hole_k_shape=12):
        self.bw = self.preprocess()

        # outer spline
        contours = self.extract_contours()
        if len(contours) == 0:
            return []  # no digit, no holes

        outer_contour = contours[0]
        outer_res = self.resample_polyline(outer_contour, n=256, closed=True)
        outer_tck = self.fit_periodic_spline(outer_res, s=1.0, per=True)
        outer_pts = self.sample_spline(outer_tck, n=256)

        c_outer = outer_pts.mean(axis=0)
        s_outer = np.sqrt(np.mean(np.sum((outer_pts - c_outer) ** 2, axis=1))) + 1e-12

        hole_cs = self.extract_hole_contours()
        feats = []
        for hc in hole_cs:
            # centroid
            h_res = self.resample_polyline(hc, n=128, closed=True)
            h_tck = self.fit_periodic_spline(h_res, s=0.5, per=True)
            h_pts = self.sample_spline(h_tck, n=128)
            hole_centroid = h_pts.mean(axis=0)

            t, dist = self.hole_relative_to_outer(outer_pts, hole_centroid)
            dist_norm = dist / s_outer

            c_outer, s_outer, R_xy = self.canonical_frame_with_signfix(outer_pts)

            canon = self.canonicalize_point_yx(hole_centroid, c_outer, s_outer, R_xy)
            py, px = canon  # RTS-invariant coordinates of the hole centroid

            f_shape = self.hole_feature_vector(hc, k=hole_k_shape)  # 12
            # tail = tail_ratio_feature(bw, outer_pts, hole_centroid)
            com_yx = self.foreground_com_canonical(c_outer, s_outer, R_xy)
            com_y, com_x = com_yx
            f = np.hstack([f_shape, [py, px, com_y, com_x]]).astype(np.float64)  # 14
            feats.append(f)
        return feats
        
    
