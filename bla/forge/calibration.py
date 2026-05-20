"""BF-0.1 — camera calibration + table-coordinate transform.

The minimum harness that:
  - serializes intrinsics + extrinsics to disk (calibration.json),
  - projects world→image and image-plane→world,
  - reports reprojection error,
  - has both a REAL OpenCV-backed estimator (checkerboard intrinsics +
    solvePnP extrinsics) AND a MOCK estimator that uses a known
    synthetic camera so the rest of the BF-0 stack can be developed
    before hardware arrives.

Convention:
  - "world" frame = gantry frame (origin at workspace center; +z up;
    +x along gantry rail). Real-world coords in meters.
  - "image" frame = pixel coords (row, col) of the overhead camera.
  - Intrinsics format: standard OpenCV 3×3 K matrix + 1×5 distortion
    coefficients.
  - Extrinsics format: T_cam_to_world is a 4×4 rigid transform.
    project_world_to_image uses its inverse (T_world_to_cam).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ---------- dataclasses ----------
@dataclass
class CameraIntrinsics:
    """Standard OpenCV pinhole intrinsics."""
    camera_matrix: np.ndarray   # [3, 3]
    distortion_coeffs: np.ndarray   # [5]
    image_size_wh: tuple[int, int]  # (width, height) pixels

    def __post_init__(self):
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        self.distortion_coeffs = np.asarray(self.distortion_coeffs,
                                                  dtype=np.float64).reshape(-1)
        if self.camera_matrix.shape != (3, 3):
            raise ValueError(
                f"camera_matrix must be 3×3, got {self.camera_matrix.shape}")
        if self.distortion_coeffs.size not in (4, 5, 8, 12, 14):
            raise ValueError(
                f"distortion_coeffs has invalid size {self.distortion_coeffs.size}")

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])


@dataclass
class CameraExtrinsics:
    """Camera → world rigid transform as a 4×4 matrix.

    `T_cam_to_world @ [p_cam; 1]` = `[p_world; 1]`.
    """
    T_cam_to_world: np.ndarray   # [4, 4]

    def __post_init__(self):
        self.T_cam_to_world = np.asarray(self.T_cam_to_world, dtype=np.float64)
        if self.T_cam_to_world.shape != (4, 4):
            raise ValueError(
                f"T_cam_to_world must be 4×4, got {self.T_cam_to_world.shape}")

    @property
    def T_world_to_cam(self) -> np.ndarray:
        return np.linalg.inv(self.T_cam_to_world)


@dataclass
class CalibrationBundle:
    """Full calibration: intrinsics + extrinsics + provenance metadata."""
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    metadata: dict = field(default_factory=dict)
    # Provenance suggestions for metadata:
    #   - "source": "checkerboard" | "mock" | "synthetic_robosuite"
    #   - "intrinsics_reprojection_px": float
    #   - "extrinsics_reprojection_px": float
    #   - "n_calibration_images": int
    #   - "date_iso": str


# ---------- serialization ----------
def save_calibration(bundle: CalibrationBundle, path: str | Path) -> Path:
    """Write JSON with numpy arrays as nested lists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "intrinsics": {
            "camera_matrix": bundle.intrinsics.camera_matrix.tolist(),
            "distortion_coeffs": bundle.intrinsics.distortion_coeffs.tolist(),
            "image_size_wh": list(bundle.intrinsics.image_size_wh),
        },
        "extrinsics": {
            "T_cam_to_world": bundle.extrinsics.T_cam_to_world.tolist(),
        },
        "metadata": bundle.metadata,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_calibration(path: str | Path) -> CalibrationBundle:
    path = Path(path)
    d = json.loads(path.read_text())
    intr = CameraIntrinsics(
        camera_matrix=np.asarray(d["intrinsics"]["camera_matrix"]),
        distortion_coeffs=np.asarray(d["intrinsics"]["distortion_coeffs"]),
        image_size_wh=tuple(d["intrinsics"]["image_size_wh"]),
    )
    extr = CameraExtrinsics(
        T_cam_to_world=np.asarray(d["extrinsics"]["T_cam_to_world"]),
    )
    return CalibrationBundle(
        intrinsics=intr,
        extrinsics=extr,
        metadata=d.get("metadata", {}),
    )


# ---------- projection ----------
def project_world_to_image(
    world_points_xyz: np.ndarray,
    bundle: CalibrationBundle,
) -> np.ndarray:
    """Project world-frame 3D points to image pixel coordinates.

    world_points_xyz: [N, 3] or [3] in meters (gantry frame).
    Returns: [N, 2] or [2] pixel coords (x=col, y=row), float.

    Uses pinhole model with distortion. No fisheye support.
    """
    world_points = np.atleast_2d(np.asarray(world_points_xyz, dtype=np.float64))
    if world_points.shape[1] != 3:
        raise ValueError(f"world_points must be N×3, got {world_points.shape}")

    # World → camera
    T_world_to_cam = bundle.extrinsics.T_world_to_cam
    ones = np.ones((world_points.shape[0], 1))
    homog = np.hstack([world_points, ones])   # [N, 4]
    cam_points = (T_world_to_cam @ homog.T).T[:, :3]   # [N, 3]

    # Camera → image
    K = bundle.intrinsics.camera_matrix
    fx, fy, cx, cy = bundle.intrinsics.fx, bundle.intrinsics.fy, \
        bundle.intrinsics.cx, bundle.intrinsics.cy

    # Avoid div-by-zero behind camera; clip z to small positive
    z = np.clip(cam_points[:, 2], 1e-6, None)
    x_norm = cam_points[:, 0] / z
    y_norm = cam_points[:, 1] / z

    # Apply distortion (radial k1, k2; tangential p1, p2; k3)
    dist = bundle.intrinsics.distortion_coeffs
    k1 = dist[0] if dist.size > 0 else 0.0
    k2 = dist[1] if dist.size > 1 else 0.0
    p1 = dist[2] if dist.size > 2 else 0.0
    p2 = dist[3] if dist.size > 3 else 0.0
    k3 = dist[4] if dist.size > 4 else 0.0
    r2 = x_norm ** 2 + y_norm ** 2
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_dist = x_norm * radial + 2.0 * p1 * x_norm * y_norm \
        + p2 * (r2 + 2.0 * x_norm ** 2)
    y_dist = y_norm * radial + p1 * (r2 + 2.0 * y_norm ** 2) \
        + 2.0 * p2 * x_norm * y_norm

    px_x = fx * x_dist + cx
    px_y = fy * y_dist + cy
    pixels = np.stack([px_x, px_y], axis=-1)

    if world_points_xyz.ndim == 1:
        return pixels[0]
    return pixels


def project_image_to_world_plane(
    pixel_points_xy: np.ndarray,
    bundle: CalibrationBundle,
    world_plane_z: float = 0.0,
) -> np.ndarray:
    """Inverse projection: pixel (x, y) → world (x, y, z=world_plane_z).

    Assumes the point lies on the horizontal world plane at z = world_plane_z
    (the table top). This is the standard table-coordinate-transform use case
    for BLA-Forge.

    pixel_points_xy: [N, 2] or [2] in pixels.
    Returns: [N, 3] or [3] in world frame.

    NOTE: ignores distortion in inverse direction. For real-world use with
    significant distortion, undistort the pixels first via cv2.undistortPoints.
    """
    pix = np.atleast_2d(np.asarray(pixel_points_xy, dtype=np.float64))
    if pix.shape[1] != 2:
        raise ValueError(f"pixel_points must be N×2, got {pix.shape}")

    K_inv = np.linalg.inv(bundle.intrinsics.camera_matrix)
    homog_px = np.hstack([pix, np.ones((pix.shape[0], 1))])   # [N, 3]
    rays_cam = (K_inv @ homog_px.T).T  # [N, 3] (directions in cam frame)

    # Transform ray origins + directions to world frame
    T_c_w = bundle.extrinsics.T_cam_to_world
    R = T_c_w[:3, :3]
    t = T_c_w[:3, 3]
    cam_origin_world = t   # camera center in world frame
    rays_world = (R @ rays_cam.T).T   # [N, 3] directions in world

    # Intersect each ray with z = world_plane_z
    # P(s) = cam_origin_world + s * rays_world
    # Solve P_z(s) = world_plane_z:
    #   s = (world_plane_z - cam_origin_world[2]) / rays_world[:, 2]
    rays_z = rays_world[:, 2]
    # Behind camera or parallel-to-plane → mark with NaN
    valid = np.abs(rays_z) > 1e-9
    s = np.full_like(rays_z, np.nan)
    s[valid] = (world_plane_z - cam_origin_world[2]) / rays_z[valid]
    world_points = cam_origin_world[None, :] + s[:, None] * rays_world

    if pixel_points_xy.ndim == 1:
        return world_points[0]
    return world_points


def reprojection_error(
    world_points_xyz: np.ndarray,
    measured_pixels_xy: np.ndarray,
    bundle: CalibrationBundle,
) -> float:
    """Mean pixel reprojection error between projected world points
    and their measured pixel locations.

    Used as the calibration sanity gate (≤ 2 px target per BF spec §4.4).
    """
    projected = project_world_to_image(world_points_xyz, bundle)
    projected = np.atleast_2d(projected)
    measured = np.atleast_2d(measured_pixels_xy)
    if projected.shape != measured.shape:
        raise ValueError(
            f"shape mismatch: projected {projected.shape}, "
            f"measured {measured.shape}")
    diffs = projected - measured
    return float(np.linalg.norm(diffs, axis=-1).mean())


# ---------- estimators ----------
def estimate_intrinsics_from_checkerboard(
    images: list[np.ndarray],
    pattern_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
) -> CameraIntrinsics:
    """Real-camera intrinsics estimation via OpenCV.

    images: list of BGR (or grayscale) numpy arrays, all same size.
    pattern_size: inner-corner count of the checkerboard (cols, rows).
    square_size_m: physical square size in meters.

    Returns: CameraIntrinsics. Caller should check the per-image
    reprojection error stored in metadata if exposed; this minimal
    harness returns just the K + dist.
    """
    try:
        import cv2  # opencv-python
    except ImportError as e:
        raise RuntimeError(
            "estimate_intrinsics_from_checkerboard requires opencv-python; "
            "install with `pip install opencv-python` or use mock_calibration."
        ) from e
    if len(images) == 0:
        raise ValueError("Need ≥1 checkerboard image")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) * square_size_m

    obj_points, img_points = [], []
    img_shape_wh = None
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if img_shape_wh is None:
            img_shape_wh = (gray.shape[1], gray.shape[0])
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not ok:
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                      30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                                  criteria)
        obj_points.append(objp)
        img_points.append(corners_refined)

    if len(obj_points) == 0:
        raise RuntimeError(
            "No checkerboard found in any provided image. Check pattern_size.")

    ret, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape_wh, None, None)
    return CameraIntrinsics(
        camera_matrix=K,
        distortion_coeffs=dist.reshape(-1),
        image_size_wh=img_shape_wh,
    )


def estimate_extrinsics_from_corners(
    world_corners_xyz: np.ndarray,
    image_corners_xy: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> CameraExtrinsics:
    """Estimate T_cam_to_world from 4+ corresponding 3D world ↔ 2D image
    fiducial points via solvePnP.

    Per BF spec §4.3 — "Move gantry to 4-corner + center fiducial positions;
    click fiducial centers in image; solvePnP for rigid transform."
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            "estimate_extrinsics_from_corners requires opencv-python."
        ) from e
    world_pts = np.asarray(world_corners_xyz, dtype=np.float32).reshape(-1, 1, 3)
    img_pts = np.asarray(image_corners_xy, dtype=np.float32).reshape(-1, 1, 2)
    if world_pts.shape[0] != img_pts.shape[0]:
        raise ValueError("world/image point count mismatch")
    if world_pts.shape[0] < 4:
        raise ValueError("solvePnP needs ≥ 4 correspondences")

    ok, rvec, tvec = cv2.solvePnP(
        world_pts, img_pts,
        intrinsics.camera_matrix, intrinsics.distortion_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("cv2.solvePnP failed")
    # solvePnP returns world → camera; invert for camera → world
    R_w_to_c, _ = cv2.Rodrigues(rvec)
    T_w_to_c = np.eye(4)
    T_w_to_c[:3, :3] = R_w_to_c
    T_w_to_c[:3, 3] = tvec.reshape(-1)
    T_c_to_w = np.linalg.inv(T_w_to_c)
    return CameraExtrinsics(T_cam_to_world=T_c_to_w)


# ---------- mock calibration ----------
def mock_calibration(
    image_size_wh: tuple[int, int] = (640, 480),
    fov_deg: float = 60.0,
    camera_height_m: float = 0.80,
    camera_xy_world: tuple[float, float] = (0.0, 0.0),
    camera_tilt_deg: float = 90.0,
) -> CalibrationBundle:
    """Synthetic calibration for development before hardware arrives.

    Default: a 60° FOV pinhole camera mounted 80 cm directly above the
    workspace origin, looking straight down (camera_tilt_deg=90 = optical
    axis = -z world). No lens distortion.

    This gives the BF-0 stack a known-correct calibration to develop
    against; the same JSON shape will be produced by checkerboard +
    solvePnP on real hardware.
    """
    width, height = image_size_wh
    # Pinhole: fx = width / (2 * tan(hfov/2)); assume square pixels.
    hfov_rad = np.deg2rad(fov_deg)
    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx   # square pixels
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    intr = CameraIntrinsics(K, dist, image_size_wh)

    # Camera pose: hovering above workspace.
    # Construct T_cam_to_world:
    #   camera_tilt_deg = 90 → optical axis (camera +z) points to -z world
    #   camera up (cam -y) → +y world (image y-down convention)
    tilt = np.deg2rad(camera_tilt_deg)
    # World axes:
    #   x_world = right
    #   y_world = forward
    #   z_world = up
    # When camera looks straight down (tilt=90 from horizontal), its +z
    # is the -world_z direction.
    # For a tilted-down camera by angle tilt from horizontal (about world x-axis):
    #   cam_x = world_x (right stays right)
    #   cam_y = world_y * cos(tilt) - world_z * sin(tilt) ≈ -world_z when tilt≈90
    #   cam_z = -world_y * sin(tilt) - world_z * cos(tilt)  (pointing into scene)
    # We use a simple rotation about world_x by (-tilt) for camera-pointing.
    # For tilt = 90: cam_z = -world_z (looking straight down).
    Rx = np.array([
        [1, 0,             0],
        [0, np.cos(-tilt + np.pi / 2.0), -np.sin(-tilt + np.pi / 2.0)],
        [0, np.sin(-tilt + np.pi / 2.0),  np.cos(-tilt + np.pi / 2.0)],
    ], dtype=np.float64)
    # When tilt=90: -tilt + pi/2 = -pi/2 → Rx flips y and z appropriately so
    # cam_z (col 3 of Rx, taken as col 2 here in row-major) points in -world_z.
    # To make this easier, just construct R explicitly for tilt=90 down-looking:
    if abs(tilt - np.pi / 2.0) < 1e-6:
        # camera looking straight down: cam frame axes in world frame
        #   cam_x = +world_x
        #   cam_y = -world_y   (camera image y-down convention; world y-up)
        #   cam_z = -world_z
        Rx = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ], dtype=np.float64)

    T_c_w = np.eye(4)
    T_c_w[:3, :3] = Rx
    T_c_w[:3, 3] = [camera_xy_world[0], camera_xy_world[1], camera_height_m]
    extr = CameraExtrinsics(T_c_w)

    return CalibrationBundle(
        intrinsics=intr,
        extrinsics=extr,
        metadata={
            "source": "mock",
            "fov_deg": fov_deg,
            "camera_height_m": camera_height_m,
            "camera_xy_world": list(camera_xy_world),
            "camera_tilt_deg": camera_tilt_deg,
            "notes": ("Synthetic calibration for development before hardware. "
                       "Pinhole, no distortion, top-down view."),
        },
    )
