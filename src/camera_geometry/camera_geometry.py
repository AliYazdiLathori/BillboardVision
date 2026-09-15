"""
Camera geometry utilities for BillboardVision.

Implements a pinhole-camera + flat-ground model that converts between:
  - image pixel coordinates (px, py), origin top-left, y increasing downward
  - world ground-plane coordinates (X, Z)

World coordinate convention
----------------------------
Right-handed, Y-up, anchored to the billboard (not the camera):
  X : right      (increases to the billboard's right; X = 0 is its
                   horizontal center)
  Y : up         (height above the ground)
  Z : forward    (increases with distance away from the billboard's face,
                   into the scene; Z = 0 is the billboard's face plane)

The ground plane is Y = 0 (GROUND_FLAT assumption in BILLBOARD_CONFIG).

BILLBOARD_CONFIG defines the camera's position relative to the billboard's
top-middle edge (CAMERA_HORIZONTAL_OFFSET, CAMERA_VERTICAL_OFFSET), so the
camera's world position is derived as:
    (CAMERA_HORIZONTAL_OFFSET, TOP_HEIGHT + CAMERA_VERTICAL_OFFSET, 0)
i.e. co-planar with the billboard's face (no separate depth offset is
modeled), looking, at zero pitch/yaw/roll, straight down the +Z axis with
+Y as "up" in the image.

Sign conventions (verify against real footage before trusting results —
see the self-test at the bottom of this file for a starting point):
  - CAMERA_PITCH: positive tilts the camera up (away from the ground),
    negative tilts it down (toward the ground). BILLBOARD_CONFIG's
    CAMERA_PITCH = -7.5 means "tilted slightly downward", which matches
    a camera mounted above head height looking down at a crowd.
  - CAMERA_YAW: positive pans the camera's view to the right.
  - CAMERA_ROLL: positive rotates the image clockwise around the optical
    axis (standard right-hand rotation of the pixel grid).

If any of these look backwards once tested against a real frame, flip the
sign of the corresponding config value — the math below is internally
consistent either way.

FOV_DEGREES is treated as the horizontal field of view. Vertical FOV is
derived implicitly from the frame's aspect ratio under a square-pixel
(fx == fy) assumption, which is standard for an uncalibrated pinhole model.
"""

from src.config.camera_billboard import BILLBOARD_CONFIG

import numpy as np


class CameraGeometry:
    """
    Precomputes camera intrinsics + orientation from BILLBOARD_CONFIG and
    a given frame size, then converts between pixel and ground-plane
    coordinates.

    A new instance should be created whenever the frame size changes
    (e.g. once per video, after the first frame is read).
    """

    def __init__(self, config = BILLBOARD_CONFIG , frame_width = 3840, frame_height=2160):
        """
        Args:
            config: BILLBOARD_CONFIG dict (see src/config/camera_billboard.py)
            frame_width: width in pixels of the frame these pixel
                coordinates are expressed in
            frame_height: height in pixels of the frame these pixel
                coordinates are expressed in
        """

        self.frame_width = frame_width
        self.frame_height = frame_height

        # Camera position is defined relative to the billboard's top-middle
        # edge (see BILLBOARD_CONFIG). The billboard's horizontal center is
        # world X = 0, and its face plane is world Z = 0, so:
        self.camera_horizontal_offset = config["CAMERA_HORIZONTAL_OFFSET"]
        self.camera_height = config["TOP_HEIGHT"] + config["CAMERA_VERTICAL_OFFSET"]

        self.pitch_rad = np.radians(config["CAMERA_PITCH"])
        self.yaw_rad = np.radians(config["CAMERA_YAW"])
        self.roll_rad = np.radians(config["CAMERA_ROLL"])

        self.fov_degrees = config["FOV_DEGREES"]

        # --- Intrinsics ---
        # Horizontal FOV -> focal length in pixels (square-pixel assumption)
        self.fx = (frame_width / 2.0) / np.tan(np.radians(self.fov_degrees / 2.0))
        self.fy = self.fx
        self.cx = frame_width / 2.0
        self.cy = frame_height / 2.0

        # --- Extrinsics ---
        # Camera position in world space
        self.camera_position = np.array(
            [self.camera_horizontal_offset, self.camera_height, 0.0]
        )

        # Camera-to-world rotation: pitch (tilt) applied first, then yaw (pan).
        # Roll is handled separately, directly on pixel offsets, since it
        # rotates the image plane rather than the camera's aim direction.
        self._R = self._yaw_matrix(self.yaw_rad) @ self._pitch_matrix(self.pitch_rad)

    # ------------------------------------------------------------------
    # Rotation matrices
    # ------------------------------------------------------------------

    @staticmethod
    def _pitch_matrix(pitch_rad):
        """
        Rotation about the X (right) axis such that the base forward
        vector (0, 0, 1) maps to (0, sin(pitch), cos(pitch)) -- i.e.
        positive pitch tilts the view upward (+Y).
        """
        c, s = np.cos(pitch_rad), np.sin(pitch_rad)
        return np.array([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c],
        ])

    @staticmethod
    def _yaw_matrix(yaw_rad):
        """
        Rotation about the Y (up) axis such that positive yaw pans the
        view toward +X (the camera's right).
        """
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ])

    # ------------------------------------------------------------------
    # Pixel <-> camera-local ray
    # ------------------------------------------------------------------

    def _pixel_to_local_ray(self, px, py):
        """
        Convert a pixel coordinate into a direction vector expressed in
        the camera's own local frame (x=right, y=up, z=forward), before
        pitch/yaw are applied. Roll is applied here, as a rotation of the
        pixel offset around the image center.
        """

        dx = px - self.cx
        dy = py - self.cy

        # Apply roll to the pixel offset (rotate around image center)
        cos_r, sin_r = np.cos(self.roll_rad), np.sin(self.roll_rad)
        dx_r = dx * cos_r - dy * sin_r
        dy_r = dx * sin_r + dy * cos_r

        x_local = dx_r / self.fx
        y_local = -dy_r / self.fy  # image y is down; local y is up
        z_local = 1.0

        ray = np.array([x_local, y_local, z_local])
        return ray / np.linalg.norm(ray)

    def _local_ray_to_pixel(self, local_ray):
        """
        Inverse of _pixel_to_local_ray: given a direction in the camera's
        local frame (must have positive z, i.e. in front of the camera),
        return the (px, py) it projects to. Returns None if behind the
        camera.
        """

        x_local, y_local, z_local = local_ray

        if z_local <= 0:
            return None

        dx_r = self.fx * (x_local / z_local)
        dy_r = -self.fy * (y_local / z_local)

        # Undo roll
        cos_r, sin_r = np.cos(self.roll_rad), np.sin(self.roll_rad)
        dx = dx_r * cos_r + dy_r * sin_r
        dy = -dx_r * sin_r + dy_r * cos_r

        return (self.cx + dx, self.cy + dy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pixel_to_ground(self, px, py):
        """
        Back-project an image pixel onto the ground plane (Y = 0).

        Intended for pixels that represent a point actually touching the
        ground (e.g. a person's bounding-box bottom-center / feet), since
        the ray-ground intersection is only meaningful for rays that
        point below the horizon.

        Returns:
            (X, Z) world ground coordinates, or None if the pixel is
            above the horizon (ray points up/parallel and never meets
            the ground).
        """

        local_ray = self._pixel_to_local_ray(px, py)
        world_dir = self._R @ local_ray

        if world_dir[1] >= 0:
            # Ray points at or above the horizon -- no ground intersection
            # in front of the camera.
            return None

        t = -self.camera_position[1] / world_dir[1]

        if t < 0:
            return None

        world_point = self.camera_position + t * world_dir
        return (float(world_point[0]), float(world_point[2]))

    def ground_to_pixel(self, X, Z):
        """
        Project a world ground-plane point (X, Z, with Y = 0) into image
        pixel coordinates.

        Returns:
            (px, py), or None if the point is behind the camera.
        """

        world_point = np.array([X, 0.0, Z])
        relative = world_point - self.camera_position

        # Inverse rotation: R is orthonormal, so its inverse is its transpose
        local_ray = self._R.T @ relative

        return self._local_ray_to_pixel(local_ray)

if __name__ == "__main__":
    # Minimal self-test / sanity check.
    # Run directly: python -m src.utils.camera_geometry

    geo = CameraGeometry(BILLBOARD_CONFIG, frame_width=3840, frame_height=2160)

    # Round-trip check: pixel -> ground -> pixel should return (approximately)
    # the original pixel, for any pixel below the horizon.
    test_px, test_py = 1910, 2000  # near-bottom-center, should be below horizon
    ground = geo.pixel_to_ground(test_px, test_py)
    print(f"Pixel ({test_px}, {test_py}) -> ground {ground}")

    if ground is not None:
        back = geo.ground_to_pixel(*ground)
        print(f"Ground {ground} -> pixel {back}")
        assert back is not None
        assert abs(back[0] - test_px) < 1e-3
        assert abs(back[1] - test_py) < 1e-3
        print("Round-trip OK.")

    # A pixel near the top of the frame should be above the horizon
    # (no ground intersection) for a camera tilted only slightly downward.
    sky_result = geo.pixel_to_ground(1920, 50)
    print(f"Pixel (960, 50) -> ground {sky_result} (expected None if above horizon)")