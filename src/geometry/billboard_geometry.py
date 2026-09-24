"""
Billboard world-position geometry for BillboardVision.

Derives the billboard's position and extent in the same world coordinate
system used by camera_geometry.py:
  X : right   (X = 0 is the billboard's horizontal center)
  Y : up      (height above the ground)
  Z : forward (Z = 0 is the billboard's own face plane)

See src/utils/camera_geometry.py for the full coordinate convention this
module shares.

Note: WIDTH and HEIGHT are both given directly in BILLBOARD_CONFIG, but
HEIGHT is also implied by TOP_HEIGHT - BOTTOM_HEIGHT (currently 6.0 - 3.0
= 3.0, matching HEIGHT = 3.0). This module treats BOTTOM_HEIGHT/TOP_HEIGHT
as the source of truth for vertical extent, since they also fix the
billboard's placement above the ground, not just its size -- so HEIGHT
itself is never read here. If HEIGHT and (TOP_HEIGHT - BOTTOM_HEIGHT) are
ever edited out of sync, this module will silently use the latter and
ignore HEIGHT, the same way CAMERA_HEIGHT was dropped in favor of
TOP_HEIGHT + CAMERA_VERTICAL_OFFSET in camera_geometry.py.
"""

from src.config.camera_billboard import BILLBOARD_CONFIG


def get_billboard_center(config):
    """
    Returns the billboard's center point in world coordinates: (X, Y, Z).
    """
    x = 0.0
    y = (config["BOTTOM_HEIGHT"] + config["TOP_HEIGHT"]) / 2.0
    z = 0.0
    return (x, y, z)


def get_billboard_corners(config):
    """
    Returns the billboard's four corners in world coordinates, as a dict:
    {"top_left", "top_right", "bottom_left", "bottom_right"}, each an
    (X, Y, Z) tuple.

    Useful later for projecting the billboard's outline into an image
    (via CameraGeometry.ground_to_pixel-style logic) or for computing
    the distance from a person to the billboard's nearest edge rather
    than just its center.
    """
    half_width = config["WIDTH"] / 2.0
    bottom = config["BOTTOM_HEIGHT"]
    top = config["TOP_HEIGHT"]

    return {
        "top_left": (-half_width, top, 0.0),
        "top_right": (half_width, top, 0.0),
        "bottom_left": (-half_width, bottom, 0.0),
        "bottom_right": (half_width, bottom, 0.0),
    }


if __name__ == "__main__":
    # Minimal self-test / sanity check.
    # Run directly: python -m src.utils.billboard_geometry

    center = get_billboard_center(BILLBOARD_CONFIG)
    print(f"Billboard center: {center}")
    assert center == (0.0, 4.5, 0.0)

    corners = get_billboard_corners(BILLBOARD_CONFIG)
    for name, point in corners.items():
        print(f"  {name}: {point}")

    assert corners["top_left"][0] == -3.0
    assert corners["top_right"][0] == 3.0
    assert corners["bottom_left"][1] == 3.0
    assert corners["top_left"][1] == 6.0
    print("Self-test OK.")