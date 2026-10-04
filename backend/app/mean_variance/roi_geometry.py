"""ROI geometry in pixel-boundary coordinates; statistics never use resampled pixels."""
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

EPSILON = 1e-9


def rotation(angle):
    radians = np.deg2rad(angle)
    values = np.array([np.cos(radians), np.sin(radians)])
    # Exact quarter-turns must stay exact on pixel boundaries.
    return np.where(np.abs(values - np.rint(values)) < 1e-14, np.rint(values), values)


class RotatedRectangle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[2]
    center_x: float = Field(allow_inf_nan=False)
    center_y: float = Field(allow_inf_nan=False)
    width: int = Field(ge=1, strict=True)
    height: int = Field(ge=1, strict=True)
    angle_degrees: float = Field(default=0, allow_inf_nan=False)

    @field_validator("angle_degrees")
    @classmethod
    def normalize_angle(cls, value):
        return (value + 180) % 360 - 180

    def corners(self):
        c, s = rotation(self.angle_degrees)
        local = np.array([[-self.width, -self.height], [self.width, -self.height],
                          [self.width, self.height], [-self.width, self.height]], dtype=np.float64) / 2
        return local @ np.array([[c, s], [-s, c]]) + [self.center_x, self.center_y]

    def validate_bounds(self, width, height):
        corners = self.corners()
        if (corners < -EPSILON).any() or (corners > np.array([width, height]) + EPSILON).any():
            raise ValueError("Die gedrehte ROI muss vollständig innerhalb des Bildes liegen.")


def oriented_roi(roi):
    if isinstance(roi, RotatedRectangle):
        return roi
    return RotatedRectangle(version=2, center_x=roi.x + roi.width / 2,
                            center_y=roi.y + roi.height / 2, width=roi.width, height=roi.height, angle_degrees=0)


def selection_mask(roi, width, height):
    roi = oriented_roi(roi)
    roi.validate_bounds(width, height)
    c, s = rotation(roi.angle_degrees)
    y, x = np.ogrid[:height, :width]
    dx, dy = x + .5 - roi.center_x, y + .5 - roi.center_y
    # Local left/top included, right/bottom excluded, as with the old slice.
    u, v = c * dx + s * dy, -s * dx + c * dy
    for values, half in ((u, roi.width / 2), (v, roi.height / 2)):
        values[np.abs(values - half) < EPSILON] = half
        values[np.abs(values + half) < EPSILON] = -half
    mask = (u >= -roi.width / 2) & (u < roi.width / 2) & (v >= -roi.height / 2) & (v < roi.height / 2)
    if not mask.any():
        raise ValueError("Die ROI enthält keinen Originalpixelmittelpunkt. Bitte vergrößern oder verschieben.")
    return mask


def aligned_crop(image, roi):
    """Inverse map output pixel centers to nearest original centers, no mixing."""
    roi = oriented_roi(roi)
    roi.validate_bounds(image.shape[1], image.shape[0])
    c, s = rotation(roi.angle_degrees)
    y, x = np.ogrid[:roi.height, :roi.width]
    u, v = x + .5 - roi.width / 2, y + .5 - roi.height / 2
    source_x = roi.center_x + c * u - s * v
    source_y = roi.center_y + s * u + c * v
    # Source centers are i+.5, so nearest index is floor(boundary coordinate).
    # At exact ties select the greater index. Snap floating-point boundary noise.
    source_x = np.where(np.abs(source_x - np.rint(source_x)) < EPSILON, np.rint(source_x), source_x)
    source_y = np.where(np.abs(source_y - np.rint(source_y)) < EPSILON, np.rint(source_y), source_y)
    ix = np.floor(source_x).astype(np.int64)
    iy = np.floor(source_y).astype(np.int64)
    return image[iy, ix]
