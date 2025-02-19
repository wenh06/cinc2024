"""
According to the organizers,

    Any rotation of the paper ECGs in the images is possible, but most are unlikely. The hidden validation and test sets are intended to be realistic, not adversarial or antagonistic.

Therefore, we will not consider rotated bounding boxes for the detection task for simplicity.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

__all__ = ["BBox", "RotatedBBox"]


@dataclass
class BBox:

    xmin: int
    ymin: int
    width: int
    height: int
    img_width: Optional[int] = None
    img_height: Optional[int] = None
    category_name: Optional[str] = None
    category_id: Optional[int] = None

    def __post_init__(self):
        self.xmin = int(self.xmin)
        self.ymin = int(self.ymin)
        self.width = int(self.width)
        self.height = int(self.height)
        self.center = (self.xmin + self.width // 2, self.ymin + self.height // 2)

    @property
    def left(self) -> int:
        return self.xmin

    @property
    def right(self) -> int:
        return self.xmin + self.width

    @property
    def top(self) -> int:
        return self.ymin

    @property
    def bottom(self) -> int:
        return self.ymin + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def __repr__(self) -> str:
        return f"BBox(xmin={self.xmin}, ymin={self.ymin}, width={self.width}, height={self.height})"

    __str__ = __repr__

    def __eq__(self, other) -> bool:
        return self.xmin == other.xmin and self.ymin == other.ymin and self.width == other.width and self.height == other.height

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __contains__(self, other) -> bool:
        return self.left <= other.left and self.right >= other.right and self.top <= other.top and self.bottom >= other.bottom

    def intersection(self, other: "BBox") -> "BBox":
        x = max(self.left, other.left)
        y = max(self.top, other.top)
        width = min(self.right, other.right) - x
        height = min(self.bottom, other.bottom) - y
        if width <= 0 or height <= 0:
            return None
        return BBox(x, y, width, height)

    def union(self, other: "BBox") -> "BBox":
        x = min(self.left, other.left)
        y = min(self.top, other.top)
        width = max(self.right, other.right) - x
        height = max(self.bottom, other.bottom) - y
        return BBox(x, y, width, height)

    def iou(self, other: "BBox") -> float:
        intersection = self.intersection(other)
        if intersection is None:
            return 0
        union = self.union(other)
        return intersection.width * intersection.height / union.width / union.height

    def to_coco_format(self) -> dict:
        return {
            "bbox": [self.xmin, self.ymin, self.width, self.height],
            "category_id": self.category_id,
            "category_name": self.category_name,
            "area": self.area,
        }

    def to_voc_format(self) -> dict:
        return {
            "bbox": [self.xmin, self.ymin, self.right, self.bottom],
            "category_id": self.category_id,
            "category_name": self.category_name,
            "area": self.area,
        }

    def to_yolo_format(self) -> dict:
        x_center = (self.left + self.right) / 2 / self.img_width
        y_center = (self.top + self.bottom) / 2 / self.img_height
        width = self.width / self.img_width
        height = self.height / self.img_height
        return {
            "bbox": [x_center, y_center, width, height],
            "category_id": self.category_id,
            "category_name": self.category_name,
            "area": self.area,
        }

    def to_albumentations_format(self) -> dict:
        return {
            "bbox": [
                self.xmin / self.img_width,
                self.ymin / self.img_height,
                self.right / self.img_width,
                self.bottom / self.img_height,
            ],
            "category_id": self.category_id,
            "category_name": self.category_name,
            "area": self.area,
        }

    def to_matplotlib_format(self) -> dict:
        return {
            "bbox": [self.left, self.img_height - self.bottom, self.right, self.img_height - self.top],
            "category_id": self.category_id,
            "category_name": self.category_name,
            "area": self.area,
        }

    def asdict(self) -> dict:
        return asdict(self)


@dataclass
class RotatedBBox:

    corners: list
    img_width: Optional[int] = None
    img_height: Optional[int] = None
    category_name: Optional[str] = None
    category_id: Optional[int] = None

    def __post_init__(self):
        if len(self.corners) != 4:
            raise ValueError("RotatedBBox should have 4 corners")
        # self.corners = np.array(self.corners).astype(int)
        self.corners = np.array(self.corners)
        assert self.corners.shape == (4, 2), "RotatedBBox should have 4 corners"
        assert (
            (self.corners >= 0).all()
            and (self.corners[:, 0] <= self.img_width).all()
            and (self.corners[:, 1] <= self.img_height).all()
        ), f"RotatedBBox({self.corners.tolist()}) should be within the image"
        # check if it is a degenrate bbox (any 3 points are collinear)
        for idx in range(4):
            mat = np.delete(self.corners, idx, axis=0)
            if np.linalg.det(np.diff(mat, axis=0)) == 0:
                raise ValueError(f"RotatedBBox({self.corners.tolist()}) should not be degenerate")

        self.center = self.corners.mean(axis=0)
        # choose the corner closest to the origin as the starting point
        idx = np.argmin(np.linalg.norm(self.corners, axis=1))
        # sort the rest of the corners in clockwise order
        vecs = self.center - self.corners
        vecs = vecs[:, 0] + 1j * vecs[:, 1]
        angles = np.angle(vecs)
        angles = (angles - angles[idx]) % (2 * np.pi)
        indices = np.argsort(angles)
        self.corners = self.corners[indices]
        # check if the bbox is rectangle
        vecs = self.corners - np.roll(self.corners, 1, axis=0)
        if not all([np.allclose(np.dot(vecs[i], vecs[(i + 1) % 4]), 0) for i in range(4)]):
            raise ValueError(f"RotatedBBox({self.corners.tolist()}) is not a rectangle")
        self._corners = self.corners.astype(int)

        self.bbox = BBox(
            self.corners[:, 0].min(),
            self.corners[:, 1].min(),
            self.corners[:, 0].max() - self.corners[:, 0].min(),
            self.corners[:, 1].max() - self.corners[:, 1].min(),
            self.img_width,
            self.img_height,
            self.category_name,
        )

    @property
    def vertices(self) -> list:
        return self.corners.tolist()

    @property
    def area(self) -> float:
        return np.linalg.norm(np.cross(self.corners[1] - self.corners[0], self.corners[2] - self.corners[0]))

    def __repr__(self) -> str:
        if np.issubdtype(self.corners.dtype, np.integer):
            return f"RotatedBBox(corners={self.corners.tolist()})"
        return f"RotatedBBox(corners={np.round(self.corners, 2).tolist()})"

    __str__ = __repr__

    def __eq__(self, other) -> bool:
        return np.all(self.corners == other.corners)

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __contains__(self, other: "RotatedBBox") -> bool:
        # check by orient test on the 4 corners of the other bbox
        for corner in other.corners:
            sign = set()
            for a, b in zip(self.corners, np.roll(self.corners, 1, axis=0)):
                s = np.sign((corner[0] - a[0]) * (b[1] - a[1]) - (corner[1] - a[1]) * (b[0] - a[0]))
                if s == 0:
                    # the current corner is on the edge of this bbox
                    break
                sign.add(s)
                if len(sign) > 1:
                    return False
        return True

    # def intersection(self, other: "RotatedBBox") -> "RotatedBBox":
    #     raise NotImplementedError

    # def union(self, other: "RotatedBBox") -> "RotatedBBox":
    #     raise NotImplementedError

    def iou(self, other: "RotatedBBox") -> float:
        raise NotImplementedError

    def rotate(self, angle: float, center: Optional[tuple] = None) -> "RotatedBBox":
        if center is None:
            center = self.center
        center = np.array(center)
        angle = np.deg2rad(angle)
        rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        corners = np.dot(rot_mat, (self.corners - center).T).T + center
        return RotatedBBox(corners, self.img_width, self.img_height, self.category_name)

    def to_coco_format(self) -> dict:
        return self.bbox.to_coco_format()

    def to_voc_format(self) -> dict:
        return self.bbox.to_voc_format()

    def to_yolo_format(self) -> dict:
        return self.bbox.to_yolo_format()

    def to_albumentations_format(self) -> dict:
        return self.bbox.to_albumentations_format()

    def to_matplotlib_format(self) -> dict:
        return self.bbox.to_matplotlib_format()

    def asdict(self) -> dict:
        return asdict(self)
