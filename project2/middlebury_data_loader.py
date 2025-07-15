import os
from pathlib import Path
from functools import lru_cache

import imageio.v3 as iio


def read_pfm(path):
    raise NotImplementedError("PFM reader not implemented")


def read_image(path):
    return iio.imread(path)


class StereoScene:
    """
    Lazy loader for a single scene-version (perfect/imperfect).
    Access .left, .right, .disp to load on demand.
    """
    __slots__ = ("_calib_path", "_left_path", "_right_path", "_disp_left_path", "_disp_right_path")

    def __init__(self, calib_path, left_path, right_path, disp_left_path, disp_right_path):
        self._calib_path = calib_path
        self._left_path = left_path
        self._right_path = right_path
        self._disp_left_path = disp_left_path
        self._disp_right_path = disp_right_path

    @property
    @lru_cache(maxsize=1)
    def calib(self):
        with open(self._calib_path) as f:
            return f.read()

    @property
    @lru_cache(maxsize=1)
    def left(self):
        return read_image(self._left_path)

    @property
    @lru_cache(maxsize=1)
    def right(self):
        return read_image(self._right_path)

    @property
    @lru_cache(maxsize=1)
    def disp_left(self):
        if self._disp_left_path and self._disp_left_path.exists():
            return read_pfm(self._disp_left_path)
        return None

    @property
    @lru_cache(maxsize=1)
    def disp_right(self):
        if self._disp_right_path and self._disp_right_path.exists():
            return read_pfm(self._disp_right_path)
        return None


def MiddleburyStereoDataset(base_dir=os.path.expanduser("~/data/middlebury-stereo-2014")):
    """
    Scans base_dir for scene-version folders and returns a nested dict:
      { scene_name: {
          'perfect': StereoScene(...),
          'imperfect': StereoScene(...),
        }, ... }
    Images are only read when .left/.right/.disp is accessed.
    """
    data = {}
    base = Path(base_dir)
    for entry in base.iterdir():
        if not entry.is_dir() or "-" not in entry.name: continue
        scene, version = entry.name.rsplit("-", 1)
        calib_path = entry / "calib.txt"
        left_path = entry / "im0.png"
        right_path = entry / "im1.png"
        disp_left_path = entry / "disp0.pfm"
        disp_right_path = entry / "disp1.pfm"

        scene_dict = data.setdefault(scene, {})
        scene_dict[version] = StereoScene(
            calib_path,
            left_path,
            right_path,
            disp_left_path if disp_left_path.exists() else None,
            disp_right_path if disp_right_path.exists() else None
        )
    return data
