"""Test point cloud generation from stereo images."""

import argparse
from collections.abc import Generator

import pytest

from common_utils import config
from pointcloud_generation.pointcloud_generation import PointCloudGenerator, SceneData


@pytest.fixture(scope="module")
def pc_generator() -> Generator[PointCloudGenerator, None, None]:
    """Create a PointCloudGenerator with demo settings.

    Yields:
        PointCloudGenerator: A configured generator instance.
    """
    args = argparse.Namespace(
        ckpt_dir=str(config.FOUNDATIONSTEREO_CHECKPOINT),
        scale=1.0,
        hiera=0,
        valid_iters=32,
        erosion_iterations=0,
        max_depth=3.0,
        need_confirm=False,
        use_png="demo6",
    )
    gen = PointCloudGenerator(args)
    yield gen
    gen.close()


def test_generate_pointcloud(pc_generator: PointCloudGenerator):
    """Generate a point cloud and verify object info is populated.

    Args:
        pc_generator (PointCloudGenerator): Pytest fixture providing a generator.
    """
    result = pc_generator.generate_pointcloud(["blue cup", "green cup"])

    assert isinstance(result, SceneData)
    assert "blue cup" in result.object_infos
    assert "green cup" in result.object_infos

    for name, obj in result.object_infos.items():
        assert obj.points is not None and len(obj.points) > 0, f"{name} has no points"
        assert obj.colors is not None and len(obj.colors) > 0, f"{name} has no colors"

    assert len(result.scene_info.pc_color) > 0
