from unittest import TestCase

import numpy as np

from nicoord import AffineTransform
from nicoord import CoordinateSystem
from nicoord import CoordinateSystemAxes
from nicoord import CoordinateSystemSpace
from nicoord import inverse
from nicoord import AffineTransformable


class TestCoordinateSystem(TestCase):
    """Test the nicoord.core.CoordinateSystem class"""

    def test_init(self):
        """Test the __init__ method"""

        # Test the default init.
        coordinate_system = CoordinateSystem()
        self.assertEqual(coordinate_system.axes, CoordinateSystemAxes.RAS)
        self.assertEqual(coordinate_system.space, CoordinateSystemSpace.VOXEL)

        # Change all the defaults.
        coordinate_system = CoordinateSystem(
            CoordinateSystemSpace.NATIVE, CoordinateSystemAxes.LPS)
        self.assertEqual(coordinate_system.axes, CoordinateSystemAxes.LPS)
        self.assertEqual(coordinate_system.space, CoordinateSystemSpace.NATIVE)

        # Verify wrong input types.
        self.assertRaises(TypeError, CoordinateSystem, axes=None)
        self.assertRaises(TypeError, CoordinateSystem, space='NATIVE')

    def test_repr(self):
        """Test the __repr__ method"""

        coordinate_system = CoordinateSystem()
        expected_repr = (
            'CoordinateSystem(CoordinateSystemSpace.VOXEL, '
            'CoordinateSystemAxes.RAS)')
        self.assertEqual(expected_repr, repr(coordinate_system))


class TestAffineTransform(TestCase):
    """Test the nicoord.core.AffineTransform class"""

    def test_init(self):
        """Test the __init__ method"""

        # Test using normal inputs.
        source = CoordinateSystem()
        target = CoordinateSystem(CoordinateSystemSpace.NATIVE)
        affine = np.array([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1]
        ])

        transform = AffineTransform(source, target, affine)
        self.assertEqual(source, transform.source)
        self.assertEqual(target, transform.target)
        np.testing.assert_array_equal(affine, transform.affine)

        # The affine can be any array like.
        affine = [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1]
        ]
        transform = AffineTransform(source, target, affine)
        self.assertEqual(source, transform.source)
        self.assertEqual(target, transform.target)
        np.testing.assert_array_equal(affine, transform.affine)

        # Test exceptions for bad inputs.
        self.assertRaises(TypeError, AffineTransform, 2, target, affine)
        self.assertRaises(TypeError, AffineTransform, source, 2, affine)
        self.assertRaises(ValueError, AffineTransform, source, target, [2, 1])
        self.assertRaises(TypeError, AffineTransform, source, target, ['a'])
        self.assertRaises(ValueError, AffineTransform, source, source, affine)

    def test_repr(self):
        """Test the __repr__ method"""

        source = CoordinateSystem()
        target = CoordinateSystem(CoordinateSystemSpace.NATIVE)
        affine = np.array([
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 1]
        ])

        transform = AffineTransform(source, target, affine)
        source_repr = (
            'CoordinateSystem(CoordinateSystemSpace.VOXEL, '
            'CoordinateSystemAxes.RAS)')
        target_repr = (
            'CoordinateSystem(CoordinateSystemSpace.NATIVE, '
            'CoordinateSystemAxes.RAS)')
        affine_repr = '[2. 0. 0. 0. 0. 2. 0. 0. 0. 0. 2. 0. 0. 0. 0. 1.]'
        expected_repr = (
            'AffineTransform(' + source_repr + ', ' + target_repr + ', ' +
            affine_repr + ')')
        self.assertEqual(expected_repr, repr(transform))

    def test_matmul_affine(self):
        """Test the __matmul__ method for affine inputs"""

        source = CoordinateSystem(CoordinateSystemSpace.VOXEL)
        target = CoordinateSystem(CoordinateSystemSpace.NATIVE)
        left_affine = np.eye(4)
        left_affine[:3, :] = np.random.randn(3, 4)
        left = AffineTransform(source, target, left_affine)

        right_affine = np.linalg.inv(left_affine)
        right = AffineTransform(target, source, right_affine)

        result = left @ right
        self.assertEqual(right.source, result.source)
        self.assertEqual(left.target, result.target)
        expected_affine = np.dot(left_affine, right_affine)
        np.testing.assert_array_almost_equal(expected_affine, result.affine)

        # The target of the left must match the source of the right.
        def matmul(r):
            return left @ r

        bad_right = AffineTransform(source, target, right_affine)
        self.assertRaises(ValueError, matmul, bad_right)

    def test_matmul_vertices(self):
        """Test the __matmul__ method for vertices inputs"""

        source = CoordinateSystem()
        target = CoordinateSystem(CoordinateSystemSpace.NATIVE)
        affine = np.eye(4)
        affine[:3, :] = np.random.randn(3, 4)
        transform = AffineTransform(source, target, affine)

        vertices = np.random.randn(100, 3)

        new_vertices = transform @ vertices

        homogeneous = np.hstack((vertices, np.ones((len(vertices), 1))))
        expected_vertices = np.dot(transform.affine, homogeneous.T).T[:, :3]
        np.testing.assert_array_almost_equal(expected_vertices, new_vertices)

        def matmul(v):
            return transform @ v

        # The vertices have to be (N, 3)
        self.assertRaises(ValueError, matmul, vertices.T)
        self.assertRaises(ValueError, matmul, vertices[:, :2])
        self.assertRaises(ValueError, matmul, vertices[:, 0])
        self.assertRaises(TypeError, matmul, ['a'])

        # Using iterables of the right shape is also fine.
        vertices = ((1, 0, 0),)
        new_vertices = transform @ vertices
        expected_vertices = (affine[:3, :1] + affine[:3, 3:]).T
        np.testing.assert_array_almost_equal(expected_vertices, new_vertices)


class TestInverse(TestCase):
    """Test the nicoord.core.inverse function"""

    def test_simple(self):
        """Test the simple case"""

        source = CoordinateSystem(CoordinateSystemSpace.VOXEL)
        target = CoordinateSystem(CoordinateSystemSpace.NATIVE)
        affine = np.eye(4)
        affine[:3, :] = np.random.randn(3, 4)
        transform = AffineTransform(source, target, affine)

        inv_transform = inverse(transform)

        left = inv_transform @ transform
        self.assertEqual(transform.source, left.source)
        self.assertEqual(transform.source, left.target)
        np.testing.assert_array_almost_equal(np.eye(4), left.affine)

        right = transform @ inv_transform
        self.assertEqual(transform.target, right.source)
        self.assertEqual(transform.target, right.target)
        np.testing.assert_array_almost_equal(np.eye(4), right.affine)


class SampleAffineTransformable(AffineTransformable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vertices = np.eye(3)

    @property
    def _transformable_points(self):
        return [self._vertices]

    @_transformable_points.setter
    def _transformable_points(self, points):
        self._vertices = next(p for p in points)

    @property
    def vertices(self):
        return self._vertices


class TestAffineTransformable(TestCase):
    """Test the nicoord.core.AffineTransformable class"""

    def test_init(self):
        """Test the __init__ method"""

        source = CoordinateSystem(CoordinateSystemSpace.NATIVE)
        target = CoordinateSystem(CoordinateSystemSpace.VOXEL)
        affine = np.eye(4)
        affine[:3, :] = np.random.randn(3, 4)
        transform = AffineTransform(source, target, affine)

        transformable = SampleAffineTransformable()
        self.assertEqual(transformable.coordinate_system, CoordinateSystem())

        transformable = SampleAffineTransformable(source)
        self.assertEqual(transformable.coordinate_system, source)

        transformable = SampleAffineTransformable(source, [transform])
        self.assertEqual(transformable.coordinate_system, source)

    def test_add_transform(self):
        """Test the add_transform method"""

        source = CoordinateSystem(CoordinateSystemSpace.VOXEL)
        target = CoordinateSystem(CoordinateSystemSpace.NATIVE)
        affine = np.array([
            [0, 2, 0, 1],
            [1, 0, 0, 2],
            [0, 0, 3, 3],
            [0, 0, 0, 1],
        ])
        transform = AffineTransform(source, target, affine)

        transformable = SampleAffineTransformable()
        self.assertEqual(len(transformable.transforms), 0)
        transformable.add_transform(transform)
        self.assertEqual(len(transformable.transforms), 1)
        self.assertEqual(transform, transformable.transforms[0])

        # Cannot add two transforms to the same target.
        self.assertRaises(ValueError, transformable.add_transform, transform)

        # Cannot add a transform from a different coordinate system.
        bad_transform = AffineTransform(target, source, affine)
        self.assertRaises(
            ValueError, transformable.add_transform, bad_transform)

    def test_transform_to(self):
        """Test the transform_to method"""

        source = CoordinateSystem(CoordinateSystemSpace.VOXEL)
        target = CoordinateSystem(CoordinateSystemSpace.NATIVE)
        affine = np.array([
            [0, 2, 0, 1],
            [1, 0, 0, 2],
            [0, 0, 3, 3],
            [0, 0, 0, 1],
        ])
        transform = AffineTransform(source, target, affine)

        transformable = SampleAffineTransformable(
            coordinate_system=source, transforms=[transform])
        np.testing.assert_array_almost_equal(np.eye(3), transformable.vertices)

        # Verify the vertices in the transformed coordinate system.
        transformable.transform_to(target)
        expected = np.array([
            [1, 3, 1],
            [3, 2, 2],
            [3, 3, 6],
        ]).T
        np.testing.assert_array_almost_equal(expected, transformable.vertices)

        # Asking to go to the current coordinate system does nothing.
        transformable.transform_to(target)
        np.testing.assert_array_almost_equal(expected, transformable.vertices)

        # Going back to the original coordinate system should give the original
        # vertices.
        transformable.transform_to(source)
        np.testing.assert_array_almost_equal(np.eye(3), transformable.vertices)

        # We can't transform to a space for which there is no affine.
        transformable = SampleAffineTransformable(coordinate_system=source)
        self.assertRaises(ValueError, transformable.transform_to, target)
