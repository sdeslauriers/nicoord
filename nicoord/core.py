from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np


class CoordinateSystemAxes(Enum):
    RAS = 0
    LPS = 1


class CoordinateSystemSpace(Enum):
    UNKNOWN = 0
    VOXEL = 1
    NATIVE = 2


class CoordinateSystem(object):
    def __init__(self,
                 space: CoordinateSystemSpace = CoordinateSystemSpace.VOXEL,
                 axes: CoordinateSystemAxes = CoordinateSystemAxes.RAS):
        """A neuroimaging coordinate system

        Neuroimaging coordinate system is defined by the orientation and
        order of its axes (ex: RAS, LPS, ...). To differentiate coordinate
        system with the same orientation and order, a coordinate system is also
        given a reference space (ex: VOXEL, MNI, NATIVE, ...).

        Args:
            space: The space of the coordinate system.
            axes: The orientation an direction of the axes.

        """

        if not isinstance(space, CoordinateSystemSpace):
            raise TypeError(
                f'The space must be a CoordinateSystemSpace, '
                f'not a {type(space)}.')
        self._space = space

        if not isinstance(axes, CoordinateSystemAxes):
            raise TypeError(
                f'The axes orientation and order must be a '
                f'CoordinateSystemAxes, not a {type(axes)}.')
        self._axes = axes

    def __eq__(self, other: 'CoordinateSystem') -> bool:
        """Equality between two coordinate systems

        Two coordinate systems are equal if they have the same space and the
        same orientation and order of axes.

        Args:
            other: The coordinate system on the right of the equality.

        """

        return self.space == other.space and self.axes == self.axes

    def __ne__(self, other: 'CoordinateSystem') -> bool:
        """Inequality between two coordinate systems

        Two coordinate systems are not equal if they have different spaces or
        different orientation and order of axes.

        Args:
            other: The coordinate system on the right of the inequality.

        """

        return not (self == other)

    def __repr__(self):
        """Returns a string representation of the coordinate system"""
        return f'CoordinateSystem({self.space}, {self.axes})'

    @property
    def axes(self) -> CoordinateSystemAxes:
        """Returns the orientation and order of the axes"""
        return self._axes

    @property
    def space(self) -> CoordinateSystemSpace:
        """Returns the reference space of the coordinate system"""
        return self._space


class AffineTransform(object):
    def __init__(self,
                 source: CoordinateSystem,
                 target: CoordinateSystem,
                 affine: Any):
        """An affine transformation between two coordinate systems

        An affine transform allows coordinates (or points) to be converted
        from one coordinate system to another.

        Args:
            source: The initial coordinate system of the coordinates.
            target: The coordinate system after application of the affine.
            affine: The matrix representation of the affine. Must be
                convertible to a numpy array of floats with a shape of (4, 4).

        """

        if not isinstance(source, CoordinateSystem):
            raise TypeError(
                f'The source must be a CoordinateSystem, not a '
                f'{type(source)}.')
        self._source = source

        if not isinstance(target, CoordinateSystem):
            raise TypeError(
                f'The target must be a CoordinateSystem, not a '
                f'{type(target)}.')
        self._target = target

        # The affine must be convertible to a numpy array with a shape of
        # (4, 4).
        try:
            affine = np.array(affine, dtype=np.float64)
        except ValueError:
            raise TypeError(
                'The affine must be convertible to a numpy array of floats.')

        if affine.shape != (4, 4):
            raise ValueError(
                f'The affine must have a shape of (4, 4), not '
                f'{affine.shape}.')

        if source == target:
            if not np.allclose(affine, np.eye(4)):
                raise ValueError(
                    'The source and the target coordinate systems are '
                    'identical but the affine is not the identity.')

        self._affine = affine

    def __matmul__(
            self,
            other: Union['AffineTransform', Any]
    ) -> Union['AffineTransform', np.ndarray]:
        """Matrix product of an affine transform with another or with points

        The matrix product of two affine transforms is only valid if the
        source of the left operand is the target of the right operand. The
        result is a new affine transform that goes from the source of the
        right operand to the target of the left operand.

        The matrix product of an affine transform with an array like of
        3D points with a shape of (N, 3) transforms the points to the target
        coordinate system of the affine. It is assumed that the input points
        are originally in the source coordinate system of the affine.

        Args:
            other: The affine transform on the right of the product or points
                to transform to a new coordinate system.

        """

        if isinstance(other, AffineTransform):

            # The source of the left must be the target of the right.
            if self.source != other.target:
                raise ValueError(
                    f'The source coordinate system of the left operand must '
                    f'be the target of the right operand '
                    f'({self.source} != {other.target}).')

            new_affine = np.dot(self.affine, other.affine)
            return AffineTransform(other.source, self.target, new_affine)

        else:

            # Try to convert the input to a numpy array.
            try:
                points = np.array(other, dtype=np.float)
            except ValueError:
                raise TypeError(
                    'The right operand is not an affine transform and could '
                    'not be convert to a numpy array of floats.')

            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f'The points must have a shape of (N, 3), not '
                    f'{points.shape}')

            matrix = self.affine[:3, :3]
            translation = self.affine[:3, 3:]
            return (np.dot(matrix, points.T) + translation).T

    def __repr__(self) -> str:
        """Returns the string representation of the affine transform"""
        flat_affine = self.affine.ravel()
        return f'AffineTransform({self.source}, {self.target}, {flat_affine})'

    @property
    def affine(self) -> np.ndarray:
        """Returns the numpy array representation of the affine"""
        return self._affine.copy()

    @property
    def is_identity(self) -> bool:
        """Returns whether the affine transform is the identity or not"""
        return self.source == self.target

    @property
    def source(self) -> CoordinateSystem:
        """Returns the source coordinate system of the affine transform"""
        return self._source

    @property
    def target(self) -> CoordinateSystem:
        """Returns the target coordinate system of the affine transform"""
        return self._target


class AffineTransformable(ABC):
    def __init__(
            self,
            coordinate_system: Optional[CoordinateSystem] = None,
            transforms: Optional[Iterable[AffineTransform]] = None):
        """Objects that can be transformed using an affine

        The AffineTransformable class is the base class for all objects
        that can be represented by points whose coordinate system can be
        changed using an affine.

        The only requirement is that the object provide getter/setter access to
        its points via the _transformable_points property.

        Args:
            coordinate_system: The current coordinate system of the points. If
                not provided, a VOXEL RAS coordinate system is assumed.
            transforms: The affine transformations to other coordinate systems.

        """

        coordinate_system = coordinate_system or CoordinateSystem()
        transforms = transforms or []

        self._coordinate_system: CoordinateSystem = coordinate_system
        self._transforms: List[AffineTransform] = []

        for transform in transforms:
            self.add_transform(transform)

    @property
    @abstractmethod
    def _transformable_points(self) -> Iterable[np.ndarray]:
        """Returns the points of the object"""
        pass

    @_transformable_points.setter
    @abstractmethod
    def _transformable_points(self, points: Iterable[np.ndarray]):
        """Modifies the points of the object"""
        pass

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Returns the coordinate system of the object"""
        return self._coordinate_system

    @property
    def transforms(self) -> Tuple[AffineTransform]:
        """Returns the available transforms for the object"""
        return tuple(self._transforms)

    def add_transform(self, transform: AffineTransform):
        """Adds a transform to the available transforms for the object"""

        if transform.source != self._coordinate_system:
            raise ValueError(
                f'The source of all the transforms must match the '
                f'current coordinate system '
                f'({transform.source} != {self._coordinate_system}).')

        # We cannot add two transforms to the same coordinate system.
        if any([t.target == transform.target for t in self._transforms]):
            raise ValueError(
                f'A transform to {transform.target} is already available.')

        self._transforms.append(transform)

    def transform_to(self, coordinate_system: CoordinateSystem):
        """Transforms the points to a new coordinate system"""

        if self.coordinate_system == coordinate_system:
            return

        # Find a transform to the desired space.
        def is_valid(t):
            return t.target == coordinate_system
        valid_transforms = [t for t in self._transforms if is_valid(t)]

        if len(valid_transforms) == 0:
            raise ValueError(
                f'No transforms to {coordinate_system} are available.')

        transform = valid_transforms[0]
        new_points = [transform @ p for p in self._transformable_points]
        self._transformable_points = new_points
        self._coordinate_system = coordinate_system

        # Also apply the inverse of transform to every affine of the object to
        # modify their source.
        inv_transform = inverse(transform)
        self._transforms = [t @ inv_transform for t in self._transforms]

        # Remove the identity transform as they are not useful and add the
        # inverse transform to be able to go back to the original space.
        self._transforms = [t for t in self._transforms if not t.is_identity]
        self._transforms.append(inv_transform)


def inverse(transform: AffineTransform) -> AffineTransform:
    """Returns the inverse of an affine transform

    The inverse of an affine transform that goes from space A to space B is a
    new affine transform that goes from space B to space A.

    Args:
        transform: The affine transform to invert.

    """

    inv_affine = np.linalg.inv(transform.affine)
    return AffineTransform(transform.target, transform.source, inv_affine)
