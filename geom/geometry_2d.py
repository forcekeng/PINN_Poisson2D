"""
参考:
Mindelec: https://gitee.com/mindspore/mindscience/tree/master/MindElec
DeepXDE: https://github.com/lululxvi/deepxde
"""

from __future__ import absolute_import
__all__ = ["Disk", "Polygon", "Rectangle", "Triangle"]


"""2d geometry"""


import numpy as np
import numpy.linalg as LA
from scipy import spatial
from mindspore import log as logger

from mindelec.geometry.geometry_base import Geometry, DATA_TYPES, GEOM_TYPES
from mindelec.geometry.geometry_nd import HyperCube
from mindelec.geometry.utils import sample, polar_sample, generate_mesh


class Disk(Geometry):
    r"""
    Definition of Disk object.

    Args:
        name (str): name of the disk.
        center (Union[tuple[int, float], list[int, float], numpy.ndarray]): center coordinates of the disk.
        radius (Union[int, float]): radius of the disk.
        dtype (numpy.dtype): data type of sampled point data type. Default: numpy.float32.
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Raises:
        ValueError: If `center` is neither list nor tuple of length 2.
        ValueError: If `radius` is negative.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Disk
        >>> disk_mesh = edict({'domain': edict({'random_sampling': False, 'size' : [100, 180]}),
        ...                    'BC': edict({'random_sampling': False, 'size': 200, 'with_normal' : True,})})
        >>> disk = Disk("disk", (-1.0, 0), 2.0, sampling_config=create_config_from_edict(disk_mesh))
        >>> domain = disk.sampling(geom_type="domain")
        >>> bc, bc_normal = disk.sampling(geom_type="BC")
        >>> print(bc.shape)
        (200, 2)
    """
    def __init__(self, name, center, radius, dtype=np.float32, sampling_config=None):
        self.sampling_config = sampling_config
        if not isinstance(center, (np.ndarray, tuple, list)):
            raise TypeError("Disk: {}'s center should be tuple or list, but got: {}, type: {}".format(
                name, center, type(center)))
        self.center = np.array(center)
        if len(self.center) != 2:
            raise ValueError("Disk: {}'s center should be 2D array, but got: {}, dim: {}".format(
                name, self.center, len(self.center)))
        for ele in self.center:
            if not isinstance(ele, DATA_TYPES):
                raise TypeError("data type of center should be int/float, but got: {}, type: {}".format(
                    self.center, type(ele)
                ))
        if not isinstance(radius, (int, float)) or isinstance(radius, bool):
            raise TypeError("data type of radius should be int/float, but got: {}, type: {}".format(
                radius, type(radius)
            ))
        if radius <= 0:
            raise ValueError("Disk: {}'s radius should not be non-positive, but got: {}".format(name, radius))
        self.radius = radius
        self.columns_dict = {}
        coord_min = self.center - self.radius
        coord_max = self.center + self.radius
        super(Disk, self).__init__(name, 2, coord_min, coord_max, dtype, sampling_config)

    def _inside(self, points, strict=False):
        """whether inside domain"""
        return LA.norm(points - self.center, axis=-1) < self.radius if strict \
            else LA.norm(points - self.center, axis=-1) <= self.radius

    def _on_boundary(self, points):
        """whether on domain boundary"""
        return np.isclose(LA.norm(points - self.center, axis=-1), self.radius)

    def _boundary_normal(self, points):
        """get the boundary normal vector"""
        points = points[self._on_boundary(points)]
        r = points - self.center
        r_norm = LA.norm(r, axis=-1, keepdims=True)
        return r / r_norm

    def _random_disk_boundary_points(self, need_normal=False):
        """Randomly generate boundary points"""
        size = self.sampling_config.bc.size
        sampler = self.sampling_config.bc.sampler
        theta = 2 * np.pi * sample(size, 1, sampler)
        circle_xy = np.hstack([np.cos(theta), np.sin(theta)])
        data = self.center + circle_xy * self.radius
        data = np.reshape(data, (-1, self.dim))
        if need_normal:
            normal_data = self._boundary_normal(data)
            normal_data = np.reshape(normal_data, (-1, self.dim))
            return data, normal_data
        return data

    def _random_disk_domain_points(self):
        """Randomly generate domain points"""
        size = self.sampling_config.domain.size
        sampler = self.sampling_config.domain.sampler
        r_theta = sample(size, 2, sampler)
        data = self.center + polar_sample(r_theta) * self.radius
        data = np.reshape(data, (-1, self.dim))
        return data

    def _grid_disk_boundary_points(self, need_normal=False):
        """Generate uniformly distributed domain points"""
        size = self.sampling_config.bc.size
        theta = np.linspace(0, 2 * np.pi, num=size, endpoint=False)
        cartesian = np.vstack((np.cos(theta), np.sin(theta))).T
        data = self.radius * cartesian + self.center
        data = np.reshape(data, (-1, self.dim))
        if need_normal:
            normal_data = self._boundary_normal(data)
            normal_data = np.reshape(normal_data, (-1, self.dim))
            return data, normal_data
        return data

    def _grid_disk_domain_points(self):
        """Generate uniformly distributed domain points"""
        mesh_size = self.sampling_config.domain.size
        if len(mesh_size) != self.dim:
            raise ValueError("For grid sampling, length of mesh_size list: {} should be equal to dimension: {}".format(
                mesh_size, self.dim
            ))
        r_theta_mesh = generate_mesh(np.array([0, 0]), np.array([1, 1]), mesh_size, endpoint=False)
        cartesian = np.zeros(r_theta_mesh.shape)
        cartesian[:, 0] = r_theta_mesh[:, 0] * self.radius * np.cos(2 * np.pi * r_theta_mesh[:, 1])
        cartesian[:, 1] = r_theta_mesh[:, 0] * self.radius * np.sin(2 * np.pi * r_theta_mesh[:, 1])
        data = cartesian + self.center
        data = np.reshape(data, (-1, self.dim))
        return data

    def sampling(self, geom_type="domain"):
        """
        sampling domain and boundary points.

        Args:
            geom_type (str): geometry type.

        Returns:
            Numpy.array, 2D numpy array with or without boundary normal vectors.

        Raises:
            ValueError: If `config` is None.
            KeyError: If `geom_type` is `domain` but `config.domain` is None.
            KeyError: If `geom_type` is `BC` but `config.bc` is None.
            ValueError: If `geom_type` is neither `BC` nor `domain`.
        """
        config = self.sampling_config
        if config is None:
            raise ValueError("Sampling config for {}:{} is None, please call set_sampling_config method to set".format(
                self.geom_type, self.name))
        if not isinstance(geom_type, str):
            raise TypeError("geom_type shouild be string, but got {} with type {}".format(geom_type, type(geom_type)))
        if geom_type not in GEOM_TYPES:
            raise ValueError("Unsupported geom_type: {}, only {} are supported now".format(geom_type, GEOM_TYPES))
        if geom_type.lower() == "domain":
            if config.domain is None:
                raise KeyError("Sampling config for domain of {}:{} should not be none"
                               .format(self.geom_type, self.name))
            logger.info("Sampling domain points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.domain))
            column_name = self.name + "_domain_points"
            if config.domain.random_sampling:
                disk_data = self._random_disk_domain_points()
            else:
                disk_data = self._grid_disk_domain_points()
            self.columns_dict["domain"] = [column_name]
            disk_data = disk_data.astype(self.dtype)
            return disk_data
        if geom_type.lower() == "bc":
            if config.bc is None:
                raise KeyError("Sampling config for BC of {}:{} should not be none".format(self.geom_type, self.name))
            logger.info("Sampling BC points for {}:{}, config info: {}"
                        .format(self.geom_type, self.name, config.bc))
            if config.bc.with_normal:
                if config.bc.random_sampling:
                    disk_data, disk_data_normal = self._random_disk_boundary_points(need_normal=True)
                else:
                    disk_data, disk_data_normal = self._grid_disk_boundary_points(need_normal=True)
                column_data = self.name + "_BC_points"
                column_normal = self.name + "_BC_normal"
                self.columns_dict["BC"] = [column_data, column_normal]
                disk_data = disk_data.astype(self.dtype)
                disk_data_normal = disk_data_normal.astype(self.dtype)
                return disk_data, disk_data_normal

            if config.bc.random_sampling:
                disk_data = self._random_disk_boundary_points(need_normal=False)
            else:
                disk_data = self._grid_disk_boundary_points(need_normal=False)
            column_data = self.name + "_BC_points"
            self.columns_dict["BC"] = [column_data]
            disk_data = disk_data.astype(self.dtype)
            return disk_data
        raise ValueError("Unknown geom_type: {}, only \"domain/BC\" are supported for {}:{}".format(
            geom_type, self.geom_type, self.name))


class Rectangle(HyperCube):
    r"""
    Definition of Rectangle object.

    Args:
        name (str): name of the rectangle.
        coord_min (Union[tuple[int, float], list[int, float], numpy.ndarray]): coordinates of the bottom
            left corner of rectangle.
        coord_max (Union[tuple[int, float], list[int, float], numpy.ndarray]): coordinates of the top
            right corner of rectangle.
        dtype (numpy.dtype): data type of sampled point data type. Default: numpy.float32.
        sampling_config (SamplingConfig): sampling configuration. Default: None

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from easydict import EasyDict as edict
        >>> from mindelec.geometry import create_config_from_edict, Rectangle
        >>> rectangle_mesh = edict({'domain': edict({'random_sampling': False, 'size': [50, 25]}),
        ...                         'BC': edict({'random_sampling': False, 'size': 300, 'with_normal': True,}),})
        >>> rectangle = Rectangle("rectangle", (-3.0, 1), (1, 2),
        ...                       sampling_config=create_config_from_edict(rectangle_mesh))
        >>> domain = rectangle.sampling(geom_type="domain")
        >>> bc, bc_normal = rectangle.sampling(geom_type="BC")
        >>> print(domain.shape)
        (1250, 2)
    """
    def __init__(self, name, coord_min, coord_max, dtype=np.float32, sampling_config=None):
        super(Rectangle, self).__init__(name, 2, coord_min, coord_max, dtype=dtype, sampling_config=sampling_config)
    

class Triangle(Geometry):
    def __init__(self, name, vertices, dtype=np.float32, sampling_config=None):
        
        assert len(vertices)==3, "Not triangle beacause len(vertices) != 3"
        x1, x2, x3 = vertices
        self.area = polygon_signed_area([x1, x2, x3])
        # Clockwise
        if self.area < 0:
            self.area = -self.area
            x2, x3 = x3, x2

        self.x1 = np.array(x1, dtype=np.float32)
        self.x2 = np.array(x2, dtype=np.float32)
        self.x3 = np.array(x3, dtype=np.float32)

        self.v12 = self.x2 - self.x1
        self.v23 = self.x3 - self.x2
        self.v31 = self.x1 - self.x3
        self.l12 = np.linalg.norm(self.v12)
        self.l23 = np.linalg.norm(self.v23)
        self.l31 = np.linalg.norm(self.v31)
        self.n12 = self.v12 / self.l12
        self.n23 = self.v23 / self.l23
        self.n31 = self.v31 / self.l31
        self.n12_normal = clockwise_rotation_90(self.n12)
        self.n23_normal = clockwise_rotation_90(self.n23)
        self.n31_normal = clockwise_rotation_90(self.n31)
        self.perimeter = self.l12 + self.l23 + self.l31
        
        self.columns_dict = {}
        super().__init__(name, 2,
            np.minimum(x1, np.minimum(x2, x3)), np.maximum(x1, np.maximum(x2, x3)),
            dtype, sampling_config
        )
        

    def _inside(self, points, strict=False):
        # https://stackoverflow.com/a/2049593/12679294
        _sign = np.hstack(
            [
                np.cross(self.v12, points - self.x1)[:, np.newaxis],
                np.cross(self.v23, points - self.x2)[:, np.newaxis],
                np.cross(self.v31, points - self.x3)[:, np.newaxis],
            ]
        )
        return ~np.logical_and(np.any(_sign > 0, axis=-1), np.any(_sign < 0, axis=-1))

    
    def _on_boundary(self, points):
        l1 = np.linalg.norm(points - self.x1, axis=-1)
        l2 = np.linalg.norm(points - self.x2, axis=-1)
        l3 = np.linalg.norm(points - self.x3, axis=-1)
        return np.any(
            np.isclose(
                [l1 + l2 - self.l12, l2 + l3 - self.l23, l3 + l1 - self.l31],
                0,
                atol=1e-6,
            ),
            axis=0,
        )


    def _boundary_normal(self, points):
        l1 = np.linalg.norm(points - self.x1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(points - self.x2, axis=-1, keepdims=True)
        l3 = np.linalg.norm(points - self.x3, axis=-1, keepdims=True)
        on12 = np.isclose(l1 + l2, self.l12)
        on23 = np.isclose(l2 + l3, self.l23)
        on31 = np.isclose(l3 + l1, self.l31)
        # Check points on the vertexes
        if np.any(np.count_nonzero(np.hstack([on12, on23, on31]), axis=-1) > 1):
            raise ValueError(
                "{}.boundary_normal do not accept points on the vertexes.".format(
                    self.__class__.__name__
                )
            )
        return self.n12_normal * on12 + self.n23_normal * on23 + self.n31_normal * on31

    def _random_triangle_boundary_points(self):
        n = self.sampling_config.bc.size  # 采样点数
        sampler = self.sampling_config.bc.sampler # 采样分布
        u = np.ravel(sample(n + 2, 1, sampler))
        # Remove the possible points very close to the corners
        u = u[np.logical_not(np.isclose(u, self.l12 / self.perimeter))]
        u = u[np.logical_not(np.isclose(u, (self.l12 + self.l23) / self.perimeter))]
        u = u[:n]

        u *= self.perimeter
        x = []
        for l in u:
            if l < self.l12:
                x.append(l * self.n12 + self.x1)
            elif l < self.l12 + self.l23:
                x.append((l - self.l12) * self.n23 + self.x2)
            else:
                x.append((l - self.l12 - self.l23) * self.n31 + self.x3)
        return np.vstack(x)

    def _random_triangle_domain_sampling(self):
        n = self.sampling_config.domain.size
        sampler = self.sampling_config.domain.sampler
        sqrt_r1 = np.sqrt(np.random.rand(n, 1))
        r2 = sample(n, 1, sampler)
        return (
            (1 - sqrt_r1) * self.x1
            + sqrt_r1 * (1 - r2) * self.x2
            + r2 * sqrt_r1 * self.x3
        )


    def sampling(self, geom_type="domain"):
        config = self.sampling_config
        if geom_type.lower() == "domain":
            column_name = self.name + "_domain_points"
            if not config.domain.random_sampling:
                raise ValueError("Triangle only support random sampling but get random_sampling is True")
            self.columns_dict["domain"] = [column_name]
            data = self._random_triangle_domain_sampling()
            data = data.astype(self.dtype)
            return data

        if geom_type.lower() == "bc":
            column_data = self.name + "_BC_points"
            if not config.bc.random_sampling:
                raise ValueError("Triangle only support random sampling but get random_sampling is True")
                
            self.columns_dict["BC"] = [column_data]
            data = self._random_triangle_boundary_points()
            data = data.astype(self.dtype)
        return data
    
    
class Polygon(Geometry):
    """Simple polygon.

    Args:
        vertices: The order of vertices can be in a clockwise or counterclockwise
            direction. The vertices will be re-ordered in counterclockwise (right hand
            rule).
    """

    def __init__(self, name, vertices, dtype=np.float32, sampling_config=None):
        self.vertices = np.array(vertices, dtype=np.float32)
        
        if len(vertices) == 3:
            raise ValueError("The polygon is a triangle. Use Triangle instead.")

        self.area = polygon_signed_area(self.vertices)
        # Clockwise
        if self.area < 0:
            self.area = -self.area
            self.vertices = np.flipud(self.vertices)

        self.diagonals = spatial.distance.squareform(
            spatial.distance.pdist(self.vertices)
        )
        self.columns_dict = {}
        super().__init__(
            name,
            2,
            np.amin(self.vertices, axis=0), np.amax(self.vertices, axis=0),
            dtype,
            sampling_config
        )
        self.nvertices = len(self.vertices)
        self.perimeter = np.sum(
            [self.diagonals[i, i + 1] for i in range(-1, self.nvertices - 1)]
        )
        self.bbox = np.array(
            [np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)]
        )

        self.segments = self.vertices[1:] - self.vertices[:-1]
        self.segments = np.vstack((self.vertices[0] - self.vertices[-1], self.segments))
        self.normal = clockwise_rotation_90(self.segments.T).T
        self.normal = self.normal / np.linalg.norm(self.normal, axis=1).reshape(-1, 1)
    
    
    def _inside(self, points, strict=False):
        def wn_PnPoly(P, V):
            """Winding number algorithm.

            https://en.wikipedia.org/wiki/Point_in_polygon
            http://geomalgorithms.com/a03-_inclusion.html

            Args:
                P: A point.
                V: Vertex points of a polygon.

            Returns:
                wn: Winding number (=0 only if P is outside polygon).
            """
            wn = np.zeros(len(P))  # Winding number counter

            # Repeat the first vertex at end
            # Loop through all edges of the polygon
            for i in range(-1, self.nvertices - 1):  # Edge from V[i] to V[i+1]
                tmp = np.all(
                    np.hstack(
                        [
                            V[i, 1] <= P[:, 1:2],  # Start y <= P[1]
                            V[i + 1, 1] > P[:, 1:2],  # An upward crossing
                            is_left(V[i], V[i + 1], P) > 0,  # P left of edge
                        ]
                    ),
                    axis=-1,
                )
                wn[tmp] += 1  # Have a valid up intersect
                tmp = np.all(
                    np.hstack(
                        [
                            V[i, 1] > P[:, 1:2],  # Start y > P[1]
                            V[i + 1, 1] <= P[:, 1:2],  # A downward crossing
                            is_left(V[i], V[i + 1], P) < 0,  # P right of edge
                        ]
                    ),
                    axis=-1,
                )
                wn[tmp] -= 1  # Have a valid down intersect
            return wn

        return wn_PnPoly(points, self.vertices) != 0
    
    def _on_boundary(self, points):
        _on = np.zeros(shape=len(points), dtype=np.int)
        for i in range(-1, self.nvertices - 1):
            l1 = np.linalg.norm(self.vertices[i] - points, axis=-1)
            l2 = np.linalg.norm(self.vertices[i + 1] - points, axis=-1)
            _on[np.isclose(l1 + l2, self.diagonals[i, i + 1])] += 1
        return _on > 0

    
    def _boundary_normal(self, points):
        for i in range(self.nvertices):
            if is_on_line_segment(self.vertices[i - 1], self.vertices[i], points):
                return self.normal[i]
        return np.array([0, 0])

    def _random_polygon_domain_sampling(self):
        n = self.sampling_config.domain.size
        sampler = self.sampling_config.domain.sampler
        x = np.empty((0, 2), dtype=np.float32)
        vbbox = self.bbox[1] - self.bbox[0] # 右上角的点减去左下角的点
        while len(x) < n:
            x_new = sample(n, 2, sampler=sampler) * vbbox + self.bbox[0]
            x = np.vstack((x, x_new[self._inside(x_new)]))
        return x[:n]

    def _uniform_polygon_boundary_sampling(self):
        n = self.sampling_config.domain.size
        density = n / self.perimeter
        x = []
        for i in range(-1, self.nvertices - 1):
            x.append(
                np.linspace(
                    0,
                    1,
                    num=int(np.ceil(density * self.diagonals[i, i + 1])),
                    endpoint=False,
                )[:, None]
                * (self.vertices[i + 1] - self.vertices[i])
                + self.vertices[i]
            )
        x = np.vstack(x)
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    
    def _random_polygon_boundary_sampling(self):
        n = self.sampling_config.domain.size
        sampler = self.sampling_config.domain.sampler
        u = np.ravel(sample(n + self.nvertices, 1, sampler))
        # Remove the possible points very close to the corners
        l = 0
        for i in range(0, self.nvertices - 1):
            l += self.diagonals[i, i + 1]
            u = u[np.logical_not(np.isclose(u, l / self.perimeter))]
        u = u[:n]
        u *= self.perimeter
        u.sort()

        x = []
        i = -1
        l0 = 0
        l1 = l0 + self.diagonals[i, i + 1]
        v = (self.vertices[i + 1] - self.vertices[i]) / self.diagonals[i, i + 1]
        for l in u:
            if l > l1:
                i += 1
                l0, l1 = l1, l1 + self.diagonals[i, i + 1]
                v = (self.vertices[i + 1] - self.vertices[i]) / self.diagonals[i, i + 1]
            x.append((l - l0) * v + self.vertices[i])
        out = np.vstack(x)
        np.random.shuffle(out)
        return out

    
    def sampling(self, geom_type="domain"):
        config = self.sampling_config
        if geom_type.lower() == "domain":
            column_name = self.name + "_domain_points"
            if not config.domain.random_sampling:
                raise ValueError("Triangle only support random sampling but get random_sampling is True")
            self.columns_dict["domain"] = [column_name]
            data = self._random_polygon_domain_sampling()
            data = data.astype(self.dtype)
            return data

        if geom_type.lower() == "bc":
            column_data = self.name + "_BC_points"
            if not config.bc.random_sampling: # 随机采样
                data = self._uniform_polygon_boundary_sampling()
            else:     # 均匀采样
                data = self._random_polygon_boundary_sampling()
            self.columns_dict["BC"] = [column_data]
            data = data.astype(self.dtype)
        return data



def polygon_signed_area(vertices):
    """The (signed) area of a simple polygon.

    If the vertices are in the counterclockwise direction, then the area is positive; if
    they are in the clockwise direction, the area is negative.

    Shoelace formula: https://en.wikipedia.org/wiki/Shoelace_formula
    """
    x, y = zip(*vertices)
    x = np.array(list(x) + [x[0]])
    y = np.array(list(y) + [y[0]])
    return 0.5 * (np.sum(x[:-1] * y[1:]) - np.sum(x[1:] * y[:-1]))


def clockwise_rotation_90(v):
    """Rotate a vector of 90 degrees clockwise about the origin."""
    return np.array([v[1], -v[0]])


def is_left(P0, P1, P2):
    """Test if a point is Left|On|Right of an infinite line.

    See: the January 2001 Algorithm "Area of 2D and 3D Triangles and Polygons".

    Args:
        P0: One point in the line.
        P1: One point in the line.
        P2: A array of point to be tested.

    Returns:
        >0 if P2 left of the line through P0 and P1, =0 if P2 on the line, <0 if P2
        right of the line.
    """
    return np.cross(P1 - P0, P2 - P0, axis=-1).reshape((-1, 1))


def is_rectangle(vertices):
    """Check if the geometry is a rectangle.

    https://stackoverflow.com/questions/2303278/find-if-4-points-on-a-plane-form-a-rectangle/2304031

    1. Find the center of mass of corner points: cx=(x1+x2+x3+x4)/4, cy=(y1+y2+y3+y4)/4
    2. Test if square of distances from center of mass to all 4 corners are equal
    """
    if len(vertices) != 4:
        return False

    c = np.mean(vertices, axis=0)
    d = np.sum((vertices - c) ** 2, axis=1)
    return np.allclose(d, np.full(4, d[0]))


def is_on_line_segment(P0, P1, P2):
    """Test if a point is between two other points on a line segment.

    Args:
        P0: One point in the line.
        P1: One point in the line.
        P2: The point to be tested.

    References:
        https://stackoverflow.com/questions/328107
    """
    v01 = P1 - P0
    v02 = P2 - P0
    v12 = P2 - P1
    return (
        # check that P2 is almost on the line P0 P1
        np.isclose(np.cross(v01, v02) / np.linalg.norm(v01), 0, atol=1e-6)
        # check that projection of P2 to line is between P0 and P1
        and v01 @ v02 >= 0
        and v01 @ v12 <= 0
    )
    # Not between P0 and P1, but close to P0 or P1
    # or np.isclose(np.linalg.norm(v02), 0, atol=1e-6)  # check whether P2 is close to P0
    # or np.isclose(np.linalg.norm(v12), 0, atol=1e-6)  # check whether P2 is close to P1
