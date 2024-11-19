import struct
import base64
from math import atan2, sqrt, pi
from functools import reduce

import numpy as np
import pygltflib
from openalea.plantgl import all as pgl


def shape_mesh(pgl_object, tesselator=None):
    if tesselator is None:
        tesselator = pgl.Tesselator()
    pgl_object.apply(tesselator)
    mesh = tesselator.triangulation
    if mesh:
        indices = list(map(tuple, mesh.indexList))
        pts = list(map(tuple, mesh.pointList))
    else:
        indices = []
        pts = []
    return pts, indices


def as_scene_mesh(pgl_scene):
    """Transform a PlantGL scene / PlantGL shape dict to a scene_mesh"""
    tesselator = pgl.Tesselator()

    sm = {}

    def _concat_mesh(mesh1, mesh2):
        v1, f1 = mesh1
        v2, f2 = mesh2
        v = v1 + v2
        offset = len(v1)
        f = f1 + [[i + offset, j + offset, k + offset] for i, j, k in f2]
        return v, f

    for pid, pgl_objects in pgl_scene.todict().items():
        sm[pid] = reduce(
            _concat_mesh,
            [shape_mesh(pgl_object, tesselator) for pgl_object in pgl_objects],
        )
    return sm


def to_mesh(shape):
    points, triangles = shape_mesh(shape)
    points = np.array(points, dtype=np.float32)
    triangles = np.array(triangles, dtype=np.uint16)
    return points, triangles


def unpack_string(component_type):
    """Returns the appropriate unpack string and size
    depending on the component type of gltf buffer.

    Args:
        component_type (int): The type of component of the buffer.

    Returns:
        tuple: a tuple containing the unpack string and the size.
    """
    string = ""
    size = 0
    match component_type:
        case 5120:  # signed char
            string = "<bbb"
            size = 1
        case 5121:  # unsigned char
            string = "<BBB"
            size = 1
        case 5122:  # signed short
            string = "<hhh"
            size = 2
        case 5123:  # unsigned short
            string = "<HHH"
            size = 2
        case 5125:  # unsigned int
            string = "<III"
            size = 4
        case 5126:  # float (signed)
            string = "<fff"
            size = 4
    return string, size


def rotate_shape(scene, quaternion, mesh_id):
    rotation = quaternion
    x = rotation[0]
    y = rotation[1]
    z = rotation[2]
    w = rotation[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = atan2(sinr_cosp, cosr_cosp)

    sinp = sqrt(1 + 2 * (w * y - x * z))
    cosp = sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * atan2(sinp, cosp) - pi / 2

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = atan2(siny_cosp, cosy_cosp)

    scene[mesh_id] = pgl.Shape(
        pgl.EulerRotated(roll, pitch, yaw, scene[mesh_id].geometry)
    )


def to_pgl(file, verbose=False) -> pgl.Scene:
    """convert a gltf file to a plantGL scene.

    Args:
        file (str): the path to the gltf file.
        verbose (bool, optional): verbose mode. Defaults to False.

    Returns:
        pgl.Scene: A plantGL scenes
    """
    gltf = pygltflib.GLTF2().load(file)
    scene = pgl.Scene()
    for mesh in gltf.meshes:
        vertices = []
        indices = []
        for primitive in mesh.primitives:
            # get the binary data for this mesh primitive from the buffer
            accessor = gltf.accessors[primitive.attributes.POSITION]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[buffer_view.buffer]
            try:
                data = gltf.get_data_from_buffer_uri(buffer.uri)
            except IndexError:  # File saved with different mime type than in pygltflib
                data = buffer.uri.split("data:application/gltf-buffer;base64,")[1]
                data = base64.decodebytes(bytes(data, "utf8"))

            # pull each vertex from the binary buffer and convert it into a tuple of python floats
            for i in range(accessor.count):
                comp_type = accessor.componentType
                string, size = unpack_string(comp_type)
                index = (
                    buffer_view.byteOffset + accessor.byteOffset + i * size * 3
                )  # the location in the buffer of this vertex
                d = data[index : index + size * 3]  # the vertex data
                v = struct.unpack(string, d)  # convert from base64 to three floats
                vertices.append(v)

                if verbose:
                    print(i, v)

            # triangles
            accessor = gltf.accessors[primitive.indices]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[buffer_view.buffer]
            try:
                data = gltf.get_data_from_buffer_uri(buffer.uri)
            except IndexError:  # File saved with different mime type than in pygltflib
                data = buffer.uri.split("data:application/gltf-buffer;base64,")[1]
                data = base64.decodebytes(bytes(data, "utf8"))
            # pull each vertex from the binary buffer and convert it into a tuple of python floats
            for i in range(int(accessor.count / 3)):
                comp_type = accessor.componentType
                string, size = unpack_string(comp_type)
                index = (
                    buffer_view.byteOffset + accessor.byteOffset + i * size * 3
                )  # the location in the buffer of this triangle
                d = data[index : index + size * 3]  # the index data
                v = struct.unpack(string, d)
                indices.append(v)
                if verbose:
                    print(i, v)
        ts = pgl.TriangleSet(vertices, indices)
        sh = pgl.Shape(ts)
        scene.add(sh)
    

    transform_all(gltf.nodes[0], scene, gltf)

    return scene


def get_rotation(node):
    """Gets the rotation of a node as a transformation matrix.

    Args:
        node (Node): The node to get the matrix from.

    Returns:
        np.array: The transform matrix as an numpy array.
    """
    local_rotation = [0,0,0,1]
    if node.rotation is not None:
        local_rotation = node.rotation

    x, y, z, w = local_rotation

    # Calculate the rotation matrix elements
    R = np.array(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y + 2 * w * z,
                2 * x * z - 2 * w * y,
                0.0,
            ],
            [
                2 * x * y - 2 * w * z,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z + 2 * w * x,
                0.0,
            ],
            [
                2 * x * z + 2 * w * y,
                2 * y * z - 2 * w * x,
                1 - 2 * x * x - 2 * y * y,
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return R


def get_scale(node):
    """Gets the scale of a node as a transformation matrix.

    Args:
        node (Node): The node to get the matrix from.

    Returns:
        np.array: The transform matrix as an numpy array.
    """
    # Get the node's local scale vector
    local_scale = np.ones(3)
    if node.scale is not None:
        local_scale = node.scale

    scale = np.identity(4)
    scale[0, 0] = local_scale[0]
    scale[1, 1] = local_scale[1]
    scale[2, 2] = local_scale[2]

    return scale


def get_translation(node):
    """Gets the translation of a node as a transformation matrix.

    Args:
        node (Node): The node

    Returns:
        np.array: The transform matrix as an numpy array.
    """
    # Get the node's local translation vector
    local_translation = np.zeros(3)
    if node.translation is not None:
        local_translation = node.translation

    translation = np.identity(4)
    translation[0, 3] = local_translation[0]
    translation[1, 3] = local_translation[1]
    translation[2, 3] = local_translation[2]

    return translation


def transform_all(node, scene, gltf, matrix=None):
    if matrix is None:
        matrix = np.eye(4)

    # Get the node's local transformation matrix
    local_matrix = np.eye(4)
    if node.matrix is not None:
        local_matrix = np.array(node.matrix).reshape(4, 4).T

    # Accumulate the transformation matrices
    global_matrix = matrix @ local_matrix

    local_translation = get_translation(node)
    local_rotation = get_rotation(node)
    local_scale = get_scale(node)

    local_transform = local_translation @ local_rotation @ local_scale

    global_matrix = global_matrix @ local_transform

    if node.mesh is not None:
        vertices = scene[node.mesh].geometry.pointList
        for i in range(len(vertices)):
            vertex = vertices[i]
            homogeneous_point = np.array([vertex[0], vertex[1], vertex[2], 1])

            transformed_point = np.dot(global_matrix, homogeneous_point)
            vertices[i] = pgl.Vector3(transformed_point[:3])
        scene[node.mesh].geometry.pointList = vertices

    for child in node.children:
        transform_all(gltf.nodes[child], scene, gltf, global_matrix)


class GLTFScene:
    def __init__(self, scene):
        self.scene = scene
        self._buffer_views = []
        self._accessors = []
        self._primitives = []
        self._meshes = []
        self._nodes = []
        self._blobs = []
        self._current_size = 0

    def run(self):
        uids = 0
        self._current_size = 0
        objs = self.scene.todict()
        for _, shapes in objs.items():
            for sh in shapes:
                points, triangles = to_mesh(sh)
                self.populate(points, triangles, uids)
                uids += 1

    def to_gltf(self, filename="toto.gltf"):
        if self._blobs:
            blobs = self._blobs[0]
            for b in self._blobs[1:]:
                blobs += b

        buffer = pygltflib.Buffer(byteLength=len(blobs))

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=list(range(len(self._nodes))))],
            nodes=self._nodes,
            meshes=self._meshes,
            accessors=self._accessors,
            bufferViews=self._buffer_views,
            buffers=[buffer],
        )

        gltf.set_binary_blob(blobs)

        gltf.save(filename)
        return True

    def populate(self, points, triangles, uids):
        triangles_binary_blob = triangles.flatten().tobytes()
        points_binary_blob = points.tobytes()

        offset = self._current_size

        triangle_buffer_view = pygltflib.BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(triangles_binary_blob),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
        array_buffer_view = pygltflib.BufferView(
            buffer=0,
            byteOffset=offset + len(triangles_binary_blob),
            byteLength=len(points_binary_blob),
            target=pygltflib.ARRAY_BUFFER,
        )

        triangle_access = pygltflib.Accessor(
            bufferView=2 * uids,
            byteOffset=0,
            componentType=pygltflib.UNSIGNED_SHORT,
            count=triangles.size,
            type=pygltflib.SCALAR,
            max=[int(triangles.max())],
            min=[int(triangles.min())],
        )
        points_access = pygltflib.Accessor(
            bufferView=2 * uids + 1,
            byteOffset=0,
            componentType=pygltflib.FLOAT,
            count=len(points),
            type=pygltflib.VEC3,
            max=points.max(axis=0).tolist(),
            min=points.min(axis=0).tolist(),
        )

        primitive = pygltflib.Primitive(
            attributes={"POSITION": 2 * uids + 1}, indices=2 * uids
        )

        mesh = pygltflib.Mesh(primitives=[primitive])
        node = pygltflib.Node(mesh=uids, name=str(uids))

        self._nodes.append(node)
        self._meshes.append(mesh)
        self._primitives.append(primitive)
        self._accessors.append(triangle_access)
        self._accessors.append(points_access)
        self._buffer_views.append(triangle_buffer_view)
        self._buffer_views.append(array_buffer_view)

        self._blobs.append(triangles_binary_blob + points_binary_blob)
        self._current_size += len(triangles_binary_blob) + len(points_binary_blob)
