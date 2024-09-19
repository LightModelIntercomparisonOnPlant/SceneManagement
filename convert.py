import numpy
import numpy as np

from openalea.plantgl import all as pgl
from functools import reduce

import pygltflib

def shape_mesh(pgl_object, tesselator=None):
    if tesselator is None:
        tesselator = pgl.Tesselator()
    pgl_object.apply(tesselator)
    mesh = tesselator.triangulation
    if mesh:
        indices = list(map(tuple,mesh.indexList))
        pts = list(map(tuple,mesh.pointList))
    return pts, indices


def as_scene_mesh(pgl_scene):
    """ Transform a PlantGL scene / PlantGL shape dict to a scene_mesh"""
    tesselator = pgl.Tesselator()

    sm = {}

    def _concat_mesh(mesh1,mesh2):
        v1, f1 = mesh1
        v2, f2 = mesh2
        v = v1 + v2
        offset = len(v1)
        f = f1 + [[i + offset, j + offset, k + offset] for i, j, k in f2]
        return v, f

    for pid, pgl_objects in pgl_scene.todict().items():
        sm[pid] = reduce(_concat_mesh, [shape_mesh(pgl_object, tesselator) for pgl_object in
                       pgl_objects])
    return sm

def to_mesh(shape):

    points, triangles = shape_mesh(shape)
    points = np.array(points, dtype=np.float32)
    triangles = np.array(triangles, dtype=np.uint8)
    return points, triangles

def to_gltf(points, triangles):
    triangles_binary_blob = triangles.flatten().tobytes()
    points_binary_blob = points.tobytes()
    gltf = pygltflib.GLTF2(
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(mesh=0)],
        meshes=[
            pygltflib.Mesh(
                primitives=[
                    pygltflib.Primitive(
                        attributes=pygltflib.Attributes(POSITION=1), indices=0
                    )
                ]
            )
        ],
        accessors=[
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.UNSIGNED_BYTE,
                count=triangles.size,
                type=pygltflib.SCALAR,
                max=[int(triangles.max())],
                min=[int(triangles.min())],
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            ),
        ],
        bufferViews=[
            pygltflib.BufferView(
                buffer=0,
                byteLength=len(triangles_binary_blob),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(triangles_binary_blob),
                byteLength=len(points_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
        ],
        buffers=[
            pygltflib.Buffer(
                byteLength=len(triangles_binary_blob) + len(points_binary_blob)
            )
        ],
    )
    gltf.set_binary_blob(triangles_binary_blob + points_binary_blob)
    return gltf


def to_gltf_file(gltf, filename='toto.gltf'):
    gltf.save(filename)



class GLTFScene:
    def __init__(self, scene):
        self.scene = scene

        self._bufferViews = []
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
        for vid, shapes in objs.items():
            for sh in shapes:

                points, triangles = to_mesh(sh)
                self.populate(points, triangles, uids)
                uids += 1

    def to_gltf(self, filename):

        if self._blobs:
            blobs = self._blobs[0]
            for b in self._blobs[1:]:
                blobs += b

        buffer = pygltflib.Buffer(byteLength=len(blobs))

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes = [pygltflib.Scene(nodes=list(range(len(self._nodes))))],
            nodes = self._nodes,
            meshes=self._meshes,
            accessors=self._accessors,
            bufferViews=self._bufferViews,
            buffers=[buffer],
        )

        gltf.set_binary_blob(blobs)

        to_gltf_file(gltf, filename=filename)
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
                byteOffset=offset+len(triangles_binary_blob),
                byteLength=len(points_binary_blob),
                target=pygltflib.ARRAY_BUFFER,
            )


        triangle_access = pygltflib.Accessor(
                bufferView=2*uids,
                byteOffset = 0,
                componentType=pygltflib.UNSIGNED_BYTE,
                count=triangles.size,
                type=pygltflib.SCALAR,
                max=[int(triangles.max())],
                min=[int(triangles.min())],
            )
        points_access = pygltflib.Accessor(
                bufferView=2*uids+1,
                byteOffset = 0,
                componentType=pygltflib.FLOAT,
                count=len(points),
                type=pygltflib.VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist(),
            )

        primitive = pygltflib.Primitive(attributes={"POSITION": 2*uids+1}, indices=2*uids)

        mesh = pygltflib.Mesh(primitives=[primitive])
        node = pygltflib.Node(mesh=uids, name=str(uids))

        self._nodes.append(node)
        self._meshes.append(mesh)
        self._primitives.append(primitive)
        self._accessors.append(triangle_access)
        self._accessors.append(points_access)
        self._bufferViews.append(triangle_buffer_view)
        self._bufferViews.append(array_buffer_view)

        self._blobs.append(triangles_binary_blob + points_binary_blob)
        self._current_size += len(triangles_binary_blob) + len(points_binary_blob)


