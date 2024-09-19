import numpy
import numpy as np

from openalea.plantgl import all as pgl
from functools import reduce

import pygltflib

def shape_mesh(pgl_shape, tesselator=None):
    if tesselator is None:
        tesselator = pgl.Tesselator()
    tesselator.process(pgl_shape)
    tset = tesselator.result
    return list(tset.pointList), list(tset.indexList)


def as_scene_mesh(pgl_scene):
    """ Transform a PlantGL scene / PlantGL shape dict to a scene_mesh"""
    tesselator = pgl.Tesselator()

    if isinstance(pgl_scene, pgl.Scene):
        sm = {}

        def _concat_mesh(mesh1,mesh2):
            v1, f1 = mesh1
            v2, f2 = mesh2
            v = numpy.array(v1.tolist() + v2.tolist())
            offset = len(v1)
            f = numpy.array(f1.tolist() + [[i + offset, j + offset, k + offset] for i, j, k
                               in f2.tolist()])
            return v, f

        for pid, pgl_objects in pgl_scene.todict().items():
            sm[pid] = reduce(_concat_mesh, [shape_mesh(pgl_object, tesselator) for pgl_object in
                           pgl_objects])
        return sm
    elif isinstance(pgl_scene, dict):
        return {sh_id: shape_mesh(sh,tesselator) for sh_id, sh in
                pgl_scene.iteritems()}
    else:
        return pgl_scene

def to_mesh(shape):

    points, triangles = shape_mesh(shape)
    points = np.array(points, dtype=np.float32)
    triangles = np.array(triangles, dtype=np.uint8)
    return points, triangles

def to_gltf(points, triangles)
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
    #gltf.set_binary_blob(triangles_binary_blob + points_binary_blob)
    return gltf


def to_gltf_file(gltf, filename='toto.gltf'):
    gltf.save(filename)



class GLTFScene:
    def __init__(scene):
        self.scene = scene

        self._buffers=[]
        self._bufferViews = []
        self._accessors = []
        self._meshes = []
        self._nodes = []

    def run(self):
        pass 



