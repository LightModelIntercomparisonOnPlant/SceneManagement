from openalea.plantgl.all import *
from convert import *

f1 = "models/leaf.geom"
f2 = "models/tree.geom"

s1 = Scene(f1)
s2 = Scene(f2)

# sm1 = as_scene_mesh(s1)

s3 = Scene()
sh1 = s1[0]
sh2 = Shape(geometry=Translated((1, 1, 1), sh1.geometry))
s3.add(sh1)
s3.add(sh2)
