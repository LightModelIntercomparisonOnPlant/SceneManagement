from openalea.plantgl.all import *
from convert import *

f1 = 'leaf.geom'
f2='tree.geom'

s1 = Scene(f1)
s2=Scene(f2)

sm1 = as_scene_mesh(s1)

