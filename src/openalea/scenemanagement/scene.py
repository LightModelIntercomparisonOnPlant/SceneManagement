import json
import numpy as np
import openalea.plantgl.all as pgl
from openalea.scenemanagement.convert import to_pgl
from oawidgets.plantgl import PlantGL
from alinea.caribu.CaribuScene import CaribuScene


class Scene:
    def __init__(self, file):
        self.file = file
        self.sc = pgl.Scene()
        self.desc = None
        self.spec_prop = None
        self.light_sources = None
        self.environment = None
        self.sensors = None
        self._parse_scene_file()

    def read_gltf(self, file):
        """Reads a gltf file and adds it to the scene

        Args:
            file (str): the gltf file to add
        """
        scene = to_pgl(file)
        self.sc += scene

    def _parse_scene_file(self):
        """Parse the scene file and initialize all parameters."""
        with open(self.file, encoding="UTF8") as f:
            self.desc = json.load(f)
        plants = self.desc["Plant Architecture"]
        for plant in plants:
            self.read_gltf(plant)
        backgrounds = self.desc["Background"]
        for background in backgrounds:
            self.read_gltf(background)
        self.spec_prop = self.desc["Spectral_Properties"]

    def run(self):
        """Run caribu on the scene with the previous scene information

        Returns:
            list: the caribu result
        """
        cscene = CaribuScene(self.sc, scene_unit="m")
        raw, _ = cscene.run(direct=True, simplify=True)
        _, values = cscene.plot(raw["Eabs"], display=False)

        v99 = np.percentile(values, 99)
        nvalues = np.array(values)
        nvalues[nvalues > v99] = v99
        values = nvalues.tolist()

        return values

    def display(self):
        """Displays the scene.

        Returns:
            oawidgets.plantgl.PlantGL: A 3d widget to display the scene
        """
        return PlantGL(self.sc)
