from pathlib import Path

import numpy as np
import openalea.plantgl.all as pgl

from openalea.scenemanagement.convert import to_pgl, GLTFScene
from openalea.scenemanagement.project import Project
from oawidgets.plantgl import PlantGL
from alinea.caribu.CaribuScene import CaribuScene


class Model:
    def __init__(self):
        pass

    def run(self):
        pass

    def display(self):
        pass


class CaribuModel(Model):
    def __init__(self):
        super().__init__()
        self.scene = None
        self.options = None
        self._values = None

    def run(self):
        """Run Caribu on the scene."""
        cscene = CaribuScene(self.scene, scene_unit="m")
        raw, _ = cscene.run(direct=True, simplify=True)
        _, values = cscene.plot(raw["Eabs"], display=False)

        v99 = np.percentile(values, 99)
        nvalues = np.array(values)
        nvalues[nvalues > v99] = v99
        self._values = nvalues.tolist()

    @property
    def values(self):
        return self._values

    def display(self):
        return PlantGL(self.scene, group_by_color=False, property=self._values)


class Scene:
    def __init__(self, file: Path):
        self.file: Path = Path(file)
        self.sc = pgl.Scene()
        self.project = None
        self.models = []
        self.spec_prop = None
        self.light_sources = None
        self.environment = None
        self.sensors = None
        self._parse_scene_file()

    def add_model(self, model):
        """Add a light model for the scene.

        Args:
            model (Model): The model to add.
        """
        self.models.append(model)

    def add_models(self, models):
        """Add a list of light model for the scene.

        Args:
            models (list[Model]): The list of models to add.
        """
        self.models += models

    def read_gltf(self, file):
        """Reads a gltf file and adds it to the scene.

        Args:
            file (str): the gltf file to add.
        """
        scene = to_pgl(file)
        self.sc += scene

    def save(self, filename):
        """Saves the scene as a gltf file.
        Args:
            filename (str): the name of the file to write.
        """
        gltf = GLTFScene(self.sc)
        gltf.run()
        gltf.to_gltf(filename)

    def _parse_scene_file(self):
        """Parse the scene file and initialize all parameters."""
        self.project = Project(self.file)
        for plant in self.project.plant_architecture:
            self.read_gltf(plant)
        for background in self.project.background:
            self.read_gltf(background)

    def run(self):
        """Run caribu on the scene with the previous scene information.

        Returns:
            list: the caribu result
        """
        for model in self.models:
            model.scene = self.sc
            model.run()

    def display(self):
        """Displays the scene.

        Returns:
            oawidgets.plantgl.PlantGL: A 3d widget to display the scene
        """
        return PlantGL(self.sc)
