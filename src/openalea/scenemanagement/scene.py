import os.path
from pathlib import Path

import openalea.plantgl.all as pgl

from openalea.scenemanagement.convert import to_pgl, GLTFScene
from openalea.scenemanagement.project import Project
from oawidgets.plantgl import PlantGL


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
        return scene

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
        # for plant in self.project.plants:
        #     archi = plant['architecture']
        #     self.read_gltf(archi)
        # for env in self.project.environment:
        #     archi = env['architecture']
        #     self.read_gltf(archi)

        for instance in self.project.instances:
            obj = self.project.objects[instance['object']]
            archi = obj['architecture']
            pos = instance['position']
            if os.path.exists(archi):
                sc = self.read_gltf(archi)
                for sh in sc:
                    sh.geometry = pgl.Translated(pgl.Vector3(pos[0],pos[1],pos[2]),
                                        sh.geometry)
                self.sc.add(sc)
            elif archi.startswith("$"):
                # TODO: Find a common way to add those objects.
                pass


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
