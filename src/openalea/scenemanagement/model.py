import numpy as np
from alinea.caribu.CaribuScene import CaribuScene
from oawidgets.plantgl import PlantGL
from openalea.spice import Vec3
from openalea.spice.common.convert import pgl_to_spice
from openalea.spice.simulator import Simulator


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
        cscene = CaribuScene(self.scene)
        raw, _ = cscene.run(direct=True, simplify=True)
        _, values = cscene.plot(raw["Ei"], display=False)
        self._values = values

    @property
    def values(self):
        return self._values

    def display(self):
        return PlantGL(self.scene, group_by_color=False, property=self._values)


class SpiceModel(Model):
    def __init__(self):
        super().__init__()
        self.simulator = None
        self.scene = None
        self.options = None

    def run(self):
        self.simulator = Simulator()
        self.simulator.configuration.NB_PHOTONS = int(1e6)
        self.simulator.configuration.SCALE_FACTOR = 1
        self.simulator.configuration.MAXIMUM_DEPTH = 5
        self.simulator.configuration.T_MIN = 0.01
        self.simulator.configuration.BACKFACE_CULLING = False
        self.simulator.configuration.KEEP_ALL = True

        sp_scene = pgl_to_spice(self.scene)
        self.simulator.scene_pgl = self.scene
        self.simulator.scene = sp_scene

        self.simulator.addPointLight(Vec3(0, 0, 12), 1000, Vec3(1, 1, 1))

        self.simulator.run()

    def display(self, mode="mesh"):
        if mode == "mesh":
            return self.simulator.visualizeResults("oawidgets")
        if mode == "photon":
            return self.simulator.visualizePhotons("oawidgets")
