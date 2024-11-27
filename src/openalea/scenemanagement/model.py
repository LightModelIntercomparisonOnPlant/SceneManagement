from alinea.caribu.CaribuScene import CaribuScene
from openalea.photonmap.simulator import Simulator
from openalea.photonmap import Vec3
from openalea.plantgl.all import *
from oawidgets.plantgl import PlantGL
import numpy as np

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

class PhmapModel(Model):
    def __init__(self):
        super().__init__()
        self.simulator = None
        self.scene = None
        self.options = None
        
    def run(self):
        self.simulator = Simulator()
        self.simulator.nb_photons = 100000
        self.simulator.max_depth = 5
        self.simulator.nb_thread = 4
        self.simulator.resetScene()

        for sh in self.scene:
            self.simulator.addFaceSensor(sh)

        
        self.simulator.addPointLight(Vec3(0,0,12), 1000, Vec3(1,1,1))

        # self.simulator.addSpotLight(Vec3(0,0,12), 10000, Vec3(0,0,-1), 
        #                60, Vec3(1,1,1))

        self.simulator.run()

    def display(self):
        return self.simulator.visualizeResults('oawidgets')
