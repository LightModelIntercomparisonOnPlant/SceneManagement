import json

from dataclasses import dataclass

from pathlib import Path


@dataclass
class Project:
    """Class containing all data for a light simulation project."""

    description: str = ""
    plant_architecture: list[str] = None
    background: list[str] = None
    sensors: list[str] = None
    scale: list[str] = None
    spectral_band: dict = None
    spectral_properties: dict = None
    illumination: dict = None
    measurement: dict = None

    def __init__(self, file: Path):
        if file.exists():
            self.parse_project_file(file)

    def parse_project_file(self, file):
        """Parse a project file.

        Args:
            file (str): file path of the project description
        """
        with open(file, encoding="UTF8") as f:
            data = json.load(f)
        self.description = data["Description"]
        self.plant_architecture = data["Plant_Architecture"]
        self.background = data["Background"]
        self.sensors = data["Sensors"]
        self.scale = data["Scale"]
        self.spectral_band = data["Spectral_Band"]
        self.spectral_properties = data["Spectral_Properties"]
        self.illumination = data["Illumination"]
        self.measurement = data["Measurements"]
