import json

from dataclasses import dataclass

from pathlib import Path


@dataclass
class Project:
    """Class containing all data for a light simulation project."""

    description: str = ""
    objects = None
    sensors: list[str] = None
    scale: list[str] = None
    scene: list[object] = None
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
            file (Path): file path of the project description
        """
        with open(file, encoding="UTF8") as f:
            data = json.load(f)
        self.description = data["description"]
        self.objects = data["objects"]
        self.scene = data["scene"]
        self.spectral_band = data["Spectral_Band"]
        self.spectral_properties = data["Spectral_Properties"]
        self.illumination = data["Illumination"]
        self.measurement = data["Measurements"]
