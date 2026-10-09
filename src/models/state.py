from dataclasses import dataclass
from .package import Package

ORIENTATIONS = (
    (0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0),
)

@dataclass
class PackagePlacement:
    package: Package
    position: tuple[int, int, int] | None
    orientation: tuple[int, int, int] = (0, 1, 2)

    @property
    def dimensions(self):
        return tuple(self.package.dimensions[i] for i in self.orientation)

@dataclass
class State:
    placements: list[PackagePlacement]

    def clone(self):
        return State([
            PackagePlacement(p.package, p.position, p.orientation)
            for p in self.placements
        ])

    def __repr__(self):
        return "\\n".join(
            f"{p.package.id}: position={p.position}, dimensions={p.dimensions}, "
            f"orientation={p.orientation}"
            for p in self.placements
        )
