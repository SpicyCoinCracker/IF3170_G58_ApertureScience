from dataclasses import dataclass

@dataclass(frozen=True)
class Package:
    id: str
    width: int
    length: int
    height: int
    value: float
    weight: float
    is_fragile: bool = False
    eta: int = 0

    def __post_init__(self):
        if min(self.width, self.length, self.height) <= 0:
            raise ValueError("Package dimensions must be positive.")
        if self.weight < 0:
            raise ValueError("Package weight cannot be negative.")

    @property
    def dimensions(self):
        return (self.width, self.length, self.height)
