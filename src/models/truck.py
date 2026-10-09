from dataclasses import dataclass

@dataclass(frozen=True)
class Truck:
    width: int
    length: int
    height: int
    max_capacity: float

    def __post_init__(self):
        if min(self.width, self.length, self.height) <= 0:
            raise ValueError("Truck dimensions must be positive.")
        if self.max_capacity < 0:
            raise ValueError("Truck capacity cannot be negative.")
