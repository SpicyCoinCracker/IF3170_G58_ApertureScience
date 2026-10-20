import json
from pathlib import Path

from models.package import Package
from models.truck import Truck


def load_instance(path: Path | str) -> tuple[Truck | list[Truck], list[Package]]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if "trucks" in data:
        truck_list: list[Truck] = []
        for idx, t_data in enumerate(data["trucks"]):
            truck_list.append(
                Truck(
                    id=str(t_data.get("id", f"Truck_{idx+1}")),
                    width=int(t_data["width"]),
                    length=int(t_data["length"]),
                    height=int(t_data["height"]),
                    max_capacity=float(t_data["max_capacity"]),
                )
            )
        truck_res: Truck | list[Truck] = truck_list
    else:
        truck_data = data["truck"]
        truck_res = Truck(
            id=str(truck_data.get("id", "Truck")),
            width=int(truck_data["width"]),
            length=int(truck_data["length"]),
            height=int(truck_data["height"]),
            max_capacity=float(truck_data["max_capacity"]),
        )

    packages: list[Package] = []
    for raw in data["packages"]:
        packages.append(
            Package(
                id=str(raw["id"]),
                width=int(raw["width"]),
                length=int(raw["length"]),
                height=int(raw["height"]),
                value=float(raw["value"]),
                weight=float(raw["weight"]),
                is_fragile=bool(raw.get("is_fragile", False)),
                eta=int(raw.get("eta", 0)),
            )
        )

    return truck_res, packages

