# IF3170_G58_ApertureScience

Tugas Besar 1 IF3170 Inteligensi Artifisial 2026/2027. Program mencari konfigurasi pemuatan paket pada peti kemas truk yang memaksimalkan total value, dengan batasan paket tidak melayang, tidak menembus dinding, tidak tumpang tindih, paket fragile tidak berada di bawah paket lain, dan berat muatan tidak melebihi kapasitas.

Algoritma yang diimplementasikan:

- Hill-climbing: Simple, Steepest-ascent, Stochastic, Sideways Move, Random-restart
- Simulated Annealing
- Genetic Algorithm

Bonus yang dikerjakan: objective function alternatif, seluruh varian Hill-climbing, dan multi-truk.

## Struktur

```
src/
  main.py             program utama
  algorithms/         hill climbing, simulated annealing, genetic algorithm
  models/             Package, Truck, State
  utils/              constraint, objective function, langkah neighbor, loader instance
  visualization/      visualisasi 3D 
data/instances/       file instance 
output/               hasil run 
docs/                 laporan
```

## Setup

Membutuhkan Python 3.10 atau lebih baru.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan Program

```
python src/main.py --algorithm <algoritma> [opsi]
```

| Opsi | Keterangan | Default |
|---|---|---|
| `--algorithm` | `simple`, `steepest`, `stochastic`, `sideways`, `random_restart`, `sa`, `ga` | `steepest` |
| `--instance` | path file instance JSON | `data/instances/sample.json` |
| `--objective` | `default`, `eta`, `density`, `volume` | `default` |
| `--runs` | banyak run; run ke-i memakai seed + i | `3` |
| `--seed` | seed awal | `42` |
| `--population-size` | jumlah populasi (GA) | `40` |
| `--iterations` | banyak iterasi atau generasi | bergantung algoritma |
| `--truck-count` | menggandakan truk pada instance satu truk menjadi N truk | - |
| `--output-dir` | folder hasil | `output` |
| `--interactive` | membuka viewer 3D setelah run (dengan `--runs 1`) | - |
| `--view-only` | hanya menampilkan state awal acak | - |

Contoh:

```
python src/main.py --algorithm sa --runs 3 --instance data/instances/dense.json
python src/main.py --algorithm ga --population-size 40 --iterations 100 --output-dir output/ga/pop40_iter100
python src/main.py --algorithm steepest --runs 1 --instance data/instances/sample_multitruck.json --interactive
```

Setiap run menyimpan `initial.png`, `final.png`, `objective.png`, dan `metrics.txt` di `output/<algoritma>/seed_<n>/`. Simulated Annealing juga menyimpan `acceptance.png` (plot e^(dE/T)).

## Format Instance

Satu truk memakai key `truck`, multi-truk memakai key `trucks` berisi daftar truk.

```json
{
  "truck": {"width": 10, "length": 20, "height": 5, "max_capacity": 500},
  "packages": [
    {"id": "P01", "width": 6, "length": 3, "height": 2, "value": 45, "weight": 17, "is_fragile": true, "eta": 4}
  ]
}
```
