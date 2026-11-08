"""Real Romanian localities where the synthetic prosumers live.

Mostly peri-urban communes around big cities, where residential PV is concentrated. Coordinates are approximate
locality centres (WGS84). #9 can import LOCALITIES to download ERA5 / Open-Meteo weather for exactly these points.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Locality:
    id: str
    name: str
    county: str
    lat: float
    lon: float


LOCALITIES: list[Locality] = [
    # Ilfov (Bucharest ring)
    Locality("voluntari", "Voluntari", "Ilfov", 44.490, 26.180),
    Locality("otopeni", "Otopeni", "Ilfov", 44.550, 26.070),
    Locality("chiajna", "Chiajna", "Ilfov", 44.460, 25.975),
    Locality("bragadiru", "Bragadiru", "Ilfov", 44.371, 25.975),
    Locality("popesti_leordeni", "Popești-Leordeni", "Ilfov", 44.380, 26.170),
    Locality("corbeanca", "Corbeanca", "Ilfov", 44.600, 26.050),
    # Cluj
    Locality("floresti", "Florești", "Cluj", 46.747, 23.490),
    Locality("apahida", "Apahida", "Cluj", 46.808, 23.740),
    Locality("baciu", "Baciu", "Cluj", 46.790, 23.525),
    # Timiș
    Locality("dumbravita", "Dumbrăvița", "Timiș", 45.795, 21.243),
    Locality("giroc", "Giroc", "Timiș", 45.694, 21.238),
    Locality("ghiroda", "Ghiroda", "Timiș", 45.763, 21.300),
    Locality("mosnita_noua", "Moșnița Nouă", "Timiș", 45.718, 21.320),
    # Bihor
    Locality("sanmartin", "Sânmartin", "Bihor", 47.010, 21.958),
    Locality("osorhei", "Oșorhei", "Bihor", 47.033, 22.042),
    # Brașov
    Locality("ghimbav", "Ghimbav", "Brașov", 45.663, 25.508),
    Locality("sanpetru", "Sânpetru", "Brașov", 45.710, 25.633),
    Locality("harman", "Hărman", "Brașov", 45.718, 25.683),
    # Iași
    Locality("miroslava", "Miroslava", "Iași", 47.150, 27.520),
    Locality("valea_lupului", "Valea Lupului", "Iași", 47.180, 27.500),
    # Constanța
    Locality("valu_lui_traian", "Valu lui Traian", "Constanța", 44.165, 28.470),
    Locality("ovidiu", "Ovidiu", "Constanța", 44.270, 28.560),
    Locality("cumpana", "Cumpăna", "Constanța", 44.112, 28.556),
    # Prahova
    Locality("blejoi", "Blejoi", "Prahova", 44.980, 26.020),
    Locality("paulesti", "Păulești", "Prahova", 45.000, 25.990),
    # Sibiu
    Locality("selimbar", "Șelimbăr", "Sibiu", 45.765, 24.200),
    Locality("cisnadie", "Cisnădie", "Sibiu", 45.713, 24.150),
    # Argeș
    Locality("bradu", "Bradu", "Argeș", 44.820, 24.890),
    Locality("maracineni", "Mărăcineni", "Argeș", 44.900, 24.880),
    # Dolj
    Locality("isalnita", "Ișalnița", "Dolj", 44.390, 23.740),
    Locality("pielesti", "Pielești", "Dolj", 44.340, 23.960),
    # Galați
    Locality("sendreni", "Șendreni", "Galați", 45.400, 27.920),
    # Bacău
    Locality("magura", "Măgura", "Bacău", 46.560, 26.860),
]

BY_ID = {loc.id: loc for loc in LOCALITIES}
