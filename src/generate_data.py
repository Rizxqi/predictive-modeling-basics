"""Membuat data sintetis okupansi hotel harian (3 tahun).

Data ini BUATAN, dirancang punya pola yang mirip hotel sungguhan:
- trend naik pelan (bisnis bertumbuh)
- pola mingguan (akhir pekan lebih ramai)
- pola tahunan (libur sekolah Jun-Jul dan akhir tahun ramai)
- noise acak

Ganti dengan data hotelmu sendiri: cukup siapkan CSV dengan kolom
`date` dan `rooms_sold` (lihat README).
"""
import numpy as np
import pandas as pd

TOTAL_ROOMS = 120
SEED = 42


def generate(start="2022-01-01", end="2024-12-31"):
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    base = 62 + np.linspace(0, 8, n)                                # trend
    weekly = np.array([-6, -5, -3, 0, 6, 12, 3])[dates.dayofweek]   # Sen..Min
    doy = dates.dayofyear.values
    yearly = 8 * np.sin(2 * np.pi * (doy - 120) / 365.25)           # puncak sekitar Jul
    holiday = np.zeros(n)
    holiday[(dates.month == 12) & (dates.day >= 20)] = 15
    holiday[(dates.month == 1) & (dates.day <= 3)] = 12
    holiday[(dates.month == 7) & (dates.day <= 15)] = 6
    noise = rng.normal(0, 3.5, n)

    occ = np.clip(base + weekly + yearly + holiday + noise, 20, 100)  # persen
    rooms_sold = np.round(occ / 100 * TOTAL_ROOMS).astype(int)

    return pd.DataFrame({"date": dates, "rooms_sold": rooms_sold})


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/hotel_daily.csv", index=False)
    print(f"Tersimpan: data/hotel_daily.csv ({len(df)} baris)")
