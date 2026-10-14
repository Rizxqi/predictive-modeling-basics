"""Membuat gambar penjelas konsep forecasting dari data hotel.

Jalankan:  python src/explain_concepts.py
Gambar tersimpan di docs/images/ dan dipakai oleh docs/PANDUAN_FORECASTING.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

IMG = Path("docs/images")
IMG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/hotel_daily.csv", parse_dates=["date"])
s = df.set_index("date")["rooms_sold"].asfreq("D").astype(float)

# 1. Dekomposisi: data = trend + musiman + sisa
dec = seasonal_decompose(s, model="additive", period=7)
fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
for a, (title, ser) in zip(ax, [("Data asli", s), ("Trend (arah jangka panjang)", dec.trend),
                                 ("Musiman mingguan (pola berulang)", dec.seasonal),
                                 ("Sisa / noise (tidak bisa diprediksi)", dec.resid)]):
    a.plot(ser, lw=0.8)
    a.set_title(title, loc="left", fontsize=10)
plt.tight_layout()
plt.savefig(IMG / "dekomposisi.png", dpi=120)
plt.close()

# 2. Autokorelasi
fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(s, lags=40, ax=ax)
ax.set_title("Autokorelasi: seberapa mirip data hari ini dengan data N hari lalu")
ax.set_xlabel("Selisih hari (lag)")
plt.tight_layout()
plt.savefig(IMG / "autokorelasi.png", dpi=120)
plt.close()

# 3. Ilustrasi split berurutan vs acak
fig, ax = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
n = len(s)
cut = n - 60
ax[0].plot(s.index[:cut], s.values[:cut], color="steelblue", lw=0.8, label="Latih")
ax[0].plot(s.index[cut:], s.values[cut:], color="crimson", lw=1.2, label="Uji")
ax[0].set_title("BENAR: latih di masa lalu, uji di masa depan", loc="left", fontsize=10)
ax[0].legend(loc="upper left")
rng = np.random.default_rng(1)
mask = rng.random(n) < 0.2
ax[1].scatter(s.index[~mask], s.values[~mask], s=2, color="steelblue")
ax[1].scatter(s.index[mask], s.values[mask], s=4, color="crimson")
ax[1].set_title("SALAH: diacak, model 'mengintip' masa depan (data leakage)", loc="left", fontsize=10)
plt.tight_layout()
plt.savefig(IMG / "split_waktu.png", dpi=120)
plt.close()

# 4. Pola mingguan dan bulanan
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
dow = s.groupby(s.index.dayofweek).mean()
ax[0].bar(["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"], dow.values, color="steelblue")
ax[0].set_ylim(60, dow.max() + 5)
ax[0].set_title("Rata-rata per hari dalam seminggu")
mon = s.groupby(s.index.month).mean()
ax[1].plot(mon.index, mon.values, marker="o", color="darkorange")
ax[1].set_xticks(range(1, 13))
ax[1].set_title("Rata-rata per bulan")
plt.tight_layout()
plt.savefig(IMG / "pola_mingguan_bulanan.png", dpi=120)
plt.close()

# Angka ringkas untuk dipakai di panduan
from statsmodels.tsa.stattools import acf
a = acf(s, nlags=28)
print("ACF lag1=%.2f lag7=%.2f lag14=%.2f lag3=%.2f" % (a[1], a[7], a[14], a[3]))
print("rata2 per hari:", dow.round(1).to_dict())
print("rata2 per bulan:", mon.round(1).to_dict())
print("trend awal vs akhir:", round(dec.trend.dropna().iloc[0], 1), round(dec.trend.dropna().iloc[-1], 1))
print("std sisa:", round(dec.resid.std(), 2))
