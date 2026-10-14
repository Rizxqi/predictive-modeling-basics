"""Forecasting kamar terjual harian hotel.

Alur:
1. Muat data dan plot
2. Bagi data berurutan waktu (latih di awal, uji di akhir)
3. Bandingkan 4 model: Seasonal Naive (baseline), Holt-Winters, SARIMA,
   Gradient Boosting (ML dengan fitur lag)
4. Pilih model terbaik berdasarkan MAE
5. Latih ulang dengan semua data, forecast ke depan, simpan hasil

Jalankan:  python src/forecast.py --data data/hotel_daily.csv
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
OUT = Path("outputs")


# ---------- Data ----------
def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    s = df.set_index("date")["rooms_sold"].asfreq("D")
    if s.isna().any():  # tanggal kosong diisi interpolasi
        s = s.interpolate()
    return s.astype(float)


def plot_overview(s):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    s.plot(ax=ax[0], lw=0.8)
    s.rolling(30).mean().plot(ax=ax[0], lw=2, color="red", label="rata-rata 30 hari")
    ax[0].set_title("Kamar terjual harian (trend dan musiman tahunan)")
    ax[0].legend()
    dow = s.groupby(s.index.dayofweek).mean()
    dow.index = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
    dow.plot.bar(ax=ax[1], color="steelblue")
    ax[1].set_title("Rata-rata per hari dalam seminggu (pola mingguan)")
    plt.tight_layout()
    plt.savefig(OUT / "01_eksplorasi.png", dpi=130)
    plt.close()


# ---------- Metrik ----------
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


# ---------- Model ----------
def seasonal_naive(train, h, season=7):
    """Baseline: ulangi pola 7 hari terakhir."""
    last = train.values[-season:]
    vals = np.tile(last, int(np.ceil(h / season)))[:h]
    return vals


def holt_winters(train, h):
    m = ExponentialSmoothing(train, trend="add", seasonal="add",
                             seasonal_periods=7).fit()
    return m.forecast(h).values


def sarima(train, h):
    m = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
    return m.forecast(h).values


def _calendar_features(idx):
    f = pd.DataFrame(index=idx)
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["doy_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["is_year_end"] = (((idx.month == 12) & (idx.day >= 20)) |
                        ((idx.month == 1) & (idx.day <= 3))).astype(int)
    return f


def _lag_features(history, date):
    """Fitur dari nilai masa lalu (hanya yang sudah diketahui di `date`)."""
    return {
        "lag_1": history[date - pd.Timedelta(days=1)],
        "lag_7": history[date - pd.Timedelta(days=7)],
        "lag_14": history[date - pd.Timedelta(days=14)],
        "lag_28": history[date - pd.Timedelta(days=28)],
        "roll_7": history[date - pd.Timedelta(days=7):
                          date - pd.Timedelta(days=1)].mean(),
    }


def gradient_boosting(train, h):
    cal = _calendar_features(train.index)
    rows = []
    for d in train.index[28:]:
        rows.append(_lag_features(train, d))
    lag = pd.DataFrame(rows, index=train.index[28:])
    X = pd.concat([cal.loc[lag.index], lag], axis=1)
    y = train.loc[lag.index]
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, random_state=0)
    model.fit(X, y)

    # Forecast rekursif: hasil prediksi hari ini dipakai sebagai lag besok
    history = train.copy()
    future_idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=h)
    preds = []
    for d in future_idx:
        feat = pd.concat([_calendar_features(pd.DatetimeIndex([d])).iloc[0],
                          pd.Series(_lag_features(history, d))])
        p = model.predict(feat[X.columns].values.reshape(1, -1))[0]
        preds.append(p)
        history.loc[d] = p
    return np.array(preds)


MODELS = {
    "Seasonal Naive": seasonal_naive,
    "Holt-Winters": holt_winters,
    "SARIMA": sarima,
    "Gradient Boosting": gradient_boosting,
}


# ---------- Pipeline ----------
def main(args):
    OUT.mkdir(exist_ok=True)
    s = load_data(args.data)
    print(f"Data: {s.index[0].date()} s/d {s.index[-1].date()} ({len(s)} hari)")
    plot_overview(s)

    # Bagi data BERURUTAN WAKTU (jangan diacak!)
    train, test = s[:-args.test_days], s[-args.test_days:]
    print(f"Latih: {len(train)} hari | Uji: {len(test)} hari\n")

    results, preds = {}, {}
    for name, fn in MODELS.items():
        preds[name] = fn(train, len(test))
        results[name] = metrics(test.values, preds[name])
    res = pd.DataFrame(results).T.sort_values("MAE").round(2)
    res.to_csv(OUT / "metrics.csv")
    print("Perbandingan model (data uji):")
    print(res.to_string(), "\n")

    # Plot perbandingan
    plt.figure(figsize=(11, 5))
    test_plot = s[-args.test_days - 60:]
    plt.plot(test_plot.index, test_plot.values, color="black", lw=1.5, label="Aktual")
    for name, p in preds.items():
        plt.plot(test.index, p, lw=1.2, label=f"{name} (MAE={res.loc[name, 'MAE']})")
    plt.axvline(test.index[0], color="gray", ls="--")
    plt.title("Perbandingan model pada data uji")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "02_perbandingan_model.png", dpi=130)
    plt.close()

    # Forecast final dengan model terbaik, dilatih pada SEMUA data
    best = res.index[0]
    print(f"Model terbaik: {best}. Forecast {args.horizon} hari ke depan...")
    fc_vals = MODELS[best](s, args.horizon)
    fc_idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=args.horizon)
    fc = pd.DataFrame({"date": fc_idx,
                       "forecast_rooms_sold": np.round(fc_vals, 1)})
    fc.to_csv(OUT / "forecast.csv", index=False)

    plt.figure(figsize=(11, 5))
    hist = s[-90:]
    plt.plot(hist.index, hist.values, color="black", label="Historis")
    plt.plot(fc_idx, fc_vals, color="crimson", label=f"Forecast ({best})")
    plt.title(f"Forecast {args.horizon} hari ke depan")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "03_forecast.png", dpi=130)
    plt.close()
    print(f"Selesai. Hasil ada di folder {OUT}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hotel_daily.csv")
    ap.add_argument("--test-days", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=90)
    main(ap.parse_args())
