
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "hotels.csv"
OUTPUT_DIR = ROOT / "outputs" / "real_data"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE, low_memory=False)
    df.columns = df.columns.str.strip()

    required = {
        "hotel",
        "is_canceled",
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}")

    month_map = {
        month: number
        for number, month in enumerate(
            [
                "January", "February", "March", "April",
                "May", "June", "July", "August",
                "September", "October", "November", "December",
            ],
            start=1,
        )
    }

    df["arrival_month"] = df["arrival_date_month"].map(month_map)

    df["arrival_date"] = pd.to_datetime(
        {
            "year": df["arrival_date_year"],
            "month": df["arrival_month"],
            "day": df["arrival_date_day_of_month"],
        },
        errors="coerce",
    )

    # Reservasi batal tidak dihitung sebagai kedatangan aktual.
    actual = df.loc[
        (df["is_canceled"] == 0) & df["arrival_date"].notna()
    ].copy()

    # Pilih City Hotel sebagai studi kasus pertama.
    city = actual.loc[actual["hotel"] == "City Hotel"]

    daily = (
        city.groupby("arrival_date")
        .size()
        .rename("reservations")
        .sort_index()
    )

    # Lengkapi kalender agar tanggal tanpa kedatangan tercatat sebagai nol.
    full_index = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D",
    )
    daily = daily.reindex(full_index, fill_value=0)
    daily.index.name = "date"

    output_file = OUTPUT_DIR / "city_hotel_daily.csv"
    daily.rename_axis("date").rename("reservations").to_csv(output_file)

    print(f"Total baris sumber: {len(df):,}")
    print(f"Reservasi tidak batal: {len(actual):,}")
    print(f"Rentang tanggal: {daily.index.min().date()} sampai "
          f"{daily.index.max().date()}")
    print(f"Jumlah hari: {len(daily):,}")
    print(f"Rata-rata reservasi per hari: {daily.mean():.2f}")
    print(f"Jumlah hari bernilai nol: {(daily == 0).sum()}")
    print(f"File output: {output_file}")


if __name__ == "__main__":
    main()
