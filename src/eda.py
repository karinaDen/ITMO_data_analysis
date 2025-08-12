import os, math, json, datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
REP_DIR = Path(__file__).resolve().parents[1] / "reports"
FIG_DIR.mkdir(exist_ok=True, parents=True)
REP_DIR.mkdir(exist_ok=True, parents=True)

TX = DATA_DIR / "transaction_fraud_data.parquet"
FX = DATA_DIR / "historical_currency_exchange.parquet"

def load_parquet(path, columns=None):
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        try:
            import duckdb
            con = duckdb.connect(database=":memory:")
            if columns:
                cols = ", ".join(columns)
            else:
                cols = "*"
            return con.execute(f"SELECT {cols} FROM read_parquet('{path.as_posix()}')").fetchdf()
        except Exception as e:
            raise RuntimeError(f"Cannot read {path}: {e}")

def to_utc(ts):
    """
    Преобразует временные метки в формат UTC, что важно для корректного анализа временных паттернов
    """
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    return ts

def save_fig(name):
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    return out

def main():
    tx_cols = [
        "transaction_id", "customer_id", "timestamp", "amount", "currency",
        "country", "city", "vendor", "vendor_type",
        "device", "card_type",
        "is_high_risk_vendor", "is_fraud",
        "last_hour_activity"
    ]
    #Загружаются только нужные столбцы.
    tx = load_parquet(TX, columns=[c for c in tx_cols if True])
    #Удаляются строки с пропущенными ключевыми значениями.
    tx = tx.dropna(subset=["timestamp", "amount", "currency"])
    #Временные метки приводятся к единому формату.
    tx["timestamp"] = to_utc(tx["timestamp"])
    tx = tx.dropna(subset=["timestamp"])
    # Добавляется столбец с датой для дальнейшей агрегации.
    tx["date"] = tx["timestamp"].dt.date

    amount_usd_col = "amount_usd"
    #Если есть файл с историей курсов валют, суммы транзакций пересчитываются в USD по курсу на дату транзакции
    #Это позволяет корректно сравнивать суммы между странами и валютами, что важно для анализа и построения моделей
    if FX.exists():
        fx = load_parquet(FX)
        if "date" in fx.columns:
            fx_long = fx.melt(id_vars=["date"], var_name="currency", value_name="per_usd")
            fx_long["date"] = pd.to_datetime(fx_long["date"]).dt.date
            tx = tx.merge(fx_long, on=["date","currency"], how="left")
            tx[amount_usd_col] = np.where(tx["currency"].eq("USD"),
                                          tx["amount"],
                                          tx["amount"] / tx["per_usd"])
        else:
            tx[amount_usd_col] = tx["amount"]
    else:
        tx[amount_usd_col] = tx["amount"]

    # --- Basic KPIs ---
    #Считается общее число транзакций, число мошеннических и их доля (fraud rate) 
    total = len(tx)
    fraud = int(tx["is_fraud"].sum()) if "is_fraud" in tx.columns else None
    fraud_rate = float(tx["is_fraud"].mean()*100) if "is_fraud" in tx.columns else None

    # --- Plots ---
    if "is_fraud" in tx.columns and "country" in tx.columns:
        fr_by_country = tx.groupby("country")["is_fraud"].mean().sort_values(ascending=False).head(20)
        fr_by_country.plot(kind="bar")
        save_fig("fraud_rate_by_country.png")

    # Amount distrib (USD if available)
    tx[amount_usd_col].dropna().plot(kind="hist", bins=50, alpha=0.7)
    save_fig("amount_usd_hist.png")

    # Fraud rate by hour of day
    if "is_fraud" in tx.columns:
        tx["hour"] = tx["timestamp"].dt.hour
        tx.groupby("hour")["is_fraud"].mean().plot(kind="line", marker="o")
        save_fig("fraud_rate_by_hour.png")

    # High-risk vendors: fraud rate
    if "is_high_risk_vendor" in tx.columns and "is_fraud" in tx.columns:
        tx.groupby("is_high_risk_vendor")["is_fraud"].mean().plot(kind="bar")
        save_fig("fraud_rate_high_risk_flag.png")

    # Vendor type: top categories fraud rate
    if "vendor_type" in tx.columns and "is_fraud" in tx.columns:
        top_types = tx["vendor_type"].value_counts().head(12).index
        tx[tx["vendor_type"].isin(top_types)].groupby("vendor_type")["is_fraud"].mean().sort_values(ascending=False).plot(kind="bar")
        save_fig("fraud_rate_by_vendor_type_top12.png")

    # --- Concise Summary Report ---
    lines = []
    lines.append(f"# EDA Summary ({datetime.date.today().isoformat()})")
    lines.append("")
    lines.append(f"- Total transactions: **{total}**")
    if fraud_rate is not None:
        lines.append(f"- Fraudulent transactions: **{fraud}**")
        lines.append(f"- Fraud rate: **{fraud_rate:.1f}%**")
    if amount_usd_col in tx.columns:
        lines.append(f"- Amount normalized column: `{amount_usd_col}`")

    # Top-5 countries by fraud count
    if "is_fraud" in tx.columns and "country" in tx.columns:
        top5 = (tx.loc[tx["is_fraud"]==True, "country"].value_counts().head(5))
        lines.append("")
        lines.append("## Top-5 countries by fraud count")
        for c, v in top5.items():
            lines.append(f"- {c}: {v}")

    # Q60 style: median unique merchants last hour (if struct present)
    if "last_hour_activity" in tx.columns:
        try:
            uniq = tx["last_hour_activity"].apply(lambda x: x.get("unique_merchants") if isinstance(x, dict) else None)
            tmp = tx[["customer_id"]].copy()
            tmp["uniq_last_hour"] = uniq
            tmp = tmp.dropna(subset=["uniq_last_hour"])
            med = tmp.groupby("customer_id")["uniq_last_hour"].median()
            thr = med.quantile(0.95, interpolation="linear")
            count_strict = int((med > thr).sum())
            lines.append("")
            lines.append("## Unique merchants in last hour (dataset struct)")
            lines.append(f"- Customers strictly above 95th percentile: **{count_strict}**")
        except Exception as e:
            lines.append("")
            lines.append("## Unique merchants in last hour")
            lines.append(f"- Could not parse struct: {e}")

    REP_DIR.joinpath("eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", REP_DIR / "eda_summary.md")

if __name__ == "__main__":
    main()
