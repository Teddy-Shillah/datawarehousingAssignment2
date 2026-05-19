
import pandas as pd
import os
import sqlite3
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RAW_FILE     = "superstore_raw.csv"       
OUTPUT_DIR   = "warehouse_output"         
DB_FILE      = f"{OUTPUT_DIR}/superstore_warehouse.db"  

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — EXTRACT
# ─────────────────────────────────────────────
print("\n📥 [EXTRACT] Loading raw data...")

try:
    df = pd.read_csv(RAW_FILE, encoding="latin-1")
except FileNotFoundError:
    print(f"❌ ERROR: Could not find '{RAW_FILE}'.")
    print("   Please download the dataset from Kaggle and rename it to 'superstore_raw.csv'")
    exit()

print(f"   ✅ Loaded {len(df):,} rows and {len(df.columns)} columns.")
print(f"   Columns found: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# STEP 2 — TRANSFORM
# ─────────────────────────────────────────────
print("🔧 [TRANSFORM] Cleaning data...\n")

# --- Standardize column names (strip spaces, lowercase) ---
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# --- Rename columns to standard names used in this script ---
df.rename(columns={
    "row_id":       "row_id",
    "order_id":     "order_id",
    "order_date":   "order_date",
    "ship_date":    "ship_date",
    "ship_mode":    "ship_mode",
    "customer_id":  "customer_id",
    "customer_name":"customer_name",
    "segment":      "segment",
    "country":      "country",
    "city":         "city",
    "state":        "state",
    "postal_code":  "postal_code",
    "region":       "region",
    "product_id":   "product_id",
    "category":     "category",
    "sub-category": "sub_category",
    "sub_category": "sub_category",
    "product_name": "product_name",
    "sales":        "sales",
    "quantity":     "quantity",
    "discount":     "discount",
    "profit":       "profit",
}, inplace=True)

# --- Fix date formats ---
df["order_date"] = pd.to_datetime(df["order_date"], dayfirst=False, errors="coerce")
df["ship_date"]  = pd.to_datetime(df["ship_date"],  dayfirst=False, errors="coerce")

# --- Data Quality Report ---
print("📋 DATA QUALITY REPORT (before cleaning):")
print(f"   Total rows           : {len(df):,}")
print(f"   Duplicate rows       : {df.duplicated().sum():,}")
print(f"   Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")

# --- Remove duplicates ---
before = len(df)
df.drop_duplicates(inplace=True)
print(f"   🗑️  Removed {before - len(df)} duplicate rows.")

# --- Drop rows where critical fields are missing ---
df.dropna(subset=["order_id", "customer_id", "product_id", "order_date", "sales"], inplace=True)
print(f"   🗑️  Rows after dropping critical nulls: {len(df):,}")

# --- Fill missing postal codes ---
df["postal_code"] = df["postal_code"].fillna("UNKNOWN").astype(str)

# --- Ensure numeric columns are correct types ---
df["sales"]    = pd.to_numeric(df["sales"],    errors="coerce").fillna(0).round(2)
df["profit"]   = pd.to_numeric(df["profit"],   errors="coerce").fillna(0).round(2)
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0).astype(int)
df["discount"] = pd.to_numeric(df["discount"], errors="coerce").fillna(0).round(2)

# --- Strip whitespace from text columns ---
for col in ["customer_name", "segment", "city", "state", "region", "category", "sub_category", "product_name"]:
    if col in df.columns:
        df[col] = df[col].str.strip()

print("   ✅ Data cleaned successfully.\n")

# ─────────────────────────────────────────────
# STEP 3 — BUILD STAR SCHEMA TABLES
# ─────────────────────────────────────────────
print("⭐ [LOAD] Building Star Schema tables...\n")

# ── DIM_CUSTOMER ──────────────────────────────
dim_customer = df[["customer_id", "customer_name", "segment"]].drop_duplicates()
dim_customer = dim_customer.reset_index(drop=True)
print(f"   👤 dim_customer  : {len(dim_customer):,} records")

# ── DIM_PRODUCT ───────────────────────────────
dim_product = df[["product_id", "product_name", "category", "sub_category"]].drop_duplicates()
dim_product = dim_product.reset_index(drop=True)
print(f"   📦 dim_product   : {len(dim_product):,} records")

# ── DIM_LOCATION ──────────────────────────────
dim_location = df[["postal_code", "city", "state", "region"]].drop_duplicates()
dim_location = dim_location.reset_index(drop=True)
dim_location.insert(0, "location_id", range(1, len(dim_location) + 1))
print(f"   🌍 dim_location  : {len(dim_location):,} records")

# ── DIM_DATE ──────────────────────────────────
all_dates = pd.concat([df["order_date"], df["ship_date"]]).dropna().drop_duplicates()
dim_date = pd.DataFrame({"full_date": all_dates})
dim_date = dim_date.drop_duplicates().reset_index(drop=True)
dim_date.insert(0, "date_id", range(1, len(dim_date) + 1))
dim_date["day"]       = dim_date["full_date"].dt.day
dim_date["month"]     = dim_date["full_date"].dt.month
dim_date["month_name"]= dim_date["full_date"].dt.strftime("%B")
dim_date["quarter"]   = dim_date["full_date"].dt.quarter
dim_date["year"]      = dim_date["full_date"].dt.year
dim_date["weekday"]   = dim_date["full_date"].dt.strftime("%A")
print(f"   📅 dim_date      : {len(dim_date):,} records")

# ── FACT_SALES ────────────────────────────────
# Merge location_id into main df
df = df.merge(dim_location[["location_id", "postal_code", "city", "state", "region"]],
              on=["postal_code", "city", "state", "region"], how="left")

# Merge date_id for order_date
df = df.merge(dim_date[["date_id", "full_date"]].rename(columns={"date_id": "order_date_id", "full_date": "order_date"}),
              on="order_date", how="left")

fact_sales = df[[
    "order_id",
    "customer_id",
    "product_id",
    "location_id",
    "order_date_id",
    "ship_mode",
    "sales",
    "quantity",
    "discount",
    "profit"
]].copy()

fact_sales.rename(columns={"order_date_id": "date_id"}, inplace=True)
fact_sales = fact_sales.reset_index(drop=True)
fact_sales.insert(0, "sale_id", range(1, len(fact_sales) + 1))

print(f"   💰 fact_sales    : {len(fact_sales):,} records\n")

# ─────────────────────────────────────────────
# SAVE TO CSV FILES
# ─────────────────────────────────────────────
print("💾 Saving tables as CSV files...")

dim_customer.to_csv(f"{OUTPUT_DIR}/dim_customer.csv",  index=False)
dim_product.to_csv( f"{OUTPUT_DIR}/dim_product.csv",   index=False)
dim_location.to_csv(f"{OUTPUT_DIR}/dim_location.csv",  index=False)
dim_date.to_csv(    f"{OUTPUT_DIR}/dim_date.csv",       index=False)
fact_sales.to_csv(  f"{OUTPUT_DIR}/fact_sales.csv",     index=False)

print("   ✅ CSV files saved in 'warehouse_output/' folder.")

# ─────────────────────────────────────────────
# SAVE TO SQLITE DATABASE
# ─────────────────────────────────────────────
print("\n🗄️  Loading into SQLite database...")

conn = sqlite3.connect(DB_FILE)

dim_customer.to_sql("dim_customer",  conn, if_exists="replace", index=False)
dim_product.to_sql( "dim_product",   conn, if_exists="replace", index=False)
dim_location.to_sql("dim_location",  conn, if_exists="replace", index=False)
dim_date.to_sql(    "dim_date",      conn, if_exists="replace", index=False)
fact_sales.to_sql(  "fact_sales",    conn, if_exists="replace", index=False)

conn.close()
print(f"   ✅ Database saved: {DB_FILE}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("🎉 ETL COMPLETE! Summary:")
print("="*55)
print(f"   dim_customer  → {len(dim_customer):>5,} rows  |  dim_customer.csv")
print(f"   dim_product   → {len(dim_product):>5,} rows  |  dim_product.csv")
print(f"   dim_location  → {len(dim_location):>5,} rows  |  dim_location.csv")
print(f"   dim_date      → {len(dim_date):>5,} rows  |  dim_date.csv")
print(f"   fact_sales    → {len(fact_sales):>5,} rows  |  fact_sales.csv")
print(f"\n   📁 All files saved in: warehouse_output/")
print(f"   🗄️  SQLite DB saved  : {DB_FILE}")
print("="*55)

