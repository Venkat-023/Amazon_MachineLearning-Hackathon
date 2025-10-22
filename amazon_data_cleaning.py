import pandas as pd
import numpy as np
import re

df=pd.read_csv("dataset.csv")
df['catalog_content'] = df['catalog_content'].astype(str)

df['item_name'] = df['catalog_content'].str.extract(r'Item Name:\s*([^\n]+)', flags=re.IGNORECASE)


for i in range(1, 6):
    pattern = rf'Bullet Point {i}:\s*([^\n]+)'
    df[f'bullet_point_{i}'] = df['catalog_content'].str.extract(pattern, flags=re.IGNORECASE)


df['product_description'] = df['catalog_content'].str.extract(r'Product Description:\s*([^\n]+)', flags=re.IGNORECASE)


df['quantity_value'] = (
    df['catalog_content']
    .str.extract(r'Value:\s*([\d\.]+)', flags=re.IGNORECASE)[0]
    .astype(float)
)
df['quantity_unit'] = (
    df['catalog_content']
    .str.extract(r'Unit:\s*([A-Za-z ]+)', flags=re.IGNORECASE)[0]
    .str.strip()
    .str.lower()
)

# --- STEP 6: Normalize units ---
unit_map = {
    "ounce": "oz", "ounces": "oz", "fl oz": "oz",
    "pound": "lb", "pounds": "lb",
    "count": "count", "ct": "count",
    "gram": "g", "grams": "g",
    "kilogram": "kg", "kilograms": "kg",
    "milliliter": "ml", "milliliters": "ml",
    "liter": "l", "liters": "l",
    "pack": "pack", "piece": "count"
}

def normalize_unit(u):
    if pd.isna(u):
        return np.nan
    u = u.lower().strip()
    for key, val in unit_map.items():
        if key in u:
            return val
    return u

df['normalized_unit'] = df['quantity_unit'].apply(normalize_unit)

# --- STEP 7: Create a “quantity_value_converted” column ---
conversion = {
    "g": 1,
    "kg": 1000,
    "oz": 28.3495,
    "lb": 453.592,
    "ml": 1,
    "l": 1000,
    "count": 1,
    "pack": 1
}

def convert_qty(row):
    val, unit = row['quantity_value'], row['normalized_unit']
    if pd.notna(val) and unit in conversion:
        return val * conversion[unit]
    return np.nan

df['quantity_value_converted'] = df.apply(convert_qty, axis=1)

# --- STEP 8: Clean item name text ---
df['item_name'] = (
    df['item_name']
    .str.replace(r'^Item Name:\s*', '', regex=True)
    .str.strip()
)


df['brand'] = df['item_name'].str.extract(r'^([A-Za-z0-9\'&\-\s]+)', expand=False).str.split().str[0]


columns = [
    'item_name', 'brand',
    'bullet_point_1', 'bullet_point_2', 'bullet_point_3',
    'bullet_point_4', 'bullet_point_5',
    'product_description',
    'quantity_value', 'normalized_unit', 'quantity_value_converted'
]


if 'price' in df.columns:
    columns.insert(1, 'price')

df_clean = df[columns]

df_clean.to_csv("structured_catalog_dataset.csv", index=False)

print("✅ Clean structured dataset created successfully!")
print(df_clean.head(10))
