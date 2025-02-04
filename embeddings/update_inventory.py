import os
import sys
import mysql.connector
import pandas as pd
from dotenv import load_dotenv
import numpy as np

# Load environment variables from the .env file
load_dotenv()

# Function to connect to the MySQL database
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "embeddings_db_resnet_2"),
            port=int(os.getenv("DB_PORT", 3306))
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        sys.exit(1)

# Function to update the inventory count in the database
def update_inventory(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Ensure all SKUs are uppercase and handle NaN or empty SKUs
    df['SKUs'] = df['SKUs'].str.upper()
    df = df.dropna(subset=['SKUs'])  # Remove rows with NaN SKU values

    # Replace missing or NaN inventory values with 0
    df['Inventory (Roll)'] = df['Inventory (Roll)'].fillna(0)

    # Connect to the database
    conn = get_db_connection()
    cursor = conn.cursor()

    updated_count = 0
    skipped_count = 0

    for index, row in df.iterrows():
        sku = row['SKUs']
        inventory = row['Inventory (Roll)']

        # Skip if SKU is invalid or empty
        if pd.isna(sku) or not sku.strip():
            print(f"Invalid SKU at row {index}, skipping.")
            skipped_count += 1
            continue

        # Ensure SKU is not a NaN or empty string
        if sku == 'nan' or not sku.strip():
            print(f"SKU at row {index} is invalid (nan or empty), skipping.")
            skipped_count += 1
            continue

        # Check if the SKU matches any filename in the database
        cursor.execute("SELECT id FROM embeddings WHERE filename = %s", (sku,))
        result = cursor.fetchone()

        if result:
            # If SKU matches, update the inventory count
            cursor.execute("""
                UPDATE embeddings
                SET inventory_count = %s
                WHERE filename = %s
            """, (inventory, sku))
            updated_count += 1
            print(f"Updated inventory count for SKU {sku}.")
        else:
            skipped_count += 1
            print(f"SKU {sku} not found in the database. Skipping.")

    # Commit the changes and close the connection
    conn.commit()
    print(f"Database update complete. {updated_count} records updated, {skipped_count} skipped.")
    cursor.close()
    conn.close()

# Run the script
if __name__ == "__main__":
    csv_file_path = os.path.join(os.getcwd(), "data", "Ai Covers - Covers.csv")
    update_inventory(csv_file_path)
