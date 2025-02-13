import os
import mysql.connector
import gspread
from dotenv import load_dotenv
from oauth2client.service_account import ServiceAccountCredentials
import re  

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = "resnet_3_db"  
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")

# Google credentials path
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")

# Validate Google credentials file existence
if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
    raise FileNotFoundError(f"❌ Credentials file not found at: {GOOGLE_CREDENTIALS_PATH}")

# Authenticate with Google Sheets API
def authenticate_google_sheets():
    """Authenticate with Google Sheets API using credentials file."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_CREDENTIALS_PATH, scope)
    client = gspread.authorize(creds)
    return client

# Function to clean and format column names 
def clean_column_name(column):
    """Sanitize column names to match MySQL field names correctly."""
    column = column.strip()  
    column = re.sub(r"[^\w\s/]", "", column)  # remove special characters
    column = re.sub(r"\s+", "_", column)  # replace spaces with underscores
    column = column.replace("/", "_")  # replace slashes with underscores
    return column

# Get data from the Google Sheet
def get_product_data():
    """Fetch product data from the Google Sheet."""
    client = authenticate_google_sheets()
    sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/17E0zrKB4OFNz6B2s_TgC7tmnIYC-vFALGOGHuMglTP8/edit?usp=sharing")
    worksheet = sheet.get_worksheet(0)  

    # Get all rows of data, including headers dynamically
    raw_headers = worksheet.row_values(1)  # Fetch column names from first row
    headers = [clean_column_name(header) for header in raw_headers]  # format headers

    # Remove unwanted or auto-generated columns
    headers = [col for col in headers if col.lower() not in ["_row_id", "row_id", "id"]]

    # Ensure the "Embeddings" column is present
    if "Embeddings" not in headers:
        headers.append("Embeddings")  # adds if missing

    rows = worksheet.get_all_records(head=1)

    # Filter out SKUs starting with "TOOLS"
    filtered_rows = [row for row in rows if not row.get("SKU", "").startswith("TOOLS")]

    ignored_count = len(rows) - len(filtered_rows)
    print(f"🚫 Ignored {ignored_count} rows with SKUs starting with 'TOOLS'.")

    return headers, filtered_rows  

# Ensure MySQL table exists 
def ensure_table_exists(headers):
    """Creates the MySQL table dynamically based on Google Sheets headers."""
    connection = mysql.connector.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        database=DB_NAME,
        port=DB_PORT
    )
    cursor = connection.cursor()

    # Generate a table creation query dynamically
    columns_sql = ", ".join([f"`{col}` TEXT" if col != "Embeddings" else "`Embeddings` JSON NOT NULL DEFAULT ('{}')" for col in headers])
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS embeddings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        {columns_sql}
    );
    """

    cursor.execute(create_table_sql)
    connection.commit()
    cursor.close()
    connection.close()
    print("✅ Table checked/created successfully.")

# Insert or update product data in MySQL
def update_database(headers, products):
    """Insert or update product data in MySQL database dynamically."""
    connection = mysql.connector.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        database=DB_NAME,
        port=DB_PORT
    )

    cursor = connection.cursor()

    # Track how many rows were processed for logging
    rows_updated = 0  

    # Loop through each row of product data
    for index, product in enumerate(products):
        # Convert keys to match MySQL column names
        product_data = {clean_column_name(key): value for key, value in product.items()}
        
        # Remove unwanted columns like "_Row_ID" before inserting into MySQL
        product_data = {k: v for k, v in product_data.items() if k.lower() not in ["_row_id", "row_id", "id"]}

        # Ensure "Embeddings" has a default JSON value `{}` if missing
        if "Embeddings" not in product_data:
            product_data["Embeddings"] = "{}"  # default empty JSON

        # Generate column and value placeholders dynamically
        columns = ", ".join([f"`{col}`" for col in product_data.keys()])
        placeholders = ", ".join(["%s"] * len(product_data))
        update_stmt = ", ".join([f"`{col}` = VALUES(`{col}`)" for col in product_data.keys()])

        # Insert or update statement
        sql = f"""
        INSERT INTO embeddings ({columns})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_stmt};
        """

        cursor.execute(sql, list(product_data.values()))
        rows_updated += 1  
        print(f"✅ Processed row {index+1} / {len(products)}")

    # Commit changes and close the connection
    connection.commit()
    cursor.close()
    connection.close()

    print(f"✅ Database updated successfully with {rows_updated} rows updated.")

# fetch and update data
def main():
    print("🔄 Fetching data from Google Sheets...")
    headers, products = get_product_data()  
    ensure_table_exists(headers)  
    update_database(headers, products)  
    print("✅ Update process completed.")

if __name__ == "__main__":
    main()
