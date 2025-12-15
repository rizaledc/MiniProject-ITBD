import pandas as pd
import re
import os

print("=== TAHAP 1: DATA CLEANING & PREPROCESSING ===")

# 1. Definisi Fungsi Parsing Regex
def parse_sql_to_csv(sql_file, csv_file):
    print(f"[PROCESS] Sedang memproses {sql_file}...")
    
    # Pola Regex untuk menangkap: ('customer_id', 'registered_date')
    pattern = r"\('([^']*)',\s*'([^']*)'\)"
    
    extracted_data = []
    
    try:
        with open(sql_file, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(pattern, content)
            
            for match in matches:
                extracted_data.append({
                    'customer_id': match[0],
                    'registered_date': match[1]
                })
        
        # Konversi ke DataFrame dan Simpan CSV
        if extracted_data:
            df = pd.DataFrame(extracted_data)
            df.to_csv(csv_file, index=False)
            print(f"[SUCCESS] Berhasil mengekstrak {len(df)} data pelanggan.")
            print(f"[OUTPUT] File tersimpan di: {csv_file}")
        else:
            print("[WARNING] Tidak ada data yang cocok dengan pola Regex.")
            
    except FileNotFoundError:
        print(f"[ERROR] File {sql_file} tidak ditemukan.")

# 2. Eksekusi
if __name__ == "__main__":
    # Cek file dataset lain
    files = ['order_detail.csv', 'sku_detail.csv', 'payment_detail.csv']
    for f in files:
        if os.path.exists(f):
            print(f"[CHECK] File {f} ditemukan.")
        else:
            print(f"[ERROR] File {f} TIDAK ditemukan!")

    # Jalankan Parsing Customer
    parse_sql_to_csv('customer_detail.csv', 'customer_detail_clean.csv')