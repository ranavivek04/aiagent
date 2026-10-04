import sqlite3
import os

base_dir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(base_dir, ".litellm_cache", "cache.db")

def inspect_local_cache():
    print(f"🕵️‍♂️ Accessing custom SQLite cache layout from disk: '{db_path}'...")
    
    if not os.path.exists(db_path):
        print("❌ Systems Exception: File missing. Start the LiteLLM Proxy with config.yaml first.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # =====================================================================
        # PART A: READ CACHE STATISTICS (Hits vs Misses)
        # =====================================================================
        print("\n📈 [CACHE SUB-SYSTEM METRICS]:")
        cursor.execute("SELECT key, value FROM Settings WHERE key IN ('hits', 'misses', 'count');")
        stats = cursor.fetchall()
        
        for stat_key, stat_value in stats:
            print(f"   ↳ {stat_key.upper()}: {stat_value}")

        # =====================================================================
        # PART B: PARSE BINARY BLOB KEYS NATIVELY
        # =====================================================================
        print("\n⏳ Extracting active data cache row mappings...")
        
        # We use SQLite's HEX() function to cast the binary BLOB keys into readable strings
        cursor.execute("SELECT rowid, HEX(key), store_time, access_count FROM Cache LIMIT 5;")
        records = cursor.fetchall()
        
        if not records:
            print("ℹ️ No active prompt query keys are written to the database yet.")
            print("   Execute your multi-agent script or send a payload request to populate them!")
            conn.close()
            return
            
        print(f"🎯 Total cached record hashes found: {len(records)}")
        for rowid, hex_key, store_time, access_count in records:
            print(f"\n[CACHE SLOT {rowid}]")
            print(f"  🔑 Cryptographic Hex Key: {hex_key[:32]}...")
            print(f"  ⏱️ Ingestion Unix Time:   {store_time}")
            print(f"  🔄 Total Row Hits:        {access_count}")
            
        conn.close()
    except Exception as e:
        print(f"⚠️ Internal Error reading database layer: {e}")

if __name__ == "__main__":
    inspect_local_cache()
