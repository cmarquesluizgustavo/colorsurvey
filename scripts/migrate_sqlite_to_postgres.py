#!/usr/bin/env python3
"""
Migrate data from SQLite dumps to PostgreSQL.
Uses Python's built-in sqlite3 module.
"""
import sqlite3
import psycopg2
import os
import sys


def load_sqlite_dump(db_path, dump_file):
    """Load SQLite dump file into SQLite database."""
    print(f"\n📄 Loading {dump_file} into SQLite...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    with open(dump_file, 'r', encoding='utf-8') as f:
        sql_script = f.read()
    
    cursor.executescript(sql_script)
    conn.commit()
    
    # Get table info
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"   ✓ Tables loaded: {', '.join([t[0] for t in tables])}")
    
    conn.close()
    return True


def migrate_table(sqlite_conn, pg_conn, table_name, pg_table_name=None):
    """Migrate a table from SQLite to PostgreSQL."""
    if pg_table_name is None:
        pg_table_name = table_name
    
    print(f"\n📊 Migrating table: {table_name} -> {pg_table_name}")
    
    sqlite_cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()
    
    # Get table schema from SQLite
    sqlite_cursor.execute(f"PRAGMA table_info({table_name})")
    columns_info = sqlite_cursor.fetchall()
    
    # Map SQLite types to PostgreSQL types
    type_mapping = {
        'INTEGER': 'INTEGER',
        'TEXT': 'TEXT',
        'REAL': 'REAL',
        'BLOB': 'BYTEA',
        'NUMERIC': 'NUMERIC'
    }
    
    # Create table in PostgreSQL
    columns = []
    for col in columns_info:
        col_name = col[1]
        col_type = col[2] if col[2] else 'TEXT'
        pg_type = type_mapping.get(col_type, 'TEXT')
        
        if col[5]:  # is primary key
            columns.append(f"{col_name} {pg_type} PRIMARY KEY")
        else:
            columns.append(f"{col_name} {pg_type}")
    
    create_sql = f"CREATE TABLE IF NOT EXISTS {pg_table_name} ({', '.join(columns)})"
    print(f"   Creating table...")
    pg_cursor.execute(create_sql)
    pg_conn.commit()
    
    # Get row count
    sqlite_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    total_rows = sqlite_cursor.fetchone()[0]
    print(f"   Found {total_rows} rows to migrate")
    
    # Migrate data in batches
    batch_size = 5000
    offset = 0
    
    while offset < total_rows:
        sqlite_cursor.execute(f"SELECT * FROM {table_name} LIMIT {batch_size} OFFSET {offset}")
        rows = sqlite_cursor.fetchall()
        
        if not rows:
            break
        
        # Get column names
        col_names = [description[0] for description in sqlite_cursor.description]
        placeholders = ','.join(['%s'] * len(col_names))
        insert_sql = f"INSERT INTO {pg_table_name} ({','.join(col_names)}) VALUES ({placeholders})"
        
        # Insert batch
        try:
            pg_cursor.executemany(insert_sql, rows)
            pg_conn.commit()
            offset += len(rows)
            print(f"   Migrated {offset}/{total_rows} rows...", end='\r')
        except Exception as e:
            print(f"\n   ✗ Error inserting batch at offset {offset}: {e}")
            pg_conn.rollback()
            break
    
    print(f"\n   ✓ Migrated {offset} rows")
    return offset


def main():
    """Main migration function."""
    print("=" * 60)
    print("SQLite to PostgreSQL Migration")
    print("=" * 60)
    
    # Create SQLite databases from dumps
    dumps = [
        ('data/databases/mainsurvey.db', 'data/dumps/mainsurvey_sqldump.txt', 'mainsurvey_'),
        ('data/databases/satfaces.db', 'data/dumps/satfaces_sqldump.txt', 'satfaces_')
    ]
    
    # Connect to PostgreSQL
    print("\n🔌 Connecting to PostgreSQL...")
    pg_conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432')),
        dbname=os.getenv('DB_NAME', 'colorsurvey'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'postgres')
    )
    print("   ✓ Connected to PostgreSQL")
    
    # Drop existing tables
    print("\n🗑️  Dropping existing tables...")
    pg_cursor = pg_conn.cursor()
    for prefix in ['mainsurvey_', 'satfaces_', '']:
        for table in ['users', 'answers', 'names']:
            try:
                pg_cursor.execute(f"DROP TABLE IF EXISTS {prefix}{table} CASCADE")
            except:
                pass
    pg_conn.commit()
    print("   ✓ Tables dropped")
    
    for db_file, dump_file, table_prefix in dumps:
        # Remove existing SQLite DB if it exists
        if os.path.exists(db_file):
            os.remove(db_file)
        
        # Load dump into SQLite
        load_sqlite_dump(db_file, dump_file)
        
        # Connect to SQLite
        sqlite_conn = sqlite3.connect(db_file)
        sqlite_cursor = sqlite_conn.cursor()
        
        # Get all tables
        sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in sqlite_cursor.fetchall()]
        
        # Migrate each table with prefix
        for table in tables:
            pg_table_name = table_prefix + table
            migrate_table(sqlite_conn, pg_conn, table, pg_table_name)
        
        sqlite_conn.close()
    
    pg_conn.close()
    
    print("\n" + "=" * 60)
    print("Migration Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
