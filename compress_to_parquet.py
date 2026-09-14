"""
Compress preprocessed CSV (37 GB) -> Parquet (~3-5 GB)
======================================================
Reads in chunks, writes to a single Parquet file with snappy compression.
Parquet is ~7-10x smaller AND ~10x faster to read than CSV.

Run: python compress_to_parquet.py
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import time

# ======================== CONFIG ========================
INPUT_FILE  = r'V:\DATASETS\DATASET 2\Preprocessed\CD_INTERVAL_preprocessed.csv'
OUTPUT_FILE = r'V:\DATASETS\DATASET 2\Preprocessed\CD_INTERVAL_preprocessed.parquet'
CHUNK_SIZE  = 1_000_000  # 1 million rows per chunk
# ========================================================

def main():
    print("=" * 60)
    print("Compressing CSV -> Parquet")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print("=" * 60)

    start_time = time.time()
    writer = None
    total_rows = 0

    # Read CSV in chunks
    reader = pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE,
        parse_dates=['READING_DATETIME'],
        dtype={
            'CUSTOMER_ID': 'int32',
            'GENERAL_SUPPLY_KWH': 'float32',
            'CONTROLLED_LOAD_KWH': 'float32',
            'GROSS_GENERATION_KWH': 'float32',
            'NET_GENERATION_KWH': 'float32',
            'OTHER_KWH': 'float32',
            'TOTAL_KWH': 'float32',
            'hour': 'int8',
            'dayofweek': 'int8',
            'month': 'int8',
            'year': 'int16',
            'is_weekend': 'int8',
            'hour_sin': 'float32',
            'hour_cos': 'float32',
            'month_sin': 'float32',
            'month_cos': 'float32',
        }
    )

    for i, chunk in enumerate(reader):
        total_rows += len(chunk)

        # Convert chunk to PyArrow table
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        # Create writer on first chunk (uses schema from data)
        if writer is None:
            writer = pq.ParquetWriter(
                OUTPUT_FILE,
                table.schema,
                compression='snappy',  # fast compression, good ratio
            )

        # Write chunk
        writer.write_table(table)

        elapsed = time.time() - start_time
        print(f"  Chunk {i+1}: {total_rows:>12,} rows written  |  Time: {elapsed:.0f}s")

    # Close the writer
    if writer:
        writer.close()

    total_time = time.time() - start_time

    # Compare sizes
    input_size = os.path.getsize(INPUT_FILE)
    output_size = os.path.getsize(OUTPUT_FILE)
    ratio = input_size / output_size

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Total rows:     {total_rows:,}")
    print(f"  CSV size:       {input_size / 1e9:.2f} GB")
    print(f"  Parquet size:   {output_size / 1e9:.2f} GB")
    print(f"  Compression:    {ratio:.1f}x smaller")
    print(f"  Space saved:    {(input_size - output_size) / 1e9:.2f} GB")
    print(f"  Time taken:     {total_time / 60:.1f} minutes")
    print("=" * 60)

    print(f"\nYou can now safely delete the large CSV:")
    print(f'  del "{INPUT_FILE}"')
    print(f"\nTo load the Parquet file later:")
    print(f"  df = pd.read_parquet(r'{OUTPUT_FILE}')")


if __name__ == '__main__':
    main()
