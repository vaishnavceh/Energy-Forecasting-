"""
compress_to_parquet.py
======================
Convert large preprocessed CSV files to Apache Parquet format.

Parquet advantages over CSV:
  - 7-10x smaller file size (columnar + compression)
  - 10x faster read speed (binary format)
  - Supports column-level loading (read only needed columns)

Uses chunked reading to handle files larger than available RAM.

Usage:
    python scripts/compress_to_parquet.py --input path/to/file.csv
    python scripts/compress_to_parquet.py --input path/to/file.csv --compression snappy

Output:
    Same directory as input, with .parquet extension
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import time
import argparse


# Memory-efficient data types for smart meter data
SMARTMETER_DTYPES = {
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


def csv_to_parquet(input_file, output_file=None, chunk_size=1_000_000,
                   compression='snappy', dtypes=None, parse_dates=None):
    """
    Convert a CSV file to Parquet format using chunked reading.

    Parameters
    ----------
    input_file : str
        Path to input CSV file.
    output_file : str, optional
        Path to output Parquet file. If None, uses same name with .parquet extension.
    chunk_size : int
        Number of rows per chunk (default: 1,000,000).
    compression : str
        Compression algorithm: 'snappy' (fast), 'gzip' (smaller), 'zstd' (balanced).
    dtypes : dict, optional
        Column data types for memory efficiency.
    parse_dates : list, optional
        Columns to parse as datetime.

    Returns
    -------
    dict
        Conversion summary statistics.
    """
    if output_file is None:
        output_file = input_file.rsplit('.', 1)[0] + '.parquet'

    print("=" * 60)
    print("CSV to Parquet Conversion")
    print(f"Input:       {input_file}")
    print(f"Output:      {output_file}")
    print(f"Compression: {compression}")
    print(f"Chunk size:  {chunk_size:,} rows")
    print("=" * 60)

    start_time = time.time()
    writer = None
    total_rows = 0

    # Build read_csv kwargs
    read_kwargs = {'chunksize': chunk_size}
    if dtypes:
        read_kwargs['dtype'] = dtypes
    if parse_dates:
        read_kwargs['parse_dates'] = parse_dates

    # Read CSV in chunks and write to Parquet
    reader = pd.read_csv(input_file, **read_kwargs)

    for i, chunk in enumerate(reader):
        total_rows += len(chunk)

        # Convert to PyArrow table
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        # Create writer on first chunk
        if writer is None:
            writer = pq.ParquetWriter(
                output_file,
                table.schema,
                compression=compression,
            )

        writer.write_table(table)

        elapsed = time.time() - start_time
        print(f"  Chunk {i+1}: {total_rows:>12,} rows written  |  Time: {elapsed:.0f}s")

    # Close writer
    if writer:
        writer.close()

    total_time = time.time() - start_time

    # Compare file sizes
    input_size = os.path.getsize(input_file)
    output_size = os.path.getsize(output_file)
    ratio = input_size / output_size if output_size > 0 else 0

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print(f"  Total rows:      {total_rows:,}")
    print(f"  CSV size:        {input_size / 1e9:.2f} GB")
    print(f"  Parquet size:    {output_size / 1e9:.2f} GB")
    print(f"  Compression:     {ratio:.1f}x smaller")
    print(f"  Space saved:     {(input_size - output_size) / 1e9:.2f} GB")
    print(f"  Time taken:      {total_time / 60:.1f} minutes")
    print("=" * 60)

    print(f"\nTo load later:")
    print(f"  df = pd.read_parquet(r'{output_file}')")
    print(f"\nTo load specific columns:")
    print(f"  df = pd.read_parquet(r'{output_file}', columns=['CUSTOMER_ID', 'TOTAL_KWH'])")

    return {
        'total_rows': total_rows,
        'csv_size_gb': round(input_size / 1e9, 2),
        'parquet_size_gb': round(output_size / 1e9, 2),
        'compression_ratio': round(ratio, 1),
        'time_minutes': round(total_time / 60, 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert CSV to Parquet format with compression'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output Parquet file (default: same name, .parquet)')
    parser.add_argument('--chunk-size', type=int, default=1_000_000,
                        help='Rows per chunk (default: 1,000,000)')
    parser.add_argument('--compression', type=str, default='snappy',
                        choices=['snappy', 'gzip', 'zstd', 'none'],
                        help='Compression algorithm (default: snappy)')
    parser.add_argument('--smartmeter', action='store_true',
                        help='Use smart meter dtypes for memory efficiency')
    args = parser.parse_args()

    dtypes = SMARTMETER_DTYPES if args.smartmeter else None
    parse_dates = ['READING_DATETIME'] if args.smartmeter else None

    csv_to_parquet(
        input_file=args.input,
        output_file=args.output,
        chunk_size=args.chunk_size,
        compression=args.compression,
        dtypes=dtypes,
        parse_dates=parse_dates,
    )


if __name__ == '__main__':
    main()
