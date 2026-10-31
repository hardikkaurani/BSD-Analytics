"""Script to download and verify raw Bike Sharing dataset."""

import logging

from bsd_analytics.data.ingestion import download_dataset, get_dataset_metadata, load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("Executing download_data.py...")
    path = download_dataset(force=False)
    print(f"Dataset downloaded/verified at: {path}")
    df = load_raw_data()
    meta = get_dataset_metadata(df)
    print(f"Total rows: {meta['total_records']}, Total rentals: {meta['total_rentals']:,}")
    print("Download completed successfully.")


if __name__ == "__main__":
    main()
