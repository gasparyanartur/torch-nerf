import gdown
import argparse
import pathlib as pl
import py7zr
import logging


def setup_data(data_url: str, dst_path: pl.Path) -> None:
    if not dst_path.parent.exists() and dst_path.is_dir():
        raise argparse.ArgumentError(
            "Could not store dataset in path {dst_path} - No parent directory found"
        )

    if dst_path.exists():
        raise argparse.ArgumentError(
            "Could not store dataset in path {dst_path} - Path already exists"
        )

    logging.info(f"Downloading file from url {data_url} to path {dst_path}...")
    downloaded = gdown.download(data_url, output=dst_path)
    if not downloaded:
        raise ValueError(f"Failed to get data from {data_url} - Download failed")

    logging.info(f"Finished downloading file. Removing 7z file...")
    with py7zr.SevenZipFile(data_url, mode="r") as zf:
        zf.extract(dst_path.stem)

    logging.info(f"Finished extracting file. Removing .7z file...")
    dst_path.unlink()

    logging.info("Finished setting up data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DataSetup",
        description="Sets up data from the data repository attached to the NeRF paper.",
    )

    parser.add_argument(
        "--url",
        default="https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1",
        required=False,
    )
    parser.add_argument(
        "--dstpath", default="./data/nerf-data.7z", type=pl.Path, required=False
    )

    args = parser.parse_args()
    setup_data(data_url=args.url, dst_path=args.dstpath)
