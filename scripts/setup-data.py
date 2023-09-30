import gdown
import argparse
import pathlib as pl
import logging
import json
import py7zr
import zipfile


def setup_nerf_llff_data(
    zipped_dataset_path: pl.Path, unzipped_dataset_path: pl.Path
) -> None:
    # TODO Implement setup functions
    raise NotImplementedError


def setup_nerf_synthetic(zipped_dataset_path: pl.Path, unzipped_dataset_path: pl.Path):
    # This dataset has a nested structure that we want to get rid of,
    # so we extract the nested folder directly into the parent dir.

    with zipfile.ZipFile(zipped_dataset_path) as zf:
        for file in zf.namelist():
            if file.startswith("nerf_synthetic/"):
                zf.extract(file, path=unzipped_dataset_path.parent)


dataset_setups = {
    "nerf_synthetic": setup_nerf_synthetic,
    # TODO: Add setup functions here
}


def load_configs(config_path: pl.Path) -> dict[str, any]:
    with open(config_path, "r") as f:
        data = json.load(f)

    return data


def setup_data(configs: dict[str, any]) -> None:
    root_data_path: pl.Path = pl.Path(configs["root_data_path"])
    if not root_data_path.exists():
        root_data_path.mkdir()

    data_id: str = configs["data_id"]
    data_path: pl.Path = root_data_path / configs["data_name"]
    data_path_7z: pl.Path = root_data_path / configs["data_name_7z"]

    if data_path.exists():
        logging.info("Found extracted folder, skipping download and extraction...")

    else:
        if data_path_7z.exists():
            logging.info(f"Found .7z file, skipping download.")

        else:
            logging.info(f"Downloading file with id {data_id} to path {data_path}...")

            downloaded = gdown.download(id=data_id, output=str(data_path))
            if not downloaded:
                raise ValueError(f"Failed to get data - Download failed")

        logging.info(f"Finished downloading file. Extracting 7z file...")
        with py7zr.SevenZipFile(data_path_7z, mode="r") as zf:
            zf.extractall(path=data_path.parent)

            if not data_path.exists():
                raise ValueError(
                    f"Expected extracted data to appear in path {data_path} but could not find it. "
                    f"Update the config file to match the correct extracted data path."
                )

        logging.info(f"Finished extracting file. Removing .7z file...")
        data_path_7z.unlink()

    dataset_names: list[str] = configs["dataset_names"]
    logging.info(f"Setting up datasets: {', '.join(dataset_names)}")
    for dataset_name in dataset_names:
        if not dataset_name in dataset_setups:
            raise ValueError(
                f"Could not find setup function for dataset: {dataset_name}. "
                f"Update config file to contain valid datasets."
            )

        unzipped_dataset_path = data_path / dataset_name
        if unzipped_dataset_path.exists():
            logging.info(
                f"\tFound dataset folder: {unzipped_dataset_path}. Skipping it."
            )
            continue

        zipped_dataset_path = unzipped_dataset_path.with_suffix(".zip")
        if not zipped_dataset_path.exists():
            raise ValueError(
                f"Could not find zipped dataset: {zipped_dataset_path}. "
                f"Update config file to contain valid datasets."
            )

        unzipped_dataset_path.mkdir()
        dataset_setup = dataset_setups[dataset_name]
        dataset_setup(zipped_dataset_path, unzipped_dataset_path)

    logging.info("Finished setting up data.")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        "DataSetup",
        description="Sets up data from the data repository attached to the NeRF paper.",
    )

    parser.add_argument(
        "--config_path", default="./config.json", type=pl.Path, required=False
    )

    args = parser.parse_args()
    configs = load_configs(args.config_path)
    setup_data(configs=configs)
