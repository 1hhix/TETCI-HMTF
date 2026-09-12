import os
import tarfile
from pathlib import Path

import requests


def _safe_extract(archive: tarfile.TarFile, destination: str) -> None:
    root = Path(destination).resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Archive member escapes destination: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Unsupported archive member: {member.name}")
    archive.extractall(path=root)


def download_vrplib():
    sets = ["A", "B", "E", "F", "M", "P", "X"]
    URL = "https://vrp.galgos.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-{}.tgz"

    for s in sets:
        url = URL.format(s)
        print(f"Downloading {url}")

        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "gzip" not in content_type and "octet-stream" not in content_type:
                print(f"Warning: File {url} is not a gzip archive. Skipping extraction.")
                continue

            archive_path = Path(f"Vrp-Set-{s}.tgz")
            try:
                with archive_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)

                with tarfile.open(archive_path, "r:gz") as archive:
                    _safe_extract(archive, "vrplib")
            finally:
                if archive_path.exists():
                    os.remove(archive_path)

if __name__ == "__main__":
    download_vrplib()
