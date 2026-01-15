import os
import shutil
import csv

from pathlib import Path


DATA_FOLDER = "data/videos"

def main():
    idx = 0
    mfile = open("data/metadata/name_mapping.csv", "w", newline='', encoding='utf-8')
    cw = csv.writer(mfile)
    cw.writerow(["name", "old_name"])
    for f in os.listdir(DATA_FOLDER):
        if f.endswith(".mp4"):
            old_path = Path(DATA_FOLDER) / f
            new_name = f"{idx:04d}.mp4"
            new_path = Path(DATA_FOLDER) / new_name
            shutil.move(old_path, new_path)
            cw.writerow([new_name, f])
            idx += 1
    mfile.close()


if __name__ == "__main__":
    main()