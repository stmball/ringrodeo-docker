"""Files for splitting models into more digestible formats - can remove later"""

from filesplit.split import Split
from filesplit.merge import Merge

from pathlib import Path


def merge_models():
    split_models_path = Path("split_models/")
    models_path = Path("models/")
    models_path.mkdir(exist_ok=True)

    for folder in list(split_models_path.glob("*")):
        merger = Merge(
            inputdir=str(folder), outputdir=str(models_path), outputfilename=folder.name
        )

        merger.merge()


def split_models():
    split_models_path = Path("./split_models")
    models_path = Path("./models")

    for model_file in list(models_path.rglob("*.pth")):
        output_dir = split_models_path / model_file.name

        output_dir.mkdir(parents=True, exist_ok=True)

        splitter = Split(inputfile=str(model_file), outputdir=str(output_dir))

        # 50Mb splits
        splitter.bysize(size=1024 * 1024 * 50)


if __name__ == "__main__":
    split_models()
