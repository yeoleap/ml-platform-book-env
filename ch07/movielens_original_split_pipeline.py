"""Author's csv_to_parquet and split_dataset unchanged; only the dataset size differs."""
import sys
from pathlib import Path

from kfp import compiler, dsl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upstream" / "movie-recommender-project"))
from data_components import csv_to_parquet, split_dataset  # noqa: E402

from movielens_components import download_movielens, unzip_movielens  # noqa: E402


@dsl.pipeline(name="ch07-movielens-original-split")
def pipeline(dataset: str = "ml-latest-small", random_init: int = 42):
    download = download_movielens(dataset=dataset)
    unzip = unzip_movielens(input_path=download.outputs["output_path_one"], dataset=dataset)
    ratings = csv_to_parquet(inputFile=unzip.outputs["ratings_output_path"])
    split_dataset(input_parquet=ratings.output, random_state=random_init)


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "ch07_movielens_original_split.yaml")
