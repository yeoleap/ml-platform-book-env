"""MovieLens preparation with the schema fix, prefix parameter and full QA."""
import sys
from pathlib import Path

from kfp import compiler, dsl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upstream" / "movie-recommender-project"))
from data_components import csv_to_parquet  # noqa: E402

from movielens_components import (  # noqa: E402
    download_movielens, put_to_s3, qa_movielens, split_ratings, unzip_movielens)


@dsl.pipeline(name="ch07-movielens-prep")
def pipeline(minio_bucket: str = "datasets", dataset: str = "ml-latest-small", random_init: int = 42,
             min_total_rows: int = 100000):
    download = download_movielens(dataset=dataset)
    unzip = unzip_movielens(input_path=download.outputs["output_path_one"], dataset=dataset)
    ratings = csv_to_parquet(inputFile=unzip.outputs["ratings_output_path"])
    movies = csv_to_parquet(inputFile=unzip.outputs["movies_output_path"])
    split = split_ratings(input_parquet=ratings.output, random_state=random_init)
    u1 = put_to_s3(inputFile=movies.output, upload_file_name="movies.parquet.gzip",
                   bucket=minio_bucket, prefix=dataset)
    u2 = put_to_s3(inputFile=split.output, bucket=minio_bucket, prefix=dataset)
    qa = qa_movielens(bucket=minio_bucket, dataset=dataset, min_total_rows=min_total_rows).after(u1, u2)
    for task in [download, unzip, ratings, movies, split, u1, u2, qa]:
        task.set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "ch07_movielens_prep.yaml")
