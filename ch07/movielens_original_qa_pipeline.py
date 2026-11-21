"""Author's qa_data unchanged, pointed at the small dataset uploaded by ch07-movielens-prep."""
import sys
from pathlib import Path

from kfp import compiler, dsl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upstream" / "movie-recommender-project"))
from data_components import qa_data  # noqa: E402


@dsl.pipeline(name="ch07-movielens-original-qa")
def pipeline(minio_bucket: str = "datasets", dataset: str = "ml-latest-small"):
    qa_data(bucket=minio_bucket, dataset=dataset).set_caching_options(False)


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "ch07_movielens_original_qa.yaml")
