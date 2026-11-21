"""Author's split_dataset unchanged; only the Box download is replaced by S3."""
import sys
from pathlib import Path

from kfp import compiler, dsl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upstream" / "object-detection-project"))
from data_prep_pipeline import output_file_contents, split_dataset  # noqa: E402

from idcard_components import check_split_pairs, download_dataset_from_s3  # noqa: E402


@dsl.pipeline(name="ch07-idcard-original-split")
def pipeline(random_state: int = 42):
    download_op = download_dataset_from_s3()
    split_op = split_dataset(random_state=random_state, input_dataset=download_op.outputs["output_dataset"])
    output_file_contents(dataset=split_op.outputs["train_dataset"])
    check_split_pairs(
        train_dataset=split_op.outputs["train_dataset"],
        validation_dataset=split_op.outputs["validation_dataset"],
        test_dataset=split_op.outputs["test_dataset"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "ch07_idcard_original.yaml")
