"""Fixed ID-card preparation: stem pairing, clip-level split, QA on all splits."""
from kfp import compiler, dsl

from idcard_components import check_split_pairs, download_dataset_from_s3, split_dataset_by_stem


@dsl.pipeline(name="ch07-idcard-prep-fixed")
def pipeline(random_state: int = 42):
    download_op = download_dataset_from_s3()
    split_op = split_dataset_by_stem(random_state=random_state, input_dataset=download_op.outputs["output_dataset"])
    check_split_pairs(
        train_dataset=split_op.outputs["train_dataset"],
        validation_dataset=split_op.outputs["validation_dataset"],
        test_dataset=split_op.outputs["test_dataset"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "ch07_idcard_fixed.yaml")
