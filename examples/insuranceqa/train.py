from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from typing import Union
import click
import logging
from functools import partial

logger = logging.getLogger(__name__)

dataset_id_or_path = "/tmp/data/insuranceqa"
dataset = load_from_disk(dataset_id_or_path)
input_model = "databricks/dolly-v2-3b"

def tokenize_function(examples, max_length, tokenizer):

    input_ids = []
    labels = []
    tokenizer = AutoTokenizer.from_pretrained(input_model)

    for text, label in zip(examples["text"], examples["labels"]):

        text = text.replace("  ", " ")
        label = label.replace("  ", " ")

        text = tokenizer(
            text,
            padding = "max_length",
            truncation = True,
            max_length = max_length,
            return_tensors = "pt"
        )

        input_ids.append(text["input_ids"][0])

        label = tokenizer(
            label,
            padding = "max_length",
            truncation = True,
            max_length = max_length,
            return_tensors = "pt"
        )

        labels.append(label["input_ids"][0])
        
    output = {
        "input_ids": input_ids,
        "labels": labels
    }

    return output


def preprocess_dataset(dataset_path: str, max_length, tokenizer):

    dataset = load_from_disk(dataset_path)
    splits_dataset = dataset.train_test_split(test_size = 0.3)
    _preprocessing_function = partial(tokenize_function, max_length = max_length, tokenizer = tokenizer)
    tokenized_datasets = splits_dataset.map(_preprocessing_function, batched = True, num_proc = 10)
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "context"])

    return tokenized_datasets

def train(
    *,
    input_model: str,
    local_output_dir: str,
    dbfs_output_dir: str,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    lr: float,
    seed: int,
    deepspeed: str,
    gradient_checkpointing: bool,
    local_rank: str,
    bf16: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    test_size: Union[float, int],
    save_total_limit: int,
    warmup_steps: int,
    dataset_path: str
):

    # Use the same max length that the model supports.  Fall back to 1024 if the setting can't be found.
    # The configuraton for the length can be stored under different names depending on the model.  Here we attempt
    # a few possible names we've encountered.
    model = AutoModelForCausalLM.from_pretrained(input_model)
    tokenizer = AutoTokenizer.from_pretrained(input_model)
    conf = model.config
    
    max_length = 512
    """for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        logger.info(f"Using default max length: {max_length}")"""

    processed_dataset = preprocess_dataset(dataset_path, max_length, tokenizer)

    logger.info("Train data size: %d", processed_dataset["train"].num_rows)
    logger.info("Test data size: %d", processed_dataset["test"].num_rows)

    if not dbfs_output_dir:
        logger.warn("Will NOT save to DBFS")

    training_args = TrainingArguments(
        output_dir = local_output_dir,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        fp16 = False,
        bf16 = bf16,
        learning_rate = lr,
        num_train_epochs = epochs,
        deepspeed = deepspeed,
        gradient_checkpointing = gradient_checkpointing,
        logging_dir = f"{local_output_dir}/runs",
        logging_strategy = "steps",
        logging_steps = logging_steps,
        evaluation_strategy = "steps",
        eval_steps = eval_steps,
        save_strategy = "steps",
        save_steps = save_steps,
        save_total_limit = save_total_limit,
        load_best_model_at_end = False,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=local_rank,
        warmup_steps=warmup_steps,
    )

    logger.info("Instantiating Trainer")

    trainer = Trainer(
        model = model,
        tokenizer = tokenizer,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    if dbfs_output_dir:
        logger.info(f"Saving Model to {dbfs_output_dir}")
        trainer.save_model(output_dir=dbfs_output_dir)

    logger.info("Done.")


@click.command()
@click.option("--input-model", type=str, help="Input model to fine tune")
@click.option("--local-output-dir", type=str, help="Write directly to this local path", required=True)
@click.option("--dbfs-output-dir", type=str, help="Sync data to this path on DBFS")
@click.option("--epochs", type=int, default=3, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=8, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=8, help="Batch size to use for evaluation.")
@click.option(
    "--test-size", type=int, default=1000, help="Number of test records for evaluation, or ratio of test records."
)
@click.option("--warmup-steps", type=int, default=None, help="Number of steps to warm up to learning rate")
@click.option("--logging-steps", type=int, default=10, help="How often to log")
@click.option("--eval-steps", type=int, default=50, help="How often to run evaluation on test records")
@click.option("--save-steps", type=int, default=400, help="How often to checkpoint the model")
@click.option("--save-total-limit", type=int, default=10, help="Maximum number of checkpoints to keep on disk")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--seed", type=int, default=123, help="Seed to use for training.")
@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bf16 (preferred on A100's).")
@click.option("--dataset-path", type=str, default="/tmp/data/insuranceqa", help="Local dataset path.")

def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise