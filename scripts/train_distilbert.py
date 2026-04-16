#!/usr/bin/env python
import json
import argparse
import logging

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import numpy as np

from services.ml.ml_utils import (
    clean_fractured_threat_records,
    compute_metrics,
    load_jsonl_records,
    record_to_window_text,
    resolve_record_label,
    split_records,
)


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    ap = argparse.ArgumentParser(description="Train DistilBERT on CloudTrail threat dataset")
    ap.add_argument("--data", default="data/regenerated/windows_dataset_sample.jsonl", help="Training data file (JSONL)")
    ap.add_argument("--output-dir", default="models/distilbert_finetuned", help="Output directory for model")
    ap.add_argument("--model-name", default="distilbert-base-uncased", help="Base model name from HuggingFace")
    ap.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    ap.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    ap.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for the held-out split")
    ap.add_argument("--split-mode", choices=["time", "stratified"], default="time", help="How to split the dataset before evaluation")
    ap.add_argument(
        "--fractured-threat-policy",
        choices=["off", "drop", "relabel"],
        default="off",
        help="How to handle attack-labeled windows whose event_count is below the minimum threshold.",
    )
    ap.add_argument(
        "--fractured-threat-min-events",
        type=int,
        default=2,
        help="Minimum event_count required before an attack-labeled window is kept as-is.",
    )
    args = ap.parse_args()

    print(f"Loading data from {args.data}...")
    records = load_jsonl_records(args.data)
    records = clean_fractured_threat_records(
        records,
        policy=args.fractured_threat_policy,
        min_event_count=args.fractured_threat_min_events,
    )
    labels = [resolve_record_label(record) for record in records]
    print(f"Loaded {len(records)} samples (label 0: {labels.count(0)}, label 1: {labels.count(1)})")

    # Split data
    split_result = split_records(
        records,
        test_size=args.test_size,
        split_mode=args.split_mode,
        seed=args.seed,
    )
    train_records, test_records = split_result
    texts_train = [record_to_window_text(record)[1] for record in train_records]
    labels_train = [resolve_record_label(record) for record in train_records]
    texts_test = [record_to_window_text(record)[1] for record in test_records]
    labels_test = [resolve_record_label(record) for record in test_records]
    print(
        f"Train: {len(texts_train)}, Test: {len(texts_test)} "
        f"(split={split_result.actual_split_mode}; requested={args.split_mode})"
    )

    # Create HuggingFace datasets
    train_data = Dataset.from_dict({"text": texts_train, "labels": labels_train})
    test_data = Dataset.from_dict({"text": texts_test, "labels": labels_test})

    # Load tokenizer and model
    print(f"\nLoading tokenizer and model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_data.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    test_dataset = test_data.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        predictions = np.argmax(probabilities, axis=-1).tolist()
        return compute_metrics(labels.tolist(), predictions, probabilities[:, 1].tolist())

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=args.learning_rate,
        seed=args.seed,
        report_to=[],
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    eval_results = trainer.evaluate()
    print(f"Test loss: {eval_results.get('eval_loss', 'N/A')}")
    print(
        "Test positive metrics: "
        f"precision={eval_results.get('eval_positive_precision', 'N/A')} "
        f"recall={eval_results.get('eval_positive_recall', 'N/A')} "
        f"f1={eval_results.get('eval_positive_f1', 'N/A')}"
    )

    print(f"\nTraining complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
