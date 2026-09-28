"""Approach 5: fine-tune AfriBERTa-small, a transformer pretrained on 11 African languages including Swahili.

Self-attention lets every subword attend to every other subword in the window, and
pretraining brings knowledge from text the other models never see.

Usage: python train_transformer.py            (full plan)
       python train_transformer.py --quick    (two short runs to check the setup)
"""

import argparse
import hashlib
import json
import time

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import NUM_CLASSES, RESULTS_DIR, get_device, set_seed
from data import load_splits
from metrics import (
    compute_metrics,
    figure_path,
    log_experiment,
    plot_confusion_matrix,
    plot_learning_curves,
    save_metrics,
    save_predictions,
)
from preprocess import clean_splits
from sequence_data import LengthBucketSampler

MODEL_NAME = "castorini/afriberta_small"
OWNER = "Gentil"
# Two seeds, not three: each 512-token run takes about 1.5 hours on a laptop GPU.
FINAL_SEEDS = [42, 7]

START = {"max_tokens": 256, "truncation": "head", "class_weights": False, "learning_rate": 5e-5,
         "epochs": 3, "batch_size": 16, "warmup_share": 0.1}
FIRST = ("AfriBERTa-small, first 256 subword tokens, lr 5e-5, 3 epochs",
         "Pretrained contextual representations should beat models trained from scratch on 20k articles.")
STEPS = [
    ({"max_tokens": 512}, "First 512 tokens instead of 256",
     "78% of articles exceed 256 tokens; the model's full 512-token window should add evidence."),
    ({"truncation": "head_tail"}, "Keep the first 128 and last 382 tokens",
     "Sun et al. (2019) found head+tail truncation best for long documents."),
]


def encode(texts, tokenizer, max_tokens, truncation):
    """Subword ids with <s> and </s>, cut to max_tokens using head or head+tail truncation."""
    budget = max_tokens - 2
    encoded = tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    sequences = []
    for ids in encoded:
        if len(ids) > budget:
            ids = ids[:budget] if truncation == "head" else ids[:128] + ids[-(budget - 128):]
        sequences.append([tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id])
    return sequences


def make_loader(sequences, labels, pad_id, batch_size, shuffle, max_tokens):
    def collate(batch):
        # Pad every batch to the full window: with changing shapes the Apple GPU cache grew until it swapped.
        ids = torch.full((len(batch), max_tokens), pad_id, dtype=torch.long)
        for row, (sequence, _) in enumerate(batch):
            ids[row, :len(sequence)] = torch.tensor(sequence)
        return ids, (ids != pad_id).long(), torch.tensor([label for _, label in batch])

    sampler = LengthBucketSampler([len(s) for s in sequences], batch_size, shuffle)
    return DataLoader(list(zip(sequences, labels)), batch_sampler=sampler, collate_fn=collate)


def predict(model, loader, device):
    model.eval()
    probabilities, total_loss, loss_function = [], 0.0, nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for ids, mask, labels in loader:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
            total_loss += loss_function(logits, labels.to(device)).item()
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    in_length_order = np.concatenate(probabilities)
    original_order = np.empty_like(in_length_order)
    original_order[loader.batch_sampler.order] = in_length_order
    return original_order, total_loss / len(original_order)


def train_one(config, splits, tokenizer, seed, device, sample=None):
    set_seed(seed)
    train, validation, test = splits
    if sample:
        train = train.sample(sample, random_state=seed)
    pad_id = tokenizer.pad_token_id
    loaders = []
    for frame, shuffle, batch_size in [(train, True, config["batch_size"]), (validation, False, 32), (test, False, 32)]:
        sequences = encode(frame["clean_text"], tokenizer, config["max_tokens"], config["truncation"])
        loaders.append(make_loader(sequences, frame["label"].tolist(), pad_id, batch_size, shuffle, config["max_tokens"]))
    train_loader, val_loader, test_loader = loaders

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(device)
    weights = None
    if config["class_weights"]:
        weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train["label"])
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config["warmup_share"] * total_steps), total_steps)

    history, best_f1, best_state = [], -1, None
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for step, (ids, mask, labels) in enumerate(train_loader, start=1):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = loss_function(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * len(labels)
            seen += len(labels)
            if step % 200 == 0:
                if device.type == "mps":
                    torch.mps.empty_cache()
                print(f"    step {step}/{len(train_loader)} loss={running_loss / seen:.4f} ({time.time() - start:.0f}s)")

        val_probabilities, val_loss = predict(model, val_loader, device)
        if device.type == "mps":
            torch.mps.empty_cache()
        val_f1 = compute_metrics(validation["label"], val_probabilities)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": running_loss / seen, "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"  epoch {epoch}: train_loss={running_loss / seen:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_probabilities, _ = predict(model, val_loader, device)
    test_probabilities, _ = predict(model, test_loader, device)
    result = {
        "history": history,
        "val_probabilities": val_probabilities,
        "test_probabilities": test_probabilities,
        "val_metrics": compute_metrics(validation["label"], val_probabilities),
        "seconds": round(time.time() - start, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def cached_train(config, splits, tokenizer, seed, device):
    """Train once per (config, seed) and keep the outputs on disk, so a crashed plan can resume."""
    key = json.dumps({**config, "seed": seed}, sort_keys=True)
    run_dir = RESULTS_DIR / "transformer_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"run_{hashlib.md5(key.encode()).hexdigest()[:10]}.npz"
    if path.exists():
        saved = np.load(path, allow_pickle=True)
        print(f"  reusing saved run {path.name}")
        return saved["result"].item()
    result = train_one(config, splits, tokenizer, seed, device)
    np.savez(path, result=np.array(result, dtype=object))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1 epoch on 2,000 articles to test the setup")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = 512
    train, validation, test = load_splits()
    # Light cleaning only: the subword tokenizer was trained on cased text with punctuation.
    splits = clean_splits(train, validation, test, mode="light")[:3]

    if args.quick:
        result = train_one({**START, "epochs": 1}, splits, tokenizer, 42, device, sample=2000)
        print(result["val_metrics"], result["seconds"])
        return

    history_dir = RESULTS_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    step_results = {}

    def run_step(experiment_id, config, change, hypothesis):
        print(f"[{experiment_id}] {change}")
        result = cached_train(config, splits, tokenizer, 42, device)
        step_results[json.dumps(config, sort_keys=True)] = result
        log_experiment(experiment_id, "AfriBERTa", OWNER, change, hypothesis, result["val_metrics"],
                       extra={"config": config, "seconds": result["seconds"], "parameters": result["parameters"]})
        with open(history_dir / f"{experiment_id}.json", "w") as f:
            json.dump(result["history"], f)
        return result["val_metrics"]["macro_f1"]

    best_config = dict(START)
    best_f1 = run_step("T01", best_config, *FIRST)
    for number, (change_values, change, hypothesis) in enumerate(STEPS, start=2):
        candidate = {**best_config, **change_values}
        score = run_step(f"T{number:02d}", candidate, change, hypothesis)
        if score > best_f1:
            best_config, best_f1 = candidate, score

    final_id = f"T{len(STEPS) + 2:02d}"
    _, validation, test = splits
    seed_scores, main_run = [], None
    for seed in FINAL_SEEDS:
        print(f"[{final_id}] final configuration, seed {seed}")
        if seed == 42:
            result = step_results[json.dumps(best_config, sort_keys=True)]
        else:
            result = cached_train(best_config, splits, tokenizer, seed, device)
        test_metrics = compute_metrics(test["label"], result["test_probabilities"])
        seed_scores.append({"seed": seed, "val_macro_f1": result["val_metrics"]["macro_f1"],
                            "test_macro_f1": test_metrics["macro_f1"], "test_accuracy": test_metrics["accuracy"]})
        if main_run is None:
            main_run = (result, test_metrics)

    result, test_metrics = main_run
    scores = [s["test_macro_f1"] for s in seed_scores]
    summary = {"config": best_config, "seeds": seed_scores, "test_macro_f1_mean": round(float(np.mean(scores)), 4),
               "test_macro_f1_std": round(float(np.std(scores)), 4), "seconds": result["seconds"],
               "parameters": result["parameters"]}
    log_experiment(final_id, "AfriBERTa", OWNER,
                   f"Final configuration, {len(FINAL_SEEDS)} seeds (test macro-F1 {summary['test_macro_f1_mean']} "
                   f"+/- {summary['test_macro_f1_std']})",
                   "Selected configuration, evaluated once on test.", result["val_metrics"], test_metrics, extra=summary)
    save_predictions("afriberta", "validation", validation, result["val_probabilities"])
    save_predictions("afriberta", "test", test, result["test_probabilities"])
    save_metrics("afriberta", {"validation": result["val_metrics"], "test": test_metrics, "summary": summary})
    with open(history_dir / f"{final_id}.json", "w") as f:
        json.dump(result["history"], f)
    plot_learning_curves(result["history"], "AfriBERTa learning curves", figure_path("learning_curves_afriberta.png"))
    plot_confusion_matrix(test["label"], result["test_probabilities"].argmax(axis=1), "AfriBERTa (test)",
                          figure_path("confusion_afriberta.png"))


if __name__ == "__main__":
    main()
