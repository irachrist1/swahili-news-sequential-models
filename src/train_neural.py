"""Train the BiLSTM with a step-by-step experiment plan.

Each step changes one thing from the best configuration so far. A change is kept
only if it improves validation macro-F1, so later experiments build on earlier ones.

Usage: python train_neural.py --model bilstm
"""

import argparse
import json
import time

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

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
from neural_models import BiLSTMClassifier
from preprocess import clean_splits
from sequence_data import Vocabulary, load_fasttext_matrix, make_loader

FINAL_SEEDS = [42, 7, 123]

PLANS = {
    "bilstm": {
        "owner": "Rene",
        "name": "BiLSTM + attention",
        "prefix": "R",
        "start": {"pooling": "last", "max_length": 256, "embeddings": "random", "class_weights": False,
                  "hidden_size": 128, "dropout": 0.3, "learning_rate": 1e-3, "epochs": 8, "batch_size": 64},
        "first": ("BiLSTM, random embeddings, final hidden state, first 256 words",
                  "A recurrent model that reads word order should match the linear baselines."),
        "steps": [
            ({"pooling": "attention"}, "Attention pooling instead of the final hidden state",
             "Final states forget the start of long articles; attention can weight the key words wherever they are."),
            ({"max_length": 512}, "Read the first 512 words instead of 256",
             "54% of articles are longer than 256 words, so truncation drops evidence."),
            ({"embeddings": "fasttext"}, "Initialise with fastText Swahili vectors",
             "Pretrained vectors give rare words a useful starting point (52% of the vocabulary appears once)."),
            ({"class_weights": True}, "Class-weighted cross-entropy",
             "Weighting minority classes should raise recall on 'afya' and 'uchumi'."),
        ],
    },
}


def build_model(model_type, config, vocabulary, embedding_cache):
    embeddings = None
    if config["embeddings"] == "fasttext":
        if "matrix" not in embedding_cache:
            embedding_cache["matrix"], embedding_cache["coverage"] = load_fasttext_matrix(vocabulary)
        embeddings = embedding_cache["matrix"]
    return BiLSTMClassifier(len(vocabulary), NUM_CLASSES, hidden_size=config["hidden_size"],
                            dropout=config["dropout"], pooling=config["pooling"], embeddings=embeddings)


def predict(model, loader, device, loss_function):
    model.eval()
    probabilities, total_loss = [], 0.0
    with torch.no_grad():
        for ids, labels in loader:
            logits = model(ids.to(device))
            total_loss += loss_function(logits, labels.to(device)).item() * len(labels)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    in_length_order = np.concatenate(probabilities)
    original_order = np.empty_like(in_length_order)
    original_order[loader.batch_sampler.order] = in_length_order
    return original_order, total_loss / len(loader.dataset)


def train_one(model_type, config, splits, vocabulary, embedding_cache, seed, device):
    """Train with early stopping on validation macro-F1 and return the best model's outputs."""
    set_seed(seed)
    train, validation, test = splits
    train_loader = make_loader(train, vocabulary, config["max_length"], config["batch_size"], shuffle=True)
    val_loader = make_loader(validation, vocabulary, config["max_length"], 256, shuffle=False)
    test_loader = make_loader(test, vocabulary, config["max_length"], 256, shuffle=False)

    model = build_model(model_type, config, vocabulary, embedding_cache).to(device)
    weights = None
    if config["class_weights"]:
        weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train["label"])
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    train_loss_function = nn.CrossEntropyLoss(weight=weights)
    eval_loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)

    history, best_f1, best_state, patience = [], -1, None, 0
    start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = train_loss_function(model(ids), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * len(labels)

        val_probabilities, val_loss = predict(model, val_loader, device, eval_loss_function)
        val_f1 = compute_metrics(validation["label"], val_probabilities)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": running_loss / len(train), "val_loss": val_loss,
                        "val_macro_f1": val_f1})
        print(f"  epoch {epoch}: train_loss={history[-1]['train_loss']:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 2:
                break

    model.load_state_dict(best_state)
    val_probabilities, _ = predict(model, val_loader, device, eval_loss_function)
    test_probabilities, _ = predict(model, test_loader, device, eval_loss_function)
    return {
        "model": model,
        "history": history,
        "val_probabilities": val_probabilities,
        "test_probabilities": test_probabilities,
        "val_metrics": compute_metrics(validation["label"], val_probabilities),
        "seconds": round(time.time() - start, 1),
        "parameters": sum(p.numel() for p in model.parameters()),
    }


def run_plan(model_type, device):
    plan = PLANS[model_type]
    train, validation, test = load_splits()
    train, validation, test, _ = clean_splits(train, validation, test)
    splits = (train, validation, test)
    vocabulary = Vocabulary(train["clean_text"])
    print(f"vocabulary size: {len(vocabulary)}")
    embedding_cache, step_results = {}, {}
    history_dir = RESULTS_DIR / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    def run_step(experiment_id, config, change, hypothesis):
        print(f"[{experiment_id}] {change}")
        result = train_one(model_type, config, splits, vocabulary, embedding_cache, 42, device)
        step_results[json.dumps(config, sort_keys=True)] = result
        extra = {"config": config, "epochs_run": len(result["history"]), "seconds": result["seconds"],
                 "parameters": result["parameters"]}
        log_experiment(experiment_id, plan["name"], plan["owner"], change, hypothesis, result["val_metrics"],
                       extra=extra)
        with open(history_dir / f"{experiment_id}.json", "w") as f:
            json.dump(result["history"], f)
        return result["val_metrics"]["macro_f1"]

    best_config = dict(plan["start"])
    best_f1 = run_step(f"{plan['prefix']}01", best_config, *plan["first"])
    for number, (change_values, change, hypothesis) in enumerate(plan["steps"], start=2):
        candidate = {**best_config, **change_values}
        score = run_step(f"{plan['prefix']}{number:02d}", candidate, change, hypothesis)
        if score > best_f1:
            best_config, best_f1 = candidate, score
            print(f"  kept (val macro-F1 {score:.4f})")
        else:
            print(f"  not kept ({score:.4f} <= {best_f1:.4f})")

    final_id = f"{plan['prefix']}{len(plan['steps']) + 2:02d}"
    seed_scores, main_run = [], None
    for seed in FINAL_SEEDS:
        print(f"[{final_id}] final configuration, seed {seed}")
        if seed == 42:
            result = step_results[json.dumps(best_config, sort_keys=True)]
        else:
            result = train_one(model_type, best_config, splits, vocabulary, embedding_cache, seed, device)
        test_metrics = compute_metrics(test["label"], result["test_probabilities"])
        seed_scores.append({"seed": seed, "val_macro_f1": result["val_metrics"]["macro_f1"],
                            "test_macro_f1": test_metrics["macro_f1"], "test_accuracy": test_metrics["accuracy"]})
        if main_run is None:
            main_run = (result, test_metrics)

    result, test_metrics = main_run
    test_scores = [s["test_macro_f1"] for s in seed_scores]
    summary = {"config": best_config, "seeds": seed_scores, "test_macro_f1_mean": round(float(np.mean(test_scores)), 4),
               "test_macro_f1_std": round(float(np.std(test_scores)), 4), "seconds": result["seconds"],
               "parameters": result["parameters"], "fasttext_coverage": embedding_cache.get("coverage")}
    log_experiment(final_id, plan["name"], plan["owner"],
                   f"Final configuration, {len(FINAL_SEEDS)} seeds (test macro-F1 {summary['test_macro_f1_mean']} "
                   f"+/- {summary['test_macro_f1_std']})",
                   "Selected configuration, evaluated once on test.", result["val_metrics"], test_metrics, extra=summary)

    save_predictions(model_type, "validation", validation, result["val_probabilities"])
    save_predictions(model_type, "test", test, result["test_probabilities"])
    save_metrics(model_type, {"validation": result["val_metrics"], "test": test_metrics, "summary": summary})
    with open(history_dir / f"{final_id}.json", "w") as f:
        json.dump(result["history"], f)
    plot_learning_curves(result["history"], f"{plan['name']} learning curves",
                         figure_path(f"learning_curves_{model_type}.png"))
    plot_confusion_matrix(test["label"], result["test_probabilities"].argmax(axis=1), f"{plan['name']} (test)",
                          figure_path(f"confusion_{model_type}.png"))
    torch.save({"state_dict": result["model"].state_dict(), "config": best_config, "words": vocabulary.words},
               RESULTS_DIR / f"{model_type}_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bilstm"], required=True)
    parser.add_argument("--device", default=None, help="cpu, cuda or mps; defaults to the fastest available")
    args = parser.parse_args()
    run_plan(args.model, torch.device(args.device) if args.device else get_device())
