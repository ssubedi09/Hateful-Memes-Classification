import os, json, pickle, csv, pathlib, torch, optuna
import numpy as np
from collections import Counter
from datasets import load_dataset, Features, Value, ClassLabel
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, default_data_collator)
from sklearn.metrics import f1_score, classification_report
from accelerate import Accelerator
from torch.nn import functional as F

# ───────────────────────── UTILITY ─────────────────────────

def compute_metrics(p):
    preds = p.predictions
    
    if preds.ndim > 1:
        preds = np.argmax(preds, axis=-1)
    
    return {"f1": f1_score(p.label_ids, preds, average="macro")}

def preprocess(b):
    enc = tokzr(
        b["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_attention_mask=True,
    )
    enc["labels"] = b["label"]  # rename to key Trainer/model expects
    return enc

# ───────────────────────── ARGUMENTS ─────────────────────────

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "run_logs")
N_TRIALS = int(os.environ.get("N_TRIALS", 25))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", 500))
pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

# ───────────────────────── DATA LOAD ─────────────────────────
features = Features({
    "id":    Value("string"),
    "img":   Value("string"),
    "label": ClassLabel(names=["clean", "hateful"]),
    "text":  Value("string"),
})

raw = load_dataset(
    "json",
    data_files={
        "train":      "../data/train.jsonl",
        "validation": "../data/dev.jsonl",
        "test":       "../data/test.jsonl",
    },
    features=features,
    split=None,
)

tokzr = AutoTokenizer.from_pretrained("GroNLP/hatebert")



data_train = raw["train"].map(preprocess, remove_columns=["id","img","text","label"])
data_val = raw["validation"].map(preprocess, remove_columns=["id","img","text","label"])
data_test = raw["test"].map(preprocess, remove_columns=["id","img","text","label"])

# ─────────────── CLASS‑WEIGHT VECTOR ────────────────
counts = Counter(raw["train"]["label"])
total = sum(counts.values())
weights = torch.tensor([total/counts[c] for c in range(2)], dtype=torch.float32)

# ───────────────────── TRAINER ──────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(
                      weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
# ────────────────────  HPO OBJECTIVE ───────────────────────
def objective(trial):
    lr           = trial.suggest_float("learning_rate", 1e-5, 7e-5, log=True)
    wd           = trial.suggest_float("weight_decay", 0.0, 0.3)
    warmup_steps = trial.suggest_int("warmup_steps", 0, 500)
    epochs       = trial.suggest_int("epochs", 2, 4)
    batch_size   = trial.suggest_categorical("batch_size", [8, 16])

    args = TrainingArguments(
        output_dir       = os.path.join(OUTPUT_DIR, f"trial-{trial.number}"),
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate    = lr,
        weight_decay     = wd,
        warmup_steps     = warmup_steps,
        eval_strategy    ="steps",
        eval_steps       = EVAL_STEPS,
        logging_steps    = EVAL_STEPS,
        num_train_epochs = epochs,
        fp16             = torch.cuda.is_available(),
        report_to        = "none",
        save_strategy    = "no",          # no mid‑trial checkpoints
        seed             = 42,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
                "GroNLP/hatebert", num_labels=2)
    
    trainer = WeightedTrainer(
        model          = model,
        class_weights  = weights,
        args           = args,
        train_dataset  = data_train,
        eval_dataset   = data_val,
        tokenizer      = tokzr,
        data_collator  = default_data_collator,
        compute_metrics=compute_metrics
    )

    eval_result = trainer.train()
    f1 = trainer.evaluate()["eval_f1"]

    # — log trial result to CSV —
    csv_path = os.path.join(OUTPUT_DIR, "trial_metrics.csv")
    first = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as cf:
        writer = csv.writer(cf)
        if first:
            writer.writerow(["trial","f1","lr","wd","warmup","epochs","bs"])
        writer.writerow([trial.number, f1, lr, wd, warmup_steps, epochs, batch_size])

    return f1

# ────────────────────── SEARCH RUN ─────────────────────
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

# ───────────────── SAVE BEST MODEL & STUDY ─────────────────
best_params = study.best_params
print("Best params:", best_params)
json.dump(best_params,
          open(os.path.join(OUTPUT_DIR, "best_params.json"), "w"),
          indent=2)

# save full Optuna study for later analysis
with open(os.path.join(OUTPUT_DIR, "optuna_study.pkl"), "wb") as f:
    pickle.dump(study, f)
