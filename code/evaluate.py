# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Functions for: evaluation metrics & reports, baseline evaluation,
# confusion matrices, and inference (prediction) helpers.
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import prepare_context_attributes


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates a trained LSTM model on test data. Prints a classification report
    and per-class accuracy.
    """
    y_pred_probs = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_, zero_division=0))

    cm = confusion_matrix(y_test, y_pred_labels)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    print("\nPer-Class Accuracy:")
    for label, acc in zip(label_encoder.classes_, per_class_accuracy):
        print(f"{label}: {acc:.2%}")


def evaluate_baseline(y_true, label_encoder, random_seed: int = 42):
    """
    Evaluate a distribution-based baseline by sampling predictions based on
    the observed class distribution in y_true.
    """
    np.random.seed(random_seed)

    class_counts = np.bincount(y_true)
    class_probs = class_counts / class_counts.sum()

    y_pred = np.random.choice(
        np.arange(len(class_probs)),
        size=len(y_true),
        p=class_probs
    )

    target_names = label_encoder.classes_
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    return report


def compare_f1_for_trained_model(dp, data_per_dp, decision_point_models):
    """
    Compares weighted F1 between a trained LSTM and the distribution-based baseline.
    Also plots the confusion matrix.
    """
    if dp not in data_per_dp or dp not in decision_point_models:
        print(f"Missing data or model for {dp}")
        return

    df_holdout = data_per_dp[dp]
    if df_holdout.empty:
        print(f"Holdout set for {dp} is empty.")
        return

    model_bundle = decision_point_models[dp]
    model = model_bundle["model"]
    activity_enc = model_bundle["activity_encoder"]
    resource_enc = model_bundle["resource_encoder"]
    label_enc = model_bundle["label_encoder"]
    context_keys = model_bundle["context_keys"]
    max_seq_len = model_bundle["max_seq_len"]

    known_resources = set(resource_enc.classes_)
    resources_cleaned = [
        [res if res in known_resources else "UNKNOWN" for res in seq]
        for seq in df_holdout["sequence_resources"]
    ]
    if "UNKNOWN" not in resource_enc.classes_:
        resource_enc.classes_ = np.append(resource_enc.classes_, "UNKNOWN")
    X_res = [resource_enc.transform(seq) for seq in resources_cleaned]

    known_activities = set(activity_enc.classes_)
    activities_cleaned = [
        [act if act in known_activities else "UNKNOWN" for act in seq]
        for seq in df_holdout["sequence"]
    ]
    if "UNKNOWN" not in activity_enc.classes_:
        activity_enc.classes_ = np.append(activity_enc.classes_, "UNKNOWN")
    X_acts = [activity_enc.transform(seq) for seq in activities_cleaned]

    X_durs = df_holdout["sequence_durations"].tolist()

    context_out = prepare_context_attributes(df_holdout, context_keys)
    X_context = context_out[0] if isinstance(context_out, tuple) else context_out

    y_true = label_enc.transform(df_holdout["label"])

    X_acts_padded = pad_sequences(X_acts, maxlen=max_seq_len, padding="pre")
    X_res_padded  = pad_sequences(X_res,  maxlen=max_seq_len, padding="pre")
    X_durs_padded = pad_sequences(X_durs, maxlen=max_seq_len, padding="pre", dtype="float32")

    y_pred_probs = model.predict([
        X_acts_padded,
        X_durs_padded,
        X_res_padded,
        X_context
    ])
    y_pred = np.argmax(y_pred_probs, axis=1)

    lstm_report = classification_report(
        y_true, y_pred, target_names=label_enc.classes_, output_dict=True, zero_division=0
    )
    baseline_report = evaluate_baseline(y_true, label_enc)

    f1_lstm = lstm_report["weighted avg"]["f1-score"]
    f1_baseline = baseline_report["weighted avg"]["f1-score"]
    relative_improvement = (f1_lstm - f1_baseline) / f1_baseline * 100 if f1_baseline > 0 else float("inf")

    print(f"\nComparison for {dp} (Holdout Set)")
    print(f"Weighted F1 (LSTM):     {f1_lstm:.3f}")
    print(f"Weighted F1 (Baseline): {f1_baseline:.3f}")
    print(f"Relative Improvement:   {relative_improvement:.2f}%")

    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_enc.classes_)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_enc.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(include_values=True, cmap="Blues", ax=ax, xticks_rotation=45)
    plt.title(f"Confusion Matrix for {dp}")
    plt.tight_layout()
    plt.show()

    return {
        "dp": dp,
        "f1_lstm": f1_lstm,
        "f1_baseline": f1_baseline,
        "relative_improvement": relative_improvement
    }


# ─────────────────────────
# Inference / Prediction API
# ─────────────────────────
def predict_next_activity_probs_simple(input_sample, decision_point, models, encoders, max_seq_len):
    """
    Predicts probabilities for each possible next activity at a given decision point.
    input_sample: dict with keys sequence, durations, resources, context (already encoded)
    """
    model = models[decision_point]
    enc = encoders[decision_point]
    act_enc = enc["activity"]
    res_enc = enc["resource"]
    label_enc = enc["label"]

    unknown_acts = [a for a in input_sample["sequence"] if a not in act_enc.classes_]
    unknown_res = [r for r in input_sample["resources"] if r not in res_enc.classes_]
    if unknown_acts:
        raise ValueError(f"Unknown activity labels: {unknown_acts}")
    if unknown_res:
        raise ValueError(f"Unknown resource labels: {unknown_res}")

    X_acts = act_enc.transform(input_sample["sequence"])
    X_res  = res_enc.transform(input_sample["resources"])
    X_durs = input_sample["durations"]
    X_ctx  = np.array([input_sample["context"]], dtype="float32")

    X_acts_padded = pad_sequences([X_acts], maxlen=max_seq_len, padding="pre")
    X_res_padded  = pad_sequences([X_res],  maxlen=max_seq_len, padding="pre")
    X_durs_padded = pad_sequences([X_durs], maxlen=max_seq_len, padding="pre", dtype="float32")

    pred_probs = model.predict([X_acts_padded, X_durs_padded, X_res_padded, X_ctx], verbose=0)
    return pred_probs[0]


def predict_next_activity_probs_advanced(input_sample, decision_point, models_dict):
    """
    Predicts next activity probabilities for advanced models at a given decision point.
    input_sample: dict with keys sequence, durations, resources, context (already encoded/scaled)
    """
    model_data = models_dict[decision_point]
    model = model_data["model"]
    act_enc = model_data["activity_encoder"]
    res_enc = model_data["resource_encoder"]
    label_enc = model_data["label_encoder"]
    max_seq_len = model_data["max_seq_len"]

    known_acts = set(act_enc.classes_)
    known_res  = set(res_enc.classes_)

    sequence = [act if act in known_acts else "UNKNOWN" for act in input_sample["sequence"]]
    resources = [res if res in known_res else "UNKNOWN" for res in input_sample["resources"]]

    X_acts = act_enc.transform(sequence)
    X_res  = res_enc.transform(resources)
    X_durs = input_sample["durations"]
    X_ctx  = np.array([input_sample["context"]], dtype="float32")

    X_acts_padded = pad_sequences([X_acts], maxlen=max_seq_len, padding="pre")
    X_res_padded  = pad_sequences([X_res],  maxlen=max_seq_len, padding="pre")
    X_durs_padded = pad_sequences([X_durs], maxlen=max_seq_len, padding="pre", dtype="float32")

    pred_probs = model.predict([X_acts_padded, X_durs_padded, X_res_padded, X_ctx], verbose=0)
    return pred_probs[0]


def predict_probs_for_samples(
    samples,                       # DataFrame or list of DataFrames
    dp_name: str,
    models_dict: Dict,
    bpmn_decision_point_map: Dict  # BPMN metadata
) -> pd.DataFrame:
    """
    Returns a DataFrame with class probabilities for all given samples at a decision point.
    If the DP has a single outgoing edge, returns a degenerate distribution (prob=1 for that branch).
    """
    if isinstance(samples, pd.DataFrame):
        samples = [samples]
    elif not isinstance(samples, (list, tuple)):
        raise TypeError("samples must be a DataFrame or a list/tuple of DataFrames")

    if dp_name not in models_dict:
        raise KeyError(f"No model for Decision Point '{dp_name}' found")

    md       = models_dict[dp_name]
    model    = md["model"]
    act_enc  = md["activity_encoder"]
    res_enc  = md["resource_encoder"]
    lbl_enc  = md["label_encoder"]
    ctx_keys = md["context_keys"]
    ctx_enc  = md["context_encoders"]
    max_len  = md["max_seq_len"]

    dp_cfg = bpmn_decision_point_map.get(dp_name)
    if dp_cfg is None:
        raise KeyError(f"'{dp_name}' not found in bpmn_decision_point_map")

    outgoing = dp_cfg["outgoing"]
    single_branch = len(outgoing) == 1
    if single_branch:
        fixed_act = outgoing[0]
        fixed_idx = np.where(lbl_enc.classes_ == fixed_act)[0]
        if len(fixed_idx) == 0:
            raise ValueError(
                f"Label encoder for {dp_name} does not know activity '{fixed_act}' "
                f"(classes: {list(lbl_enc.classes_)})"
            )
        fixed_idx = fixed_idx[0]

    result_rows = []

    def _left_pad(arr, fill):
        arr = np.asarray(arr).reshape(1, -1)
        pad_len = max_len - arr.shape[1]
        if pad_len < 0:
            arr = arr[:, -max_len:]
            pad_len = 0
        pad_block = np.full((1, pad_len), fill, dtype=arr.dtype)
        return np.hstack([pad_block, arr])

    for idx, sample_df in enumerate(samples, start=1):
        row = sample_df.iloc[0]

        # Encode context
        ctx_vec = []
        for col in ctx_keys:
            enc = ctx_enc[col]
            val = row[col]
            from sklearn.preprocessing import LabelEncoder as _LE
            if isinstance(enc, _LE):
                if val not in enc.classes_:
                    raise ValueError(f"Unknown value '{val}' in column '{col}'")
                ctx_vec.append(enc.transform([val])[0])
            else:  # StandardScaler
                ctx_vec.append(enc.transform([[float(val)]])[0][0])
        ctx_vec = np.asarray(ctx_vec, dtype="float32")[None, :]

        if single_branch:
            probs = np.zeros(len(lbl_enc.classes_), dtype="float32")
            probs[fixed_idx] = 1.0
        else:
            X_acts = _left_pad(act_enc.transform(row["sequence"]), 0)
            X_res  = _left_pad(res_enc.transform(row["sequence_resources"]), 0)
            X_durs = _left_pad(row["sequence_durations"], 0.0).astype("float32")

            probs = model.predict([X_acts, X_durs, X_res, ctx_vec], verbose=0)[0]

        result_rows.append(pd.Series(probs, index=lbl_enc.classes_, name=f"Sample {idx}"))

    return pd.DataFrame(result_rows)

def add_unknown_label(enc, token="UNKNOWN"):
    if hasattr(enc, "classes_"):
        if token not in enc.classes_:
            enc.classes_ = np.append(enc.classes_, token)
    return enc

def model_score(model, X, y_true, metric=accuracy_score):
    y_pred = model.predict(X, verbose=0)
    y_pred_classes = y_pred.argmax(axis=-1)
    if hasattr(y_true, "shape") and len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
    return metric(y_true, y_pred_classes)

def permutation_importance_context(model, X_acts, X_durs, X_res, X_ctx, y, feature_names, n_repeats=5, random_state=42, metric=accuracy_score):
    rng = np.random.RandomState(random_state)
    baseline_pred = np.argmax(model.predict([X_acts, X_durs, X_res, X_ctx], verbose=0), axis=1)
    if hasattr(y, "shape") and len(y.shape) > 1 and y.shape[1] == 1:
        y_flat = y.flatten()
    else:
        y_flat = y
    baseline_score = metric(y_flat, baseline_pred)
    importances = []
    for i in range(len(feature_names)):
        scores = []
        for _ in range(n_repeats):
            X_ctx_perm = X_ctx.copy()
            perm_idx = rng.permutation(X_ctx_perm.shape[0])
            X_ctx_perm[:, i] = X_ctx_perm[perm_idx, i]
            perm_pred = np.argmax(model.predict([X_acts, X_durs, X_res, X_ctx_perm], verbose=0), axis=1)
            score = metric(y_flat, perm_pred)
            scores.append(baseline_score - score)
        importances.append(float(np.mean(scores)))
    return np.array(importances, dtype=float)

def permutation_importance_all_features(model, X_input_list, y_true, context_feature_names, n_repeats=3, metric=accuracy_score, random_state=42):
    rng = np.random.RandomState(random_state)
    X_acts, X_durs, X_res, X_ctx = X_input_list
    if hasattr(y_true, "shape") and len(y_true.shape) > 1 and y_true.shape[1] == 1:
        y_flat = y_true.flatten()
    else:
        y_flat = y_true
    baseline_pred = np.argmax(model.predict([X_acts, X_durs, X_res, X_ctx], verbose=0), axis=1)
    baseline_score = metric(y_flat, baseline_pred)
    importances = []
    feature_names = []
    for i, feat in enumerate(context_feature_names):
        scores = []
        for _ in range(n_repeats):
            X_ctx_perm = X_ctx.copy()
            perm_idx = rng.permutation(X_ctx_perm.shape[0])
            X_ctx_perm[:, i] = X_ctx_perm[perm_idx, i]
            perm_pred = np.argmax(model.predict([X_acts, X_durs, X_res, X_ctx_perm], verbose=0), axis=1)
            score = metric(y_flat, perm_pred)
            scores.append(baseline_score - score)
        importances.append(float(np.mean(scores)))
        feature_names.append(feat)
    seqs = [
        (0, X_acts, "activity_sequence"),
        (1, X_durs, "duration_sequence"),
        (2, X_res,  "resource_sequence"),
    ]
    for idx, X_seq, name in seqs:
        scores = []
        for _ in range(n_repeats):
            perm_idx = rng.permutation(X_seq.shape[0])
            X_seq_perm = X_seq[perm_idx]
            X_inputs_perm = [X_acts, X_durs, X_res, X_ctx]
            X_inputs_perm[idx] = X_seq_perm
            perm_pred = np.argmax(model.predict(X_inputs_perm, verbose=0), axis=1)
            score = metric(y_flat, perm_pred)
            scores.append(baseline_score - score)
        importances.append(float(np.mean(scores)))
        feature_names.append(name)
    return np.array(importances, dtype=float), feature_names, float(baseline_score)

def plot_feature_importance(feature_names, importances, title):
    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(np.array(feature_names)[sorted_idx], np.array(importances)[sorted_idx])
    plt.xlabel("Permutation Importance (Δ accuracy)")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()