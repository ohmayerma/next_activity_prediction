# ───────────────────────────────────────────
# Standard-Bibliothek
# ───────────────────────────────────────────
import re
from collections import defaultdict, OrderedDict, Counter
from functools import partial

# ───────────────────────────────────────────
# Externe Bibliotheken
# ───────────────────────────────────────────
import pandas as pd
import numpy as np
import holidays
import joblib
import keras_tuner as kt               # Hyperparameter-Tuning

# ───────────────────────────────────────────
# Scikit-learn
# ───────────────────────────────────────────
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ───────────────────────────────────────────
# TensorFlow / Keras
# ───────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
    Lambda,
    GlobalMaxPooling1D,
    LayerNormalization,
    LeakyReLU
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


def extract_bpmn_decision_point_map(bpmn_model):
    """
    Extracts a unified map of BPMN decision points from a PM4Py BPMN model.

    Returns:
        dict of { dp_name: { 'incoming': [act1, act2, …], 'outgoing': [actA, actB, …] } }
    """
    # Helper to pull out numeric index from "DP N"
    def _dp_number(label):
        m = re.search(r"DP\s*(\d+)", label)
        return int(m.group(1)) if m else float('inf')

    # 1) Gather raw outgoing lists of gateway → [next_node.name]
    raw_out = defaultdict(list)
    for flow in bpmn_model.get_flows():
        src, tgt = flow.source, flow.target
        # consider any gateway node by class‐name
        if "Gateway" in src.__class__.__name__ and getattr(tgt, "name", None):
            raw_out[src.name or src.id].append(tgt.name)

    # 2) Gather raw incoming lists of gateway ← [prev_node.name]
    raw_in = defaultdict(list)
    for flow in bpmn_model.get_flows():
        src, tgt = flow.source, flow.target
        if getattr(tgt, "name", None) and "Gateway" in tgt.__class__.__name__:
            raw_in[tgt.name].append(src.name or src.id)

    # 3) Sort the gateways by DP number
    sorted_out = OrderedDict(sorted(raw_out.items(), key=lambda kv: _dp_number(kv[0])))
    sorted_in  = OrderedDict(sorted(raw_in .items(), key=lambda kv: _dp_number(kv[0])))

    # 4) Resolve any references to other DP labels (flatten)
    def _resolve(map_raw, key, seen=None):
        seen = set() if seen is None else seen
        result = []
        for name in map_raw.get(key, []):
            name = name.strip()
            if name in map_raw and name not in seen:
                seen.add(name)
                result += _resolve(map_raw, name, seen=seen.copy())
            elif not name.startswith("DP") and not name.startswith("PG"):
                result.append(name)
        return result

    # 5) Build cleaned outgoing/incoming with dedupe
    out_clean = {}
    for dp, targets in sorted_out.items():
        flat = _resolve(sorted_out, dp)
        # preserve order but drop duplicates
        seen = set()
        out_clean[dp] = [x for x in flat if x not in seen and not seen.add(x)]

    in_clean = {}
    for dp, sources in sorted_in.items():
        flat = _resolve(sorted_in, dp)
        seen = set()
        in_clean[dp] = [x for x in flat if x not in seen and not seen.add(x)]

    # 6) Merge into final map, only for DP labels
    all_dp = sorted(set(out_clean) | set(in_clean), key=_dp_number)
    dp_map = {}
    for dp in all_dp:
        if dp.startswith("DP"):
            dp_map[dp] = {
                'incoming': in_clean.get(dp, []),
                'outgoing': out_clean.get(dp, [])
            }

    return dp_map



def generate_enriched_training_sets_simple(
    df_log,
    bpmn_decision_point_map,
    max_history_len=10,
    min_sequence_count=20,
    min_class_count=10
):
    # Step 1: Build clean (prev, next) → DP map
    transition_to_dps = defaultdict(set)
    for dp, config in bpmn_decision_point_map.items():
        for prev in config['incoming']:
            for nxt in config['outgoing']:
                transition_to_dps[(prev, nxt)].add(dp)

    clean_transition_to_dp = {
        (prev, nxt): list(dps)[0]
        for (prev, nxt), dps in transition_to_dps.items()
        if len(dps) == 1
    }

    # Step 2: Case-level attributes
    case_columns = ['case:LoanGoal', 'case:ApplicationType', 'case:RequestedAmount']
    case_attr_df = df_log.drop_duplicates('case:concept:name')[
        ['case:concept:name'] + case_columns
    ].set_index('case:concept:name')

    # Step 3: Build raw training data
    dp_to_training_rows = defaultdict(list)
    df_sorted = df_log.sort_values(['case:concept:name', 'time:timestamp'])

    for case_id, group in df_sorted.groupby('case:concept:name'):
        events = group['concept:name'].tolist()
        resources = group['org:resource'].tolist()
        timestamps = group['time:timestamp'].tolist()

        for i in range(len(events) - 1):
            prev, nxt = events[i], events[i + 1]
            dp = clean_transition_to_dp.get((prev, nxt))

            if dp:
                start_idx = max(0, i - max_history_len + 1)
                sequence = events[start_idx:i + 1]
                sequence_resources = resources[start_idx:i + 1]
                sequence_timestamps = timestamps[start_idx:i + 1]

                sequence_durations = []
                for j in range(start_idx + 1, i + 1):
                    duration = (timestamps[j] - timestamps[j - 1]).total_seconds()
                    sequence_durations.append(duration)
                if len(sequence_durations) < len(sequence):
                    sequence_durations.insert(0, 0.0)

                label = nxt
                case_features = case_attr_df.loc[case_id].to_dict() if case_id in case_attr_df.index else {}

                dp_to_training_rows[dp].append({
                    'case_id': case_id,
                    'sequence': sequence,
                    'sequence_resources': sequence_resources,
                    'sequence_durations': sequence_durations,
                    'sequence_timestamps': sequence_timestamps,
                    'label': label,
                    **case_features
                })

    # Step 4: Convert to DataFrames
    dp_to_df_raw = {
        dp: pd.DataFrame(rows)
        for dp, rows in dp_to_training_rows.items()
        if rows
    }

    # Step 5: Filter out infrequent sequences AND infrequent classes
    dp_to_df_filtered = {}

    for dp, df in dp_to_df_raw.items():
        # Drop sequences that are too rare
        df['sequence_str'] = df['sequence'].apply(lambda x: tuple(x))
        sequence_counts = df['sequence_str'].value_counts()
        valid_sequences = set(sequence_counts[sequence_counts > min_sequence_count].index)
        df = df[df['sequence_str'].apply(lambda x: x in valid_sequences)].drop(columns=['sequence_str'])

        # Drop class labels (next activities) that occur less than min_class_count
        class_counts = df['label'].value_counts()
        valid_labels = class_counts[class_counts >= min_class_count].index
        df = df[df['label'].isin(valid_labels)]

        if not df.empty and df['label'].nunique() >= 2:
            dp_to_df_filtered[dp] = df

    return dp_to_df_filtered


def prepare_sequences_and_labels(df):
    """
    Prepares activity, duration, and resource sequences along with label encoding and padding.

    Args:
        df (pd.DataFrame): DataFrame with columns ["sequence", "durations", "resources", "label"]

    Returns:
        X_acts_padded: np.ndarray, padded encoded activity sequences
        X_durs_padded: np.ndarray, padded duration sequences (float32)
        X_res_padded: np.ndarray, padded encoded resource sequences
        activity_encoder: fitted LabelEncoder for activities
        resource_encoder: fitted LabelEncoder for resources
        label_encoder: fitted LabelEncoder for labels
        y: np.ndarray, encoded labels
        max_seq_len: int, maximum sequence length used for padding
    """

    # --- Step 1: Extract columns ---
    activity_seqs = df["sequence"]
    duration_seqs = df["sequence_durations"]
    resource_seqs = df["sequence_resources"]
    labels = df["label"]

    # --- Step 2: Encode activity names ---
    all_activities = [act for seq in activity_seqs for act in seq]
    activity_encoder = LabelEncoder()
    activity_encoder.fit(all_activities)
    X_acts = [activity_encoder.transform(seq) for seq in activity_seqs]

    # --- Step 3: Encode resources ---
    all_resources = [res for seq in resource_seqs for res in seq]
    resource_encoder = LabelEncoder()
    resource_encoder.fit(all_resources)
    X_res = [resource_encoder.transform(seq) for seq in resource_seqs]

    # --- Step 4: Determine max sequence length ---
    max_seq_len = max(len(seq) for seq in X_acts)

    # --- Step 5: Pad sequences with proper dtypes ---
    X_acts_padded = pad_sequences(X_acts, maxlen=max_seq_len, padding="pre").astype("int32")
    X_durs_padded = pad_sequences(duration_seqs, maxlen=max_seq_len, padding="pre", dtype="float32").astype("float32")
    X_res_padded = pad_sequences(X_res, maxlen=max_seq_len, padding="pre").astype("int32")

    # --- Step 6: Encode labels ---
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels).astype("int32")

    return X_acts_padded, X_durs_padded, X_res_padded, activity_encoder, resource_encoder, label_encoder, y, max_seq_len


def prepare_context_attributes(df, context_keys):
    """
    Encodes and scales context attributes and stores encoders/scalers.

    Returns:
        X_context_array: np.ndarray
        context_dim: int
        context_encoders: dict of encoders/scalers used
    """
    X_context = df[context_keys].copy()
    context_encoders = {}

    for col in context_keys:
        col_data = X_context[col]

        # If dtype is object or string, treat as categorical
        if col_data.dtype == "object" or col_data.dtype.name == "string":
            le = LabelEncoder()
            X_context[col] = le.fit_transform(col_data.astype(str))
            context_encoders[col] = le
        else:
            # Treat as numeric
            scaler = StandardScaler()
            X_context[col] = scaler.fit_transform(col_data.values.reshape(-1, 1))
            context_encoders[col] = scaler

    X_context_array = X_context.to_numpy().astype("float32")
    context_dim = X_context_array.shape[1]

    return X_context_array, context_dim, context_encoders


def expand_and_cast(x):
    return tf.cast(tf.expand_dims(x, axis=-1), tf.float32)


def build_lstm_model(num_activities, num_resources, context_dim, max_seq_len, num_classes):
    """
    Builds and compiles an LSTM-based model for next-activity prediction.

    Inputs:
        - Activity sequences (embedded, mask-aware)
        - Duration sequences (numerical)
        - Resource sequences (embedded, mask-aware)
        - Context features (numerical/categorical)

    Args:
        num_activities (int): Size of the activity vocabulary
        num_resources (int): Size of the resource vocabulary
        context_dim (int): Number of context features
        max_seq_len (int): Sequence length (padded)
        num_classes (int): Number of output classes (activities)

    Returns:
        model (tf.keras.Model): Compiled LSTM classification model
    """

    # --- Input layers ---
    activity_input = Input(shape=(max_seq_len,), name="activity_input")
    duration_input = Input(shape=(max_seq_len,), name="duration_input", dtype="float32")
    resource_input = Input(shape=(max_seq_len,), name="resource_input")
    context_input = Input(shape=(context_dim,), name="context_input")

    # --- Embeddings ---
    activity_emb = Embedding(input_dim=num_activities, output_dim=64, name="activity_embedding")(activity_input)
    resource_emb = Embedding(input_dim=num_resources, output_dim=16, name="resource_embedding")(resource_input)


    # --- Merge sequential inputs ---
    duration_expanded = Lambda(
        expand_and_cast,
        output_shape=lambda s: (s[0], s[1], 1),
        name="expand_and_cast"
    )(duration_input)
    seq_concat = Concatenate(axis=-1)([activity_emb, duration_expanded, resource_emb])

    # --- LSTM encoder ---
    x_seq = LSTM(units=128, return_sequences=False)(seq_concat)

    # --- Merge with context ---
    x = Concatenate()([x_seq, context_input])
    x = Dense(64, activation="relu")(x)

    # --- Output layer ---
    output = Dense(num_classes, activation="softmax")(x)

    # --- Compile model ---
    model = Model(inputs=[activity_input, duration_input, resource_input, context_input], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model


def train_model(model, X_train, y_train):
    """
    Trains the LSTM model with early stopping.
    
    Inputs:
        model: compiled Keras model
        X_train: list of inputs [X_acts, X_durs, X_res, X_context]
        y_train: target labels (encoded)
    
    Returns:
        history: training history object
    """

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=3,
        min_delta=1e-4,
        baseline=0.95,        # optional: stop as soon as ≥99.9 %
        restore_best_weights=True,
        verbose=1
    )

    # --- Train the model ---
    history = model.fit(
        X_train,            # Training input data (list of arrays: [activities, durations, resources, context])
        y_train,            # Corresponding labels (next activity class, integer-encoded)
        validation_split=0.1,   # Use 10% of the training data for validation
        callbacks=[early_stop],
        epochs=10,              # Maximum number of training epochs
        batch_size=128,         # Number of samples per gradient update
    )

    return history


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates a trained LSTM model on test data.

    Args:
        model: Trained Keras model
        X_test: List of test input arrays [X_acts_test, X_durs_test, X_res_test, X_context_test]
        y_test: Ground truth labels (encoded)
        label_encoder: Fitted LabelEncoder for decoding labels
    """

    # --- Step 1: Predict test labels ---
    y_pred_probs = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)

    # --- Step 2: Classification report ---
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_labels, target_names=label_encoder.classes_), zero_division=0)

    # --- Step 3: Per-class accuracy ---
    cm = confusion_matrix(y_test, y_pred_labels)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    print("\nPer-Class Accuracy:")
    for label, acc in zip(label_encoder.classes_, per_class_accuracy):
        print(f"{label}: {acc:.2%}")


def evaluate_baseline(y_true, label_encoder, random_seed=42):
    """
    Evaluate a distribution-based baseline model.

    Instead of predicting the majority class, this model randomly samples
    predictions based on the class distribution observed in y_true.

    Args:
        y_true (array-like): True class labels (encoded integers)
        label_encoder: Fitted LabelEncoder used to decode class labels
        random_seed (int): Random seed for reproducibility

    Returns:
        report (dict): Classification report (precision, recall, F1, etc.)
    """
    np.random.seed(random_seed)

    # Step 1: Calculate empirical class distribution
    class_counts = np.bincount(y_true)
    class_probs = class_counts / class_counts.sum()

    # Step 2: Sample predictions from this distribution
    y_pred = np.random.choice(
        np.arange(len(class_probs)),
        size=len(y_true),
        p=class_probs
    )

    # Step 3: Generate classification report
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

    # Handle unseen resources
    known_resources = set(resource_enc.classes_)
    resources_cleaned = [
        [res if res in known_resources else "UNKNOWN" for res in seq]
        for seq in df_holdout["sequence_resources"]
    ]
    if "UNKNOWN" not in resource_enc.classes_:
        resource_enc.classes_ = np.append(resource_enc.classes_, "UNKNOWN")
    X_res = [resource_enc.transform(seq) for seq in resources_cleaned]

    # Handle unseen activities
    known_activities = set(activity_enc.classes_)
    activities_cleaned = [
        [act if act in known_activities else "UNKNOWN" for act in seq]
        for seq in df_holdout["sequence"]
    ]
    if "UNKNOWN" not in activity_enc.classes_:
        activity_enc.classes_ = np.append(activity_enc.classes_, "UNKNOWN")
    X_acts = [activity_enc.transform(seq) for seq in activities_cleaned]

    X_durs = df_holdout["sequence_durations"].tolist()

    # --- Änderung hier: unpacken, falls prepare_context_attributes mehr zurückgibt ---
    context_out = prepare_context_attributes(df_holdout, context_keys)
    if isinstance(context_out, tuple):
        # nur das erste Element ist das NumPy-Array für das Context-Input
        X_context = context_out[0]
    else:
        X_context = context_out

    y_true = label_enc.transform(df_holdout["label"])

    # Pad sequences
    X_acts_padded = pad_sequences(X_acts, maxlen=max_seq_len, padding="pre")
    X_res_padded  = pad_sequences(X_res,  maxlen=max_seq_len, padding="pre")
    X_durs_padded = pad_sequences(X_durs, maxlen=max_seq_len, padding="pre", dtype="float32")

    # Jetzt eine Liste **nur** aus NumPy-Arrays
    y_pred_probs = model.predict([
        X_acts_padded,
        X_durs_padded,
        X_res_padded,
        X_context
    ])
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Evaluate
    lstm_report = classification_report(
        y_true, y_pred, target_names=label_enc.classes_, output_dict=True, zero_division=0
    )
    baseline_report = evaluate_baseline(y_true, label_enc)

    f1_lstm = lstm_report["weighted avg"]["f1-score"]
    f1_majority = baseline_report["weighted avg"]["f1-score"]
    relative_improvement = (f1_lstm - f1_majority) / f1_majority * 100 if f1_majority > 0 else float("inf")

    print(f"\nComparison for {dp} (Holdout Set)")
    print(f"Weighted F1 (LSTM):     {f1_lstm:.3f}")
    print(f"Weighted F1 (Baseline): {f1_majority:.3f}")
    print(f"Relative Improvement:   {relative_improvement:.2f}%")

    # Plot confusion matrix
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
        "f1_baseline": f1_majority,
        "relative_improvement": relative_improvement
    }


def store_lstm_model(
    decision_point,
    model,
    activity_encoder,
    resource_encoder,
    label_encoder,
    context_keys,
    context_encoders,
    max_seq_len,
    storage_dict
):
    """
    Stores all necessary components of an LSTM-based decision point predictor, including context encoders.
    """
    storage_dict[decision_point] = {
        "model": model,
        "activity_encoder": activity_encoder,
        "resource_encoder": resource_encoder,
        "label_encoder": label_encoder,
        "context_keys": context_keys,
        "context_encoders": context_encoders,  
        "max_seq_len": max_seq_len
    }
    print(f"Stored model for {decision_point}")


def enrich_with_temporal_features(df):
    nl_holidays = holidays.Netherlands()

    enriched_rows = []  

    for _, row in df.iterrows():
        seq_len = len(row["sequence"])
        durs = row["sequence_durations"]
        total_duration = sum(durs)
        time_since_prev = durs[-1] if len(durs) > 0 else 0.0    

        # Use last timestamp in the sequence (to avoid leakage)
        last_event_ts = pd.to_datetime(row["sequence_timestamps"][-1])  
        enriched_rows.append({
            **row,
            "position_in_trace": seq_len,
            "day_of_week": last_event_ts.dayofweek,                 # 0 = Monday
            "time_of_day": last_event_ts.hour,                      # Hour of day
            "month": last_event_ts.month,                           # Month
            "week_of_year": last_event_ts.isocalendar().week,       # ISO week number
            "is_weekend": int(last_event_ts.weekday() >= 5),        # 1 = Saturday/Sunday
            "is_holiday_nl": int(last_event_ts.date() in nl_holidays),
            "time_since_case_start": total_duration,
            "time_since_prev_event": time_since_prev
        })  

    return pd.DataFrame(enriched_rows)


def expand_and_cast(x):
    return tf.expand_dims(tf.cast(x, tf.float32), axis=-1)


def build_lstm_model_advanced(
    num_activities,
    num_resources,
    context_dim,
    max_seq_len,
    num_classes,
    embedding_dim=64,
    lstm_units=128,
    dropout_rate=0.3
):
    # --- Inputs ---
    activity_input = Input(shape=(max_seq_len,), name="activity_input")
    duration_input = Input(shape=(max_seq_len,), name="duration_input", dtype="float32")
    resource_input = Input(shape=(max_seq_len,), name="resource_input")
    context_input = Input(shape=(context_dim,), name="context_input")

    # --- Embeddings ---
    activity_emb = Embedding(input_dim=num_activities, output_dim=embedding_dim)(activity_input)
    resource_emb = Embedding(input_dim=num_resources, output_dim=embedding_dim // 2)(resource_input)
    
    duration_expanded = Lambda(expand_and_cast, name="expand_and_cast")(duration_input)

    # --- Sequence Modeling ---
    seq_concat = Concatenate(axis=-1)([activity_emb, duration_expanded, resource_emb])

    x_seq = LSTM(lstm_units, return_sequences=True)(seq_concat)
    x_seq = LSTM(lstm_units // 2, return_sequences=True)(x_seq)

    # Pool across time
    x_seq = GlobalMaxPooling1D()(x_seq)
    x_seq = LayerNormalization()(x_seq)
    x_seq = Dropout(dropout_rate)(x_seq)

    # --- Context Fusion ---
    x = Concatenate()([x_seq, context_input])

    # --- Dense Block 1 ---
    x = Dense(64, kernel_initializer="he_normal")(x)
    x = LeakyReLU()(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # --- Dense Block 2 ---
    x = Dense(32, kernel_initializer="he_normal")(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)

    # --- Output ---
    output = Dense(num_classes, activation="softmax")(x)

    # --- Compile Model ---
    model = Model(
        inputs=[activity_input, duration_input, resource_input, context_input],
        outputs=output
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
    

def train_model_advanced(
    model,
    X_train,
    y_train,
    label_encoder=None,
    epochs=10,
    batch_size=128,
    use_class_weight=True
):
    """
    Trains an advanced LSTM model with early stopping and optional class weighting.
    """

    # --- Compute class weights ---
    class_weights = None
    if use_class_weight and label_encoder is not None:
        classes = label_encoder.classes_
        class_weight_values = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(len(classes)),
            y=y_train
        )
        class_weights = dict(enumerate(class_weight_values))

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=3,
        min_delta=1e-4,
        baseline=0.95,        # optional: stop as soon as ≥99.9 %
        restore_best_weights=True,
        verbose=1
    )

    # --- Train ---
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stop],
    )

    return model



def build_lstm_model_tunable(hp, fixed_params):
    model = build_lstm_model_advanced(
        num_activities=fixed_params['num_activities'],
        num_resources=fixed_params['num_resources'],
        context_dim=fixed_params['context_dim'],
        max_seq_len=fixed_params['max_seq_len'],
        num_classes=fixed_params['num_classes'],
        embedding_dim=hp.Choice('embedding_dim', [32, 64, 128]),
        lstm_units=hp.Choice('lstm_units', [64, 128, 256]),
        dropout_rate=hp.Float('dropout_rate', 0.2, 0.5, step=0.1)
    )

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def encode_inputs(df, context_keys):
    """
    Prepares padded activity/resource/duration sequences, context array, and encoded labels.

    Args:
        df (pd.DataFrame): input DataFrame with process and context data
        context_keys (list): column names to be used as context attributes

    Returns:
        X_acts_padded (np.ndarray)
        X_durs_padded (np.ndarray)
        X_res_padded (np.ndarray)
        X_context (np.ndarray)
        y (np.ndarray)
        max_seq_len (int)
        activity_encoder (LabelEncoder)
        resource_encoder (LabelEncoder)
        label_encoder (LabelEncoder)
    """
    # --- Sequence preparation ---
    (
        X_acts_padded,
        X_durs_padded,
        X_res_padded,
        activity_encoder,
        resource_encoder,
        label_encoder,
        y,
        max_seq_len
    ) = prepare_sequences_and_labels(df)

    # --- Context feature array ---
    X_context, _, _ = prepare_context_attributes(df, context_keys)

    return (
        X_acts_padded,
        X_durs_padded,
        X_res_padded,
        X_context,
        y,
        max_seq_len,
        activity_encoder,
        resource_encoder,
        label_encoder
    )

def tune_hyperparameters_for_dp(
    dp_name,
    df,
    context_keys,
    max_trials: int = 15,
    executions_per_trial: int = 1,
    target_val_acc: float = 0.80,          # <- Schwellwert konfigurierbar
):
    print(f"Tuning hyperparameters for {dp_name}...")

    # ── Pre-Processing ─────────────────────────────────────────────
    (
        X_acts, X_durs, X_res, X_context,
        y, max_seq_len, act_enc, res_enc, label_enc
    ) = encode_inputs(df, context_keys)

    split = train_test_split(
        X_acts, X_durs, X_res, X_context, y,
        test_size=0.2, stratify=y, random_state=42
    )
    (
        X_acts_train, X_acts_val,
        X_durs_train, X_durs_val,
        X_res_train, X_res_val,
        X_context_train, X_context_val,
        y_train, y_val
    ) = split

    fixed_params = {
        "num_activities": len(act_enc.classes_) + 1,
        "num_resources": len(res_enc.classes_) + 1,
        "context_dim":   X_context.shape[1],
        "max_seq_len":   max_seq_len,
        "num_classes":   len(label_enc.classes_),
    }

    # ── Keras-Tuner ────────────────────────────────────────────────
    tuner = kt.RandomSearch(
        hypermodel      = partial(build_lstm_model_tunable, fixed_params=fixed_params),
        objective       = "val_accuracy",
        max_trials      = max_trials,
        executions_per_trial = executions_per_trial,
        overwrite       = True,
        directory       = "hyperparam_tuning",
        project_name    = f"dp_{dp_name.replace(' ', '_')}",
    )

    # Early-Stopping innerhalb jedes Trials
    early_stop = EarlyStopping(
        monitor   = "val_accuracy",
        baseline  = target_val_acc,     # ← sobald ≥ 0.80: abbrechen
        patience  = 0,
        mode      = "max",
        verbose   = 1,
        restore_best_weights = True,
    )

    tuner.search(
        x = [X_acts_train, X_durs_train, X_res_train, X_context_train],
        y = y_train,
        validation_data = ([X_acts_val, X_durs_val, X_res_val, X_context_val], y_val),
        epochs = 10,                    # darf groß sein – Early-Stopping stoppt früher
        batch_size = 64,
        verbose = 1,
        callbacks = [early_stop],       # ← Callback übergeben
    )

    best_hp    = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    return best_hp, best_model, label_enc


def build_prediction_input_from_case_trace_simple(case_df: pd.DataFrame, max_history_len = 10) -> pd.DataFrame:
    """
    Builds a model-ready input sample from a full case trace, without padding.
    If the trace has fewer than `max_history_len` events, just use all available ones.
    """
    # Sort chronologically
    case_df = case_df.sort_values("time:timestamp")

    # Extract core features
    sequence = case_df["concept:name"].tolist()
    resources = case_df["org:resource"].tolist()
    timestamps = pd.to_datetime(case_df["time:timestamp"]).tolist()

    # Truncate to max_history_len from the end
    sequence = sequence[-max_history_len:]
    resources = resources[-max_history_len:]
    timestamps = timestamps[-max_history_len:]

    # Compute durations
    sequence_durations = [0.0]
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        sequence_durations.append(delta)

    # Extract case-level attributes
    case_attrs = {}
    attr_cols = [col for col in case_df.columns if col.startswith("case:")]
    for col in attr_cols:
        case_attrs[col] = case_df[col].iloc[0]

    # Build final row
    row = {
        "case_id": case_df["case:concept:name"].iloc[0],
        "sequence": sequence,
        "sequence_resources": resources,
        "sequence_durations": sequence_durations,
        **case_attrs
    }

    return pd.DataFrame([row])


def predict_next_activity_probs_simple(input_sample, decision_point, models, encoders, max_seq_len):
    """
    Predicts probabilities for each possible next activity at a given decision point.

    Args:
        input_sample (dict): {
            "sequence": [...],           # List of activity strings
            "durations": [...],          # List of floats
            "resources": [...],          # List of resource strings
            "context": [...]             # List of encoded context values (already preprocessed)
        }
    """
    # --- Retrieve model and encoders ---
    model = models[decision_point]
    enc = encoders[decision_point]
    act_enc = enc["activity"]
    res_enc = enc["resource"]
    label_enc = enc["label"]

    # --- Validate input ---
    unknown_acts = [a for a in input_sample["sequence"] if a not in act_enc.classes_]
    unknown_res = [r for r in input_sample["resources"] if r not in res_enc.classes_]
    if unknown_acts:
        raise ValueError(f"Unknown activity labels: {unknown_acts}")
    if unknown_res:
        raise ValueError(f"Unknown resource labels: {unknown_res}")

    # --- Encode sequence ---
    X_acts = act_enc.transform(input_sample["sequence"])
    X_res = res_enc.transform(input_sample["resources"])
    X_durs = input_sample["durations"]
    X_context = np.array([input_sample["context"]], dtype="float32")  # Already encoded

    # --- Pad sequences ---
    X_acts_padded = pad_sequences([X_acts], maxlen=max_seq_len, padding="pre")
    X_res_padded = pad_sequences([X_res], maxlen=max_seq_len, padding="pre")
    X_durs_padded = pad_sequences([X_durs], maxlen=max_seq_len, padding="pre", dtype="float32")

    # --- Predict ---
    pred_probs = model.predict([X_acts_padded, X_durs_padded, X_res_padded, X_context], verbose=0)
    return pred_probs[0]


def build_prediction_input_from_case_trace_advanced(case_df: pd.DataFrame, max_history_len: int) -> pd.DataFrame:
    """
    Converts a full case trace into a single model-ready input sample,
    enriched with temporal features for next-activity prediction.

    Parameters:
    - case_df: DataFrame containing all events of a case (excluding the event to be predicted!)
    - max_history_len: Max number of past events to include.

    Returns:
    - pd.DataFrame with one enriched sample row.
    """
    nl_holidays = holidays.Netherlands()

    # --- Sort the events chronologically
    case_df = case_df.sort_values("time:timestamp")

    # --- Extract core columns
    sequence = case_df["concept:name"].tolist()
    resources = case_df["org:resource"].tolist()
    timestamps = pd.to_datetime(case_df["time:timestamp"]).tolist()

    # --- Truncate to max_history_len
    sequence = sequence[-max_history_len:]
    resources = resources[-max_history_len:]
    timestamps = timestamps[-max_history_len:]

    # --- Compute durations
    sequence_durations = [0.0]
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        sequence_durations.append(delta)

    # --- Case-level attributes
    attr_cols = [col for col in case_df.columns if col.startswith("case:")]
    case_attrs = {col: case_df[col].iloc[0] for col in attr_cols}

    # --- Base row
    row = {
        "case_id": case_df["case:concept:name"].iloc[0],
        "sequence": sequence,
        "sequence_resources": resources,
        "sequence_durations": sequence_durations,
        **case_attrs
    }

    # --- Enrich with temporal features
    seq_len = len(sequence)
    durs = sequence_durations
    total_duration = sum(durs)
    time_since_prev = durs[-1] if len(durs) > 0 else 0.0

    last_event_ts = pd.to_datetime(timestamps[-1])

    row.update({
        "position_in_trace": seq_len,
        "day_of_week": last_event_ts.dayofweek,                 # 0 = Monday
        "time_of_day": last_event_ts.hour,                      # Hour of day
        "month": last_event_ts.month,                           # Month
        "week_of_year": last_event_ts.isocalendar().week,       # ISO week number
        "is_weekend": int(last_event_ts.weekday() >= 5),        # 1 = Saturday/Sunday
        "is_holiday_nl": int(last_event_ts.date() in nl_holidays),
        "time_since_case_start": total_duration,
        "time_since_prev_event": time_since_prev
    })

    return pd.DataFrame([row])


def predict_next_activity_probs_advanced(input_sample, decision_point, models_dict):
    """
    Predicts next activity probabilities for advanced models at a given decision point.

    Args:
        input_sample (dict): {
            "sequence": [...],           # List of activity strings
            "durations": [...],          # List of floats
            "resources": [...],          # List of resource strings
            "context": [val1, val2, ...] # List of encoded/scaled context features
        }
        decision_point (str): e.g. "Decision Point 5"
        models_dict (dict): decision_point -> {
            "model": keras model,
            "label_encoder": fitted LabelEncoder,
            "activity_encoder": fitted LabelEncoder,
            "resource_encoder": fitted LabelEncoder,
            "context_dim": int,
            "max_seq_len": int
        }

    Returns:
        np.ndarray: Probabilities for each class in the same order as label_encoder.classes_
    """

    # --- Retrieve model and encoders ---
    model_data = models_dict[decision_point]
    model = model_data["model"]
    act_enc = model_data["activity_encoder"]
    res_enc = model_data["resource_encoder"]
    label_enc = model_data["label_encoder"]
    max_seq_len = model_data["max_seq_len"]

    # --- Handle unknowns ---
    known_acts = set(act_enc.classes_)
    known_res = set(res_enc.classes_)

    sequence = [
        act if act in known_acts else "UNKNOWN"
        for act in input_sample["sequence"]
    ]
    resources = [
        res if res in known_res else "UNKNOWN"
        for res in input_sample["resources"]
    ]

    # --- Encode inputs ---
    X_acts = act_enc.transform(sequence)
    X_res = res_enc.transform(resources)
    X_durs = input_sample["durations"]
    X_context = np.array([input_sample["context"]], dtype="float32")

    # --- Pad sequences ---
    X_acts_padded = pad_sequences([X_acts], maxlen=max_seq_len, padding="pre")
    X_res_padded = pad_sequences([X_res], maxlen=max_seq_len, padding="pre")
    X_durs_padded = pad_sequences([X_durs], maxlen=max_seq_len, padding="pre", dtype="float32")

    # --- Predict ---
    pred_probs = model.predict(
        [X_acts_padded, X_durs_padded, X_res_padded, X_context],
        verbose=0
    )
    return pred_probs[0]  # shape: (num_classes,)


def extract_case_up_to_outgoing(df_log: pd.DataFrame, dp_name: str, dp_map: dict) -> pd.DataFrame:
    """
    For a decision point, return full case traces up to each outgoing activity
    where it was directly preceded by a valid incoming activity.

    Returns:
        pd.DataFrame: Combined DataFrame of all such case traces
    """
    incoming = set(dp_map[dp_name]["incoming"])
    outgoing = set(dp_map[dp_name]["outgoing"])

    df_sorted = df_log.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)

    extracted_traces = []

    for case_id, group in df_sorted.groupby("case:concept:name"):
        events = group["concept:name"].tolist()
        for i in range(1, len(events)):
            prev_event = events[i - 1]
            curr_event = events[i]

            if curr_event in outgoing and prev_event in incoming:
                # Return trace up to (and including) the outgoing activity
                full_trace = group.iloc[:i + 1].copy()
                extracted_traces.append(full_trace)
                break  # only take the first valid DP transition per case

    return pd.concat(extracted_traces, ignore_index=True)


def predict_probs_for_samples(
        samples,                       # DataFrame ODER Liste von DataFrames
        dp_name: str,
        models_dict: dict,
        bpmn_decision_point_map: dict  # <- NEU
):
    """
    Gibt eine DataFrame-Tabelle mit Klassen-Wahrscheinlichkeiten für alle übergebenen
    Samples am angegebenen Decision Point zurück.

    * Bei >1 ausgehenden Kanten:   LSTM-Modell wird verwendet.
    * Bei genau 1 ausgehender Kante: Probability = 1 für diese Aktivität.
    """

    # ─────────── Robustheit: Einzel-DF -> Liste ───────────
    if isinstance(samples, pd.DataFrame):
        samples = [samples]
    elif not isinstance(samples, (list, tuple)):
        raise TypeError("samples muss ein DataFrame oder eine Liste/Tuple davon sein")

    # ─────────── Modell & Encoder laden ───────────
    if dp_name not in models_dict:
        raise KeyError(f"Kein Modell für Decision Point »{dp_name}« gefunden")

    md         = models_dict[dp_name]
    model      = md["model"]
    act_enc    = md["activity_encoder"]
    res_enc    = md["resource_encoder"]
    lbl_enc    = md["label_encoder"]
    ctx_keys   = md["context_keys"]
    ctx_enc    = md["context_encoders"]
    max_len    = md["max_seq_len"]

    # ─────────── BPMN-Info prüfen ───────────
    dp_cfg = bpmn_decision_point_map.get(dp_name)
    if dp_cfg is None:
        raise KeyError(f"»{dp_name}« nicht im bpmn_decision_point_map gefunden")

    outgoing = dp_cfg["outgoing"]
    single_branch = len(outgoing) == 1
    if single_branch:
        fixed_act = outgoing[0]                       # einzige erlaubte Aktivität
        fixed_idx = np.where(lbl_enc.classes_ == fixed_act)[0]
        if len(fixed_idx) == 0:
            raise ValueError(
                f"Label-Encoder von {dp_name} kennt die Aktivität "
                f"»{fixed_act}« nicht (Klassen: {list(lbl_enc.classes_)})"
            )
        fixed_idx = fixed_idx[0]

    result_rows = []

    # ─────────── Samples iterieren ───────────
    for idx, sample_df in enumerate(samples, start=1):
        row = sample_df.iloc[0]

        # Kontext kodieren
        ctx_vec = []
        for col in ctx_keys:
            enc = ctx_enc[col]
            val = row[col]
            if isinstance(enc, LabelEncoder):
                if val not in enc.classes_:
                    raise ValueError(f"Unbekannter Wert »{val}« in Spalte »{col}«")
                ctx_vec.append(enc.transform([val])[0])
            else:  # StandardScaler
                ctx_vec.append(enc.transform([[float(val)]])[0][0])
        ctx_vec = np.asarray(ctx_vec, dtype="float32")[None, :]

        # Sequenzen kodieren + paddden
        def _left_pad(arr, fill):
            arr = np.asarray(arr).reshape(1, -1)
            pad_len = max_len - arr.shape[1]
            if pad_len < 0:
                arr = arr[:, -max_len:]
                pad_len = 0
            pad_block = np.full((1, pad_len), fill, dtype=arr.dtype)
            return np.hstack([pad_block, arr])

        if single_branch:
            # Prob-Vektor selbst bauen
            probs = np.zeros(len(lbl_enc.classes_), dtype="float32")
            probs[fixed_idx] = 1.0
        else:
            X_acts = _left_pad(act_enc.transform(row["sequence"]), 0)
            X_res  = _left_pad(res_enc.transform(row["sequence_resources"]), 0)
            X_durs = _left_pad(row["sequence_durations"], 0.0).astype("float32")

            probs = model.predict([X_acts, X_durs, X_res, ctx_vec], verbose=0)[0]

        # Ergebnis-Zeile
        result_rows.append(
            pd.Series(probs, index=lbl_enc.classes_, name=f"Sample {idx}")
        )

    return pd.DataFrame(result_rows)

def enrich_with_loop_features(df):
    """
    Adds loop-related features for each trace:
    - n_repeats_current_activity: How often did the current activity occur up to now in the trace?
    - n_unique_activities: How many different activities have occurred so far?
    - immediate_loop: Was the previous activity the same as the current one?
    - n_total_repeats: Total number of repeated activities so far.
    - longest_repeat_streak: Maximum consecutive occurrences of any activity.
    """
    enriched_rows = []

    for _, row in df.iterrows():
        seq = row["sequence"]
        if len(seq) == 0:
            n_repeats_current_activity = 0
            n_unique_activities = 0
            immediate_loop = 0
            n_total_repeats = 0
            longest_repeat_streak = 0
        else:
            # Up to current event (i.e., full sequence)
            current_act = seq[-1]
            n_repeats_current_activity = seq.count(current_act)
            n_unique_activities = len(set(seq))

            # Immediate loop: compare last and second-last
            immediate_loop = int(len(seq) > 1 and seq[-1] == seq[-2])

            # Total repeats in sequence (every repeat after first occurrence)
            counts = pd.Series(seq).value_counts()
            n_total_repeats = int(sum(counts - 1))

            # Longest streak of any repeated activity
            max_streak = 1
            curr_streak = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    curr_streak += 1
                    if curr_streak > max_streak:
                        max_streak = curr_streak
                else:
                    curr_streak = 1
            longest_repeat_streak = max_streak

        enriched_rows.append({
            **row,
            "n_repeats_current_activity": n_repeats_current_activity,
            "n_unique_activities": n_unique_activities,
            "immediate_loop": immediate_loop,
            "n_total_repeats": n_total_repeats,
            "longest_repeat_streak": longest_repeat_streak
        })

    return pd.DataFrame(enriched_rows)

