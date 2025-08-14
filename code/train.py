# training.py
# ─────────────────────────────────────────────────────────────────────────────
# Functions for: model building (simple/advanced), training, hyperparameter
# tuning, input encoding, and model component storage.
# ─────────────────────────────────────────────────────────────────────────────

from functools import partial
from typing import Dict, Tuple

import numpy as np
import keras_tuner as kt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, Concatenate, Lambda,
    GlobalMaxPooling1D, LayerNormalization, LeakyReLU
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import (
    prepare_sequences_and_labels,
    prepare_context_attributes
)


def _expand_and_cast(x):
    import tensorflow as tf
    return tf.expand_dims(tf.cast(x, tf.float32), axis=-1)


def build_lstm_model(
    num_activities: int,
    num_resources: int,
    context_dim: int,
    max_seq_len: int,
    num_classes: int
) -> Model:
    """
    Baseline/simple LSTM (one LSTM layer) with activity/resource embeddings and duration channel.
    """
    activity_input = Input(shape=(max_seq_len,), name="activity_input")
    duration_input = Input(shape=(max_seq_len,), name="duration_input", dtype="float32")
    resource_input = Input(shape=(max_seq_len,), name="resource_input")
    context_input  = Input(shape=(context_dim,), name="context_input")

    activity_emb = Embedding(input_dim=num_activities, output_dim=64, name="activity_embedding")(activity_input)
    resource_emb = Embedding(input_dim=num_resources, output_dim=16, name="resource_embedding")(resource_input)
    duration_expanded = Lambda(_expand_and_cast, name="expand_and_cast")(duration_input)

    seq_concat = Concatenate(axis=-1)([activity_emb, duration_expanded, resource_emb])
    x_seq = LSTM(units=128, return_sequences=False)(seq_concat)

    x = Concatenate()([x_seq, context_input])
    x = Dense(64, activation="relu")(x)

    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=[activity_input, duration_input, resource_input, context_input], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_lstm_model_advanced(
    num_activities: int,
    num_resources: int,
    context_dim: int,
    max_seq_len: int,
    num_classes: int,
    embedding_dim: int = 64,
    lstm_units: int = 128,
    dropout_rate: float = 0.3
) -> Model:
    """
    Advanced LSTM: two stacked LSTMs (return_sequences), temporal pooling,
    LN/Dropout, and two dense blocks with LeakyReLU.
    """
    activity_input = Input(shape=(max_seq_len,), name="activity_input")
    duration_input = Input(shape=(max_seq_len,), name="duration_input", dtype="float32")
    resource_input = Input(shape=(max_seq_len,), name="resource_input")
    context_input  = Input(shape=(context_dim,), name="context_input")

    activity_emb = Embedding(input_dim=num_activities, output_dim=embedding_dim)(activity_input)
    resource_emb = Embedding(input_dim=num_resources, output_dim=embedding_dim // 2)(resource_input)
    duration_expanded = Lambda(_expand_and_cast, name="expand_and_cast")(duration_input)

    seq_concat = Concatenate(axis=-1)([activity_emb, duration_expanded, resource_emb])

    x_seq = LSTM(lstm_units, return_sequences=True)(seq_concat)
    x_seq = LSTM(lstm_units // 2, return_sequences=True)(x_seq)

    x_seq = GlobalMaxPooling1D()(x_seq)
    x_seq = LayerNormalization()(x_seq)
    x_seq = Dropout(dropout_rate)(x_seq)

    x = Concatenate()([x_seq, context_input])

    x = Dense(64, kernel_initializer="he_normal")(x)
    x = LeakyReLU()(x)
    x = LayerNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(32, kernel_initializer="he_normal")(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(num_classes, activation="softmax")(x)
    model = Model(
        inputs=[activity_input, duration_input, resource_input, context_input],
        outputs=output
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model: Model, X_train, y_train, val_split: float = 0.1, epochs: int = 10, batch_size: int = 128):
    """
    Trains an LSTM with early stopping (val_accuracy).
    """
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=3,
        min_delta=1e-4,
        baseline=0.95,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=val_split,
        callbacks=[early_stop],
        epochs=epochs,
        batch_size=batch_size,
    )
    return history


def train_model_advanced(
    model: Model,
    X_train,
    y_train,
    label_encoder: LabelEncoder = None,
    val_split: float = 0.1,
    epochs: int = 10,
    batch_size: int = 128,
    use_class_weight: bool = True
) -> Model:
    """
    Trains the advanced LSTM with optional class weighting.
    """
    class_weights = None
    if use_class_weight and label_encoder is not None:
        classes = np.arange(len(label_encoder.classes_))
        class_weight_values = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weights = dict(enumerate(class_weight_values))

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=3,
        min_delta=1e-4,
        baseline=0.95,
        restore_best_weights=True,
        verbose=1
    )

    model.fit(
        x=X_train,
        y=y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[early_stop],
    )
    return model


def build_lstm_model_tunable(hp, fixed_params: Dict):
    """
    Keras-Tuner hypermodel wrapper for the advanced LSTM.
    """
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
    """
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
    dp_name: str,
    df,
    context_keys,
    max_trials: int = 15,
    executions_per_trial: int = 1,
    target_val_acc: float = 0.80,
):
    """
    Keras-Tuner random search for a single decision point dataset.
    """
    print(f"Tuning hyperparameters for {dp_name}...")

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

    tuner = kt.RandomSearch(
        hypermodel=partial(build_lstm_model_tunable, fixed_params=fixed_params),
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        overwrite=True,
        directory="hyperparam_tuning",
        project_name=f"dp_{dp_name.replace(' ', '_')}",
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        baseline=target_val_acc,
        patience=0,
        mode="max",
        verbose=1,
        restore_best_weights=True,
    )

    tuner.search(
        x=[X_acts_train, X_durs_train, X_res_train, X_context_train],
        y=y_train,
        validation_data=([X_acts_val, X_durs_val, X_res_val, X_context_val], y_val),
        epochs=10,
        batch_size=64,
        verbose=1,
        callbacks=[early_stop],
    )

    best_hp    = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]
    return best_hp, best_model, label_enc


def store_lstm_model(
    decision_point: str,
    model: Model,
    activity_encoder: LabelEncoder,
    resource_encoder: LabelEncoder,
    label_encoder: LabelEncoder,
    context_keys,
    context_encoders,
    max_seq_len: int,
    storage_dict: Dict
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