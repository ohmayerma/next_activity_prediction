# preprocess.py
# ─────────────────────────────────────────────────────────────────────────────
# Functions for: BPMN DP extraction, dataset construction, feature engineering,
# sequence/context encoding helpers, and case-to-sample utilities.
# ─────────────────────────────────────────────────────────────────────────────

import re
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import holidays

from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences


def extract_bpmn_decision_point_map(bpmn_model) -> Dict[str, Dict[str, List[str]]]:
    """
    Extracts a unified map of BPMN decision points from a PM4Py BPMN model.

    Returns:
        dict of { dp_name: { 'incoming': [act1, act2, …], 'outgoing': [actA, actB, …] } }
    """
    def _dp_number(label):
        m = re.search(r"DP\s*(\d+)", label)
        return int(m.group(1)) if m else float('inf')

    raw_out = defaultdict(list)
    for flow in bpmn_model.get_flows():
        src, tgt = flow.source, flow.target
        if "Gateway" in src.__class__.__name__ and getattr(tgt, "name", None):
            raw_out[src.name or src.id].append(tgt.name)

    raw_in = defaultdict(list)
    for flow in bpmn_model.get_flows():
        src, tgt = flow.source, flow.target
        if getattr(tgt, "name", None) and "Gateway" in tgt.__class__.__name__:
            raw_in[tgt.name].append(src.name or src.id)

    sorted_out = OrderedDict(sorted(raw_out.items(), key=lambda kv: _dp_number(kv[0])))
    sorted_in  = OrderedDict(sorted(raw_in .items(), key=lambda kv: _dp_number(kv[0])))

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

    out_clean, in_clean = {}, {}
    for dp, targets in sorted_out.items():
        flat = _resolve(sorted_out, dp)
        seen = set()
        out_clean[dp] = [x for x in flat if x not in seen and not seen.add(x)]

    for dp, sources in sorted_in.items():
        flat = _resolve(sorted_in, dp)
        seen = set()
        in_clean[dp] = [x for x in flat if x not in seen and not seen.add(x)]

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
    df_log: pd.DataFrame,
    bpmn_decision_point_map: Dict[str, Dict[str, List[str]]],
    max_history_len: int = 10,
    min_sequence_count: int = 20,
    min_class_count: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Builds per-decision-point datasets with sequences (acts, resources, durations),
    timestamps, and basic case attributes (LoanGoal, ApplicationType, RequestedAmount).
    Filters out rare sequences and classes.
    """
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

    case_columns = ['case:LoanGoal', 'case:ApplicationType', 'case:RequestedAmount']
    case_attr_df = df_log.drop_duplicates('case:concept:name')[
        ['case:concept:name'] + case_columns
    ].set_index('case:concept:name')

    dp_to_training_rows = defaultdict(list)
    df_sorted = df_log.sort_values(['case:concept:name', 'time:timestamp'])

    for case_id, group in df_sorted.groupby('case:concept:name'):
        events = group['concept:name'].tolist()
        resources = group['org:resource'].tolist()
        timestamps = group['time:timestamp'].tolist()

        for i in range(len(events) - 1):
            prev, nxt = events[i], events[i + 1]
            dp = clean_transition_to_dp.get((prev, nxt))
            if not dp:
                continue

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

    dp_to_df_raw = {dp: pd.DataFrame(rows) for dp, rows in dp_to_training_rows.items() if rows}

    dp_to_df_filtered = {}
    for dp, df in dp_to_df_raw.items():
        df['sequence_str'] = df['sequence'].apply(lambda x: tuple(x))
        sequence_counts = df['sequence_str'].value_counts()
        valid_sequences = set(sequence_counts[sequence_counts > min_sequence_count].index)
        df = df[df['sequence_str'].apply(lambda x: x in valid_sequences)].drop(columns=['sequence_str'])

        class_counts = df['label'].value_counts()
        valid_labels = class_counts[class_counts >= min_class_count].index
        df = df[df['label'].isin(valid_labels)]

        if not df.empty and df['label'].nunique() >= 2:
            dp_to_df_filtered[dp] = df

    return dp_to_df_filtered


def prepare_sequences_and_labels(df: pd.DataFrame):
    """
    Prepares activity, duration, and resource sequences along with label encoding and padding.

    Returns:
        X_acts_padded, X_durs_padded, X_res_padded, activity_encoder,
        resource_encoder, label_encoder, y, max_seq_len
    """
    activity_seqs = df["sequence"]
    duration_seqs = df["sequence_durations"]
    resource_seqs = df["sequence_resources"]
    labels = df["label"]

    all_activities = [act for seq in activity_seqs for act in seq]
    activity_encoder = LabelEncoder()
    activity_encoder.fit(all_activities)
    X_acts = [activity_encoder.transform(seq) for seq in activity_seqs]

    all_resources = [res for seq in resource_seqs for res in seq]
    resource_encoder = LabelEncoder()
    resource_encoder.fit(all_resources)
    X_res = [resource_encoder.transform(seq) for seq in resource_seqs]

    max_seq_len = max(len(seq) for seq in X_acts)

    X_acts_padded = pad_sequences(X_acts, maxlen=max_seq_len, padding="pre").astype("int32")
    X_durs_padded = pad_sequences(duration_seqs, maxlen=max_seq_len, padding="pre", dtype="float32").astype("float32")
    X_res_padded = pad_sequences(X_res, maxlen=max_seq_len, padding="pre").astype("int32")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels).astype("int32")

    return X_acts_padded, X_durs_padded, X_res_padded, activity_encoder, resource_encoder, label_encoder, y, max_seq_len


def prepare_context_attributes(df: pd.DataFrame, context_keys: List[str]):
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
        if col_data.dtype == "object" or col_data.dtype.name == "string":
            le = LabelEncoder()
            X_context[col] = le.fit_transform(col_data.astype(str))
            context_encoders[col] = le
        else:
            scaler = StandardScaler()
            X_context[col] = scaler.fit_transform(col_data.values.reshape(-1, 1))
            context_encoders[col] = scaler

    X_context_array = X_context.to_numpy().astype("float32")
    context_dim = X_context_array.shape[1]
    return X_context_array, context_dim, context_encoders


def enrich_with_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    nl_holidays = holidays.Netherlands()
    enriched_rows = []

    for _, row in df.iterrows():
        seq_len = len(row["sequence"])
        durs = row["sequence_durations"]
        total_duration = sum(durs)
        time_since_prev = durs[-1] if len(durs) > 0 else 0.0

        last_event_ts = pd.to_datetime(row["sequence_timestamps"][-1])
        enriched_rows.append({
            **row,
            "position_in_trace": seq_len,
            "day_of_week": last_event_ts.dayofweek,
            "time_of_day": last_event_ts.hour,
            "month": last_event_ts.month,
            "week_of_year": last_event_ts.isocalendar().week,
            "is_weekend": int(last_event_ts.weekday() >= 5),
            "is_holiday_nl": int(last_event_ts.date() in nl_holidays),
            "time_since_case_start": total_duration,
            "time_since_prev_event": time_since_prev
        })

    return pd.DataFrame(enriched_rows)


def enrich_with_loop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds loop-related features for each trace.
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
            current_act = seq[-1]
            n_repeats_current_activity = seq.count(current_act)
            n_unique_activities = len(set(seq))
            immediate_loop = int(len(seq) > 1 and seq[-1] == seq[-2])

            counts = pd.Series(seq).value_counts()
            n_total_repeats = int(sum(counts - 1))

            max_streak, curr_streak = 1, 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    curr_streak += 1
                    max_streak = max(max_streak, curr_streak)
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


def build_prediction_input_from_case_trace_simple(
    case_df: pd.DataFrame,
    max_history_len: int = 10
) -> pd.DataFrame:
    """
    Builds a model-ready input sample from a full case trace, without padding.
    """
    case_df = case_df.sort_values("time:timestamp")
    sequence  = case_df["concept:name"].tolist()[-max_history_len:]
    resources = case_df["org:resource"].tolist()[-max_history_len:]
    timestamps = pd.to_datetime(case_df["time:timestamp"]).tolist()[-max_history_len:]

    sequence_durations = [0.0]
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        sequence_durations.append(delta)

    case_attrs = {}
    attr_cols = [col for col in case_df.columns if col.startswith("case:")]
    for col in attr_cols:
        case_attrs[col] = case_df[col].iloc[0]

    row = {
        "case_id": case_df["case:concept:name"].iloc[0],
        "sequence": sequence,
        "sequence_resources": resources,
        "sequence_durations": sequence_durations,
        **case_attrs
    }
    return pd.DataFrame([row])


def build_prediction_input_from_case_trace_advanced(
    case_df: pd.DataFrame,
    max_history_len: int
) -> pd.DataFrame:
    """
    Converts a full case trace into a single model-ready input sample,
    enriched with temporal features for next-activity prediction.
    """
    nl_holidays = holidays.Netherlands()
    case_df = case_df.sort_values("time:timestamp")

    sequence  = case_df["concept:name"].tolist()[-max_history_len:]
    resources = case_df["org:resource"].tolist()[-max_history_len:]
    timestamps = pd.to_datetime(case_df["time:timestamp"]).tolist()[-max_history_len:]

    sequence_durations = [0.0]
    for i in range(1, len(timestamps)):
        delta = (timestamps[i] - timestamps[i - 1]).total_seconds()
        sequence_durations.append(delta)

    attr_cols = [col for col in case_df.columns if col.startswith("case:")]
    case_attrs = {col: case_df[col].iloc[0] for col in attr_cols}

    row = {
        "case_id": case_df["case:concept:name"].iloc[0],
        "sequence": sequence,
        "sequence_resources": resources,
        "sequence_durations": sequence_durations,
        **case_attrs
    }

    seq_len = len(sequence)
    durs = sequence_durations
    total_duration = sum(durs)
    time_since_prev = durs[-1] if len(durs) > 0 else 0.0
    last_event_ts = pd.to_datetime(timestamps[-1])

    row.update({
        "position_in_trace": seq_len,
        "day_of_week": last_event_ts.dayofweek,
        "time_of_day": last_event_ts.hour,
        "month": last_event_ts.month,
        "week_of_year": last_event_ts.isocalendar().week,
        "is_weekend": int(last_event_ts.weekday() >= 5),
        "is_holiday_nl": int(last_event_ts.date() in nl_holidays),
        "time_since_case_start": total_duration,
        "time_since_prev_event": time_since_prev
    })
    return pd.DataFrame([row])


def extract_case_up_to_outgoing(
    df_log: pd.DataFrame,
    dp_name: str,
    dp_map: Dict[str, Dict[str, List[str]]]
) -> pd.DataFrame:
    """
    For a decision point, return full case traces up to each outgoing activity
    where it was directly preceded by a valid incoming activity.
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
                full_trace = group.iloc[:i + 1].copy()
                extracted_traces.append(full_trace)
                break  # only take the first valid DP transition per case

    return pd.concat(extracted_traces, ignore_index=True)