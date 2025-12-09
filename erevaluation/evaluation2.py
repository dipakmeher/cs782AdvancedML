import argparse
import pandas as pd
import re
import time
import os
from datetime import datetime
from typing import List, Dict
from collections import defaultdict

# ============================================================
# Constants
# ============================================================
TUPLE_DELIM = "{tuple_delimiter}"
RECORD_DELIM = "{record_delimiter}"
COMPLETION_DELIM = "{completion_delimiter}"

# Define all entity types you care about globally
ALL_ENTITY_TYPES = [
    "person",
    "means_of_transportation",
    "means_of_communication",
    "routes",
    "location",
    "smuggled_items",
    "organization",
]


# ============================================================
# Logging helper
# ============================================================
def log(msg: str, logfile: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(logfile, "a") as f:
        f.write(line + "\n")


# ============================================================
# Normalize
# ============================================================
def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


# ============================================================
# Parser
# ============================================================

def parse_delimited_output(raw: str, allowed_types: List[str]):
    if not isinstance(raw, str):
        return {"entities": [], "relations": []}

    parts = raw.split(RECORD_DELIM)
    entities, relations = [], []

    # normalize allowed types to lowercase
    allowed_lower = [t.lower().strip() for t in allowed_types]

    for rec in parts:
        rec = rec.strip()
        if not rec or COMPLETION_DELIM in rec:
            break

        m = re.match(r'^\(\s*"(entity|relationship)"\s*(.*)\)\s*$', rec)
        if not m:
            continue

        rec_type, rest = m.groups()
        if rest.startswith(TUPLE_DELIM):
            rest = rest[len(TUPLE_DELIM):]

        fields = rest.split(TUPLE_DELIM)

        if rec_type == "entity" and len(fields) >= 2:

            # FIX 1: normalize entity name
            name = normalize_text(fields[0])

            # FIX 2: normalize entity type
            etype = normalize_text(fields[1]).lower()

            if etype in allowed_lower:
                entities.append((name, etype))

        elif rec_type == "relationship" and len(fields) >= 2:
            head = normalize_text(fields[0])
            tail = normalize_text(fields[1])
            relations.append((head, tail))

    return {"entities": entities, "relations": relations}

# ============================================================
# PRF1
# ============================================================
def compute_prf1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ============================================================
# Entity Scoring (Global + Per-Type)
# ============================================================
def score_entities(gold_entities, pred_entities):
    """
    gold_entities: list of (name, type)
    pred_entities: list of (name, type)

    Returns:
        overall_counts: {"tp": int, "fp": int, "fn": int}
        per_type_counts: {etype: {"tp": int, "fp": int, "fn": int}}
    """
    # Initialize per-type counts for all known types
    per_type_counts: Dict[str, Dict[str, int]] = {
        etype: {"tp": 0, "fp": 0, "fn": 0} for etype in ALL_ENTITY_TYPES
    }

    gold_used = [False] * len(gold_entities)
    tp = fp = fn = 0

    # ---- STEP 1: Process predictions (TP / FP) ----
    for p_name, p_type in pred_entities:
        match_index = None
        # Find an unused matching gold entity
        for i, (g_name, g_type) in enumerate(gold_entities):
            if (not gold_used[i]) and (p_name == g_name) and (p_type == g_type):
                match_index = i
                break

        if match_index is not None:
            # True Positive
            tp += 1
            gold_used[match_index] = True
            if p_type not in per_type_counts:
                per_type_counts[p_type] = {"tp": 0, "fp": 0, "fn": 0}
            per_type_counts[p_type]["tp"] += 1
        else:
            # False Positive
            fp += 1
            if p_type not in per_type_counts:
                per_type_counts[p_type] = {"tp": 0, "fp": 0, "fn": 0}
            per_type_counts[p_type]["fp"] += 1

    # ---- STEP 2: Remaining unused gold entities are FNs ----
    for used_flag, (g_name, g_type) in zip(gold_used, gold_entities):
        if not used_flag:
            fn += 1
            if g_type not in per_type_counts:
                per_type_counts[g_type] = {"tp": 0, "fp": 0, "fn": 0}
            per_type_counts[g_type]["fn"] += 1

    overall_counts = {"tp": tp, "fp": fp, "fn": fn}
    return overall_counts, per_type_counts


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_dir", required=True,
                        help="Base directory where timestamped run folder will be created.")
    args = parser.parse_args()

    # ============================================================
    # Create timestamped folder
    # ============================================================
    timestamp = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_folder, exist_ok=True)

    output_results = os.path.join(run_folder, "row_eval.csv")
    output_summary = os.path.join(run_folder, "summary.txt")
    detailed_csv = os.path.join(run_folder, "detailed_pairs.csv")
    logfile = os.path.join(run_folder, "evaluation_log.txt")

    log("===== EVALUATION STARTED =====", logfile)
    log(f"CSV Path = {args.csv_path}", logfile)
    log(f"Run Folder = {run_folder}", logfile)

    overall_start = time.time()

    df = pd.read_csv(args.csv_path)

    if "Sr.No." in df.columns:
        df["Row_ID"] = df["Sr.No."].astype(str)
    else:
        df["Row_ID"] = df.index.astype(str)

    # Global overall counts for entities & relations
    global_tp_e = global_fp_e = global_fn_e = 0
    global_tp_r = global_fp_r = global_fn_r = 0

    # Global per-type entity counts
    global_per_type_entities: Dict[str, Dict[str, int]] = {
        etype: {"tp": 0, "fp": 0, "fn": 0} for etype in ALL_ENTITY_TYPES
    }

    row_records = []
    detailed_rows = []

    # ============================================================
    # LOOP
    # ============================================================
    for idx, row in df.iterrows():
        row_id = row["Row_ID"]
        t0 = time.time()

        log(f"Processing Row_ID={row_id} ...", logfile)

        # Entity types allowed for this row (from CSV)
        allowed_types = [t.strip() for t in str(row["Entity_Types"]).split(",") if t.strip()]

        gold = parse_delimited_output(row["Output"], allowed_types)
        pred = parse_delimited_output(row["LLM_Output"], allowed_types)

        gold_entities = gold["entities"]
        pred_entities = pred["entities"]
        gold_rel = gold["relations"]
        pred_rel = pred["relations"]

        log(f"  GOLD: {len(gold_entities)} entities, {len(gold_rel)} relations", logfile)
        log(f"  PRED: {len(pred_entities)} entities, {len(pred_rel)} relations", logfile)

        # ========================================================
        # ENTITY SCORING (GLOBAL + PER-TYPE)
        # ========================================================
        overall_e, per_type_e_row = score_entities(gold_entities, pred_entities)

        tp_e = overall_e["tp"]
        fp_e = overall_e["fp"]
        fn_e = overall_e["fn"]

        # Update global overall entity counters
        global_tp_e += tp_e
        global_fp_e += fp_e
        global_fn_e += fn_e

        # Update global per-type entity counters
        for etype, counts in per_type_e_row.items():
            if etype not in global_per_type_entities:
                global_per_type_entities[etype] = {"tp": 0, "fp": 0, "fn": 0}
            global_per_type_entities[etype]["tp"] += counts["tp"]
            global_per_type_entities[etype]["fp"] += counts["fp"]
            global_per_type_entities[etype]["fn"] += counts["fn"]

        # Row-level entity PRF
        p_e, r_e, f1_e = compute_prf1(tp_e, fp_e, fn_e)

        # ========================================================
        # RELATION SCORING (same as your original logic)
        # ========================================================
        gold_rel_used = [False] * len(gold_rel)
        tp_r = fp_r = fn_r = 0

        for pr in pred_rel:
            if pr in gold_rel:
                m = gold_rel.index(pr)
                if not gold_rel_used[m]:
                    tp_r += 1
                    gold_rel_used[m] = True
                else:
                    fp_r += 1
            else:
                fp_r += 1

        fn_r = gold_rel_used.count(False)

        # Update global relation counters
        global_tp_r += tp_r
        global_fp_r += fp_r
        global_fn_r += fn_r

        # Row-level relation PRF
        p_r, r_r, f1_r = compute_prf1(tp_r, fp_r, fn_r)

        # ========================================================
        # BUILD ROW RECORD (including per-type metrics)
        # ========================================================
        row_dict = {
            "Row_ID": row_id,
            "TP_entities": tp_e,
            "FP_entities": fp_e,
            "FN_entities": fn_e,
            "Entity_Precision": p_e,
            "Entity_Recall": r_e,
            "Entity_F1": f1_e,
            "TP_rel": tp_r,
            "FP_rel": fp_r,
            "FN_rel": fn_r,
            "Rel_Precision": p_r,
            "Rel_Recall": r_r,
            "Rel_F1": f1_r,
        }

        # Add per-type entity metrics for this row
        for etype in ALL_ENTITY_TYPES:
            counts = per_type_e_row.get(etype, {"tp": 0, "fp": 0, "fn": 0})
            tp_t = counts["tp"]
            fp_t = counts["fp"]
            fn_t = counts["fn"]
            p_t, r_t, f1_t = compute_prf1(tp_t, fp_t, fn_t)

            prefix = etype  # e.g., "person"
            row_dict[f"{prefix}_TP"] = tp_t
            row_dict[f"{prefix}_FP"] = fp_t
            row_dict[f"{prefix}_FN"] = fn_t
            row_dict[f"{prefix}_F1"] = f1_t

        row_records.append(row_dict)

        # ========================================================
        # Detailed pairs (unchanged logic)
        # ========================================================
        tp_entity_list = [str(pe) for pe in pred_entities if pe in gold_entities]
        fp_entity_list = [str(pe) for pe in pred_entities if pe not in gold_entities]
        fn_entity_list = [str(ge) for ge in gold_entities if ge not in pred_entities]

        tp_rel_list = [str(pr) for pr in pred_rel if pr in gold_rel]
        fp_rel_list = [str(pr) for pr in pred_rel if pr not in gold_rel]
        fn_rel_list = [str(gr) for gr in gold_rel if gr not in pred_rel]

        detailed_rows.append({
            "Row_ID": row_id,
            "TP_entities": len(tp_entity_list),
            "FP_entities": len(fp_entity_list),
            "FN_entities": len(fn_entity_list),
            "TP_rel": len(tp_rel_list),
            "FP_rel": len(fp_rel_list),
            "FN_rel": len(fn_rel_list),
            "TP_entity_pairs": "; ".join(tp_entity_list),
            "FP_entity_pairs": "; ".join(fp_entity_list),
            "FN_entity_pairs": "; ".join(fn_entity_list),
            "TP_relation_pairs": "; ".join(tp_rel_list),
            "FP_relation_pairs": "; ".join(fp_rel_list),
            "FN_relation_pairs": "; ".join(fn_rel_list),
        })

        dt = time.time() - t0
        log(f"Finished Row_ID={row_id} in {dt:.3f} sec", logfile)

    # ============================================================
    # Save files into timestamped folder
    # ============================================================
    pd.DataFrame(row_records).to_csv(output_results, index=False)
    pd.DataFrame(detailed_rows).to_csv(detailed_csv, index=False)

    # ============================================================
    # GLOBAL SUMMARY
    # ============================================================
    p_e, r_e, f1_e = compute_prf1(global_tp_e, global_fp_e, global_fn_e)
    p_r, r_r, f1_r = compute_prf1(global_tp_r, global_fp_r, global_fn_r)

    with open(output_summary, "w") as f:
        f.write("===== GLOBAL SUMMARY =====\n")
        f.write(f"Entities: TP={global_tp_e}, FP={global_fp_e}, FN={global_fn_e}\n")
        f.write(f"P={p_e:.4f}, R={r_e:.4f}, F1={f1_e:.4f}\n\n")
        f.write(f"Relations: TP={global_tp_r}, FP={global_fp_r}, FN={global_fn_r}\n")
        f.write(f"P={p_r:.4f}, R={r_r:.4f}, F1={f1_r:.4f}\n")

        # --------------------------------------------------------
        # PER-TYPE ENTITY SUMMARY
        # --------------------------------------------------------
        f.write("\n===== PER-TYPE ENTITY SUMMARY =====\n")
        for etype in ALL_ENTITY_TYPES:
            g_counts = global_per_type_entities.get(etype, {"tp": 0, "fp": 0, "fn": 0})
            tp_t = g_counts["tp"]
            fp_t = g_counts["fp"]
            fn_t = g_counts["fn"]
            p_t, r_t, f1_t = compute_prf1(tp_t, fp_t, fn_t)

            f.write(f"\n{etype.upper()}:\n")
            f.write(f"  TP={tp_t}, FP={fp_t}, FN={fn_t}\n")
            f.write(f"  P={p_t:.4f}, R={r_t:.4f}, F1={f1_t:.4f}\n")

    total = time.time() - overall_start
    log(f"===== EVALUATION COMPLETE in {total:.2f} sec =====", logfile)
    log(f"Saved results in: {run_folder}", logfile)


if __name__ == "__main__":
    main()
