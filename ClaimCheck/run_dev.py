import json
import sys
import os
import hashlib
from tqdm import tqdm

# Add the factchecker directory to sys.path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
factchecker_dir = os.path.join(current_dir, 'factchecker')
sys.path.append(factchecker_dir)

try:
    from factchecker import factcheck
    from tools.quota import QuotaLimitReached, load_state
except ImportError as e:
    print(f"Error importing factchecker: {e}")
    sys.exit(1)


def _run_tag():
    tag = os.getenv("CLAIMCHECK_RUN_TAG", "").strip()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in tag)
    return safe.strip("-_")


def _state_path(json_path, num_records):
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    key = f"{os.path.abspath(json_path)}::{num_records}::{_run_tag()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    tag = _run_tag()
    suffix = f"_{tag}" if tag else ""
    return os.path.join(repo_root, "temp", f"run_state{suffix}_{digest}.json")


def _record_identifier(index, claim, date):
    digest = hashlib.sha1(f"{index}::{claim}::{date}".encode("utf-8")).hexdigest()[:12]
    tag = _run_tag()
    prefix = f"{tag}-" if tag else ""
    return f"{prefix}resume-{index:04d}-{digest}"


def _load_run_state(path, num_records):
    if not os.path.exists(path):
        return {
            "num_records": num_records,
            "records": {},
        }
    with open(path, "r") as f:
        state = json.load(f)
    state.setdefault("num_records", num_records)
    state.setdefault("records", {})
    return state


def _save_run_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _build_cached_counters(state):
    num_done = 0
    num_correct = 0
    for rec in state.get("records", {}).values():
        if rec.get("status") == "completed":
            num_done += 1
            if rec.get("correct"):
                num_correct += 1
    return num_done, num_correct

def main(json_path, num_records=5):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records from {json_path}")

    subset = data[:num_records]
    state_path = _state_path(json_path, num_records)
    run_state = _load_run_state(state_path, num_records)
    num_done, num_correct = _build_cached_counters(run_state)
    print(f"Resume state: {state_path}")
    print(f"Cached progress: completed={num_done}/{num_records}, correct={num_correct}")
    progress = tqdm(subset, total=len(subset), dynamic_ncols=True, unit="claim")

    for i, item in enumerate(progress):
        progress.set_description(f"claim {i+1}/{num_records}")
        claim = item['claim']
        date = item.get('claim_date', '')
        label = item.get('label', 'Unknown')
        record_key = str(i)
        identifier = _record_identifier(i, claim, date)
        record_state = run_state["records"].get(record_key, {})

        if record_state.get("status") == "completed":
            quota_state = load_state()
            accuracy = num_correct / num_done if num_done else 0.0
            progress.set_postfix(
                accuracy=f"{accuracy:.3f}",
                serper=f"{quota_state['count']}/{quota_state['soft_limit']}",
            )
            progress.write(
                f"Skipping cached record {i+1}/{num_records}: "
                f"{record_state.get('verdict', 'UNKNOWN')} | {record_state.get('report_path', '')}"
            )
            continue

        progress.write(f"\n--- Processing Record {i+1}/{num_records} ---")
        progress.write(f"Claim: {claim}")
        progress.write(f"Date: {date}")
        progress.write(f"Expected Label: {label}")
        metadata = {
            "speaker": item.get("speaker", ""),
            "original_claim_url": item.get("original_claim_url", ""),
            "reporting_source": item.get("reporting_source", ""),
            "fact_checking_article": item.get("fact_checking_article", ""),
            "claim_date": item.get("claim_date", ""),
            "location_ISO_code": item.get("location_ISO_code", ""),
        }

        run_state["records"][record_key] = {
            "status": "running",
            "identifier": identifier,
            "claim": claim,
            "date": date,
            "label": label,
            "report_path": os.path.join(current_dir, "reports", identifier, "report.md"),
        }
        _save_run_state(state_path, run_state)
        
        try:
            verdict, report_path = factcheck(claim, date, identifier=identifier, metadata=metadata)
            if verdict == label:
                num_correct += 1
            num_done += 1
            run_state["records"][record_key] = {
                "status": "completed",
                "identifier": identifier,
                "claim": claim,
                "date": date,
                "label": label,
                "verdict": verdict,
                "correct": verdict == label,
                "report_path": report_path,
            }
            _save_run_state(state_path, run_state)
            quota_state = load_state()
            accuracy = num_correct / num_done if num_done else 0.0
            progress.set_postfix(
                accuracy=f"{accuracy:.3f}",
                serper=f"{quota_state['count']}/{quota_state['soft_limit']}",
            )
            progress.write(f"Predicted Verdict: {verdict}")
            progress.write(f"Report saved to: {report_path}")
        except QuotaLimitReached as e:
            run_state["records"][record_key]["status"] = "paused_quota"
            run_state["records"][record_key]["error"] = str(e)
            _save_run_state(state_path, run_state)
            quota_state = load_state()
            progress.write(str(e))
            progress.write(
                f"Quota state: {quota_state['count']}/{quota_state['soft_limit']} searches used. Stopping run."
            )
            break
        except Exception as e:
            run_state["records"][record_key]["status"] = "failed"
            run_state["records"][record_key]["error"] = str(e)
            _save_run_state(state_path, run_state)
            progress.write(f"Error processing record {i+1}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            quota_state = load_state()
            progress.set_postfix(
                accuracy=f"{(num_correct / num_done) if num_done else 0.0:.3f}",
                serper=f"{quota_state['count']}/{quota_state['soft_limit']}",
            )

    progress.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_dev.py <json_path> [num_records]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    num_records = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    main(json_path, num_records)
