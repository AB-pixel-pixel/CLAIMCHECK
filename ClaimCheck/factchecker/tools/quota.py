import json
import os
import fcntl
from datetime import datetime


class QuotaLimitReached(RuntimeError):
    pass


def _state_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
    default_path = os.path.join(repo_root, "temp", "serper_quota.json")
    return os.getenv("SERPER_QUOTA_STATE", default_path)


def _default_state():
    soft_limit = int(os.getenv("SERPER_SOFT_LIMIT", "2450"))
    hard_total = int(os.getenv("SERPER_TOTAL_LIMIT", "2500"))
    return {
        "count": 0,
        "soft_limit": soft_limit,
        "hard_total": hard_total,
        "queries": [],
        "last_updated": None,
    }


def load_state():
    path = _state_path()
    if not os.path.exists(path):
        return _default_state()

    with open(path, "r") as f:
        state = json.load(f)

    default = _default_state()
    for key, value in default.items():
        state.setdefault(key, value)
    return state


def save_state(state):
    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def reset_state():
    state = _default_state()
    state["last_updated"] = datetime.utcnow().isoformat() + "Z"
    save_state(state)
    return state


def register_search(query):
    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        save_state(_default_state())

    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        content = f.read().strip()
        state = json.loads(content) if content else _default_state()
        default = _default_state()
        for key, value in default.items():
            state.setdefault(key, value)

        next_count = state["count"] + 1
        if next_count > state["soft_limit"]:
            fcntl.flock(f, fcntl.LOCK_UN)
            raise QuotaLimitReached(
                f"Serper quota pause threshold reached: {state['count']} / {state['soft_limit']}"
            )

        state["count"] = next_count
        state["queries"].append(
            {
                "index": next_count,
                "query": query,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        )
        state["last_updated"] = state["queries"][-1]["timestamp"]
        f.seek(0)
        f.truncate()
        json.dump(state, f, indent=2)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return state


def remaining_until_pause():
    state = load_state()
    return max(state["soft_limit"] - state["count"], 0)
