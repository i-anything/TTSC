"""Check the submitted catalog, model, index, and offline agent entry point."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from scripts.prepare_submission import prepare_catalog


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    catalog = prepare_catalog(ROOT / "data/catalog.jsonl.gz", ROOT / "data/catalog.jsonl")
    from preprocessing.embeddings import verify_embedding_artifacts

    artifacts = verify_embedding_artifacts(ROOT / "assets/search-index-bge-small-en-v1.5-v2")
    # This command owns its process. The audit hook is deliberately scoped to
    # initialization and reset/respond; it is not installed in the agent API.
    active = True
    denied: list[str] = []

    def audit(event: str, arguments: tuple) -> None:
        if not active:
            return
        blocked = event.startswith(("socket.", "subprocess.")) or event in {"os.system", "os.posix_spawn"}
        if event == "open" and arguments and isinstance(arguments[0], str):
            path = Path(arguments[0])
            blocked |= path.suffix == ".jsonl" and path.resolve() != catalog.resolve()
        if blocked:
            denied.append(event)
            raise RuntimeError(f"offline submission check denied {event}")

    sys.addaudithook(audit)
    from agent import Agent

    started = time.perf_counter()
    agent = Agent(catalog)
    backend = agent.retrieval_backend
    if not (backend.bm25_available and backend.dense_available and backend.protocol_evidence_available):
        raise RuntimeError("submission did not initialize every required retrieval backend")
    # Exercise a broad query so the dense encoder runs even when narrow exact
    # evidence could otherwise skip it. Reset replay must be byte-identical.
    message = "I am shopping for comfortable shoes for long walks."
    responses = []
    traces = []
    for _ in range(2):
        agent.reset("submission-check", {"preference_tags": []})
        responses.append(agent.respond("submission-check", message, 1, 10))
        traces.append(agent.last_action_trace("submission-check"))
    active = False
    if denied:
        raise RuntimeError(f"runtime attempted forbidden access: {denied}")
    if responses[0] != responses[1]:
        raise RuntimeError("reset/respond replay was not deterministic")
    response = responses[0]
    identifiers = [item["parent_asin"] for item in response["recommendations"]]
    if not identifiers or len(identifiers) > 10 or len(set(identifiers)) != len(identifiers):
        raise RuntimeError("invalid recommendation slate")
    if response["usage"] != {"prompt_tokens": 0, "completion_tokens": 0}:
        raise RuntimeError("unexpected model token usage")
    if not any(trace["dense_status"] == "ok" for trace in traces):
        raise RuntimeError("dense query smoke check failed")
    print(json.dumps({
        "ready": True,
        "catalog_rows": artifacts["catalog"]["rows"],
        "offline": True,
        "deterministic_replay": True,
        "backends": ["bm25", "bge_int8_onnx", "catalog_evidence"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
