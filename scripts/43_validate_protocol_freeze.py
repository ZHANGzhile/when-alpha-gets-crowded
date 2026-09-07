"""Report whether the confirmatory protocol freeze is present and intact."""

from __future__ import annotations

import json
from pathlib import Path

from alpha_crowding.experiments import require_protocol_freeze


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "raw" / "manifests" / "protocol_freeze.json"


def main() -> int:
    try:
        freeze = require_protocol_freeze(MANIFEST, ROOT)
    except RuntimeError as exc:
        print(json.dumps({"status": "WAITING_PROTOCOL_FREEZE", "reason": str(exc)}))
        return 3
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "raw_data_cutoff": freeze.get("raw_data_cutoff"),
                "artifacts": len(freeze["artifacts"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
