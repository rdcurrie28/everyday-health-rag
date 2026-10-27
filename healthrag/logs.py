import json
import sys

SECRET_FIELDS = {"api_key", "password"}


def log_event(event: str, **fields) -> None:
    record = {"event": event}
    record.update({key: value for key, value in fields.items() if key not in SECRET_FIELDS})
    sys.stdout.write(json.dumps(record, default=str) + "\n")
    sys.stdout.flush()
