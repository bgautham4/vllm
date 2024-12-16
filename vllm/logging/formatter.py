import datetime as dt
import json
import logging
from typing import override

LOG_RECORDS_BUILTIN_ATTRS = {}


class MyJSONFormatter(logging.Formatter):
    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created,
                                                   tz=dt.timezone.utc).isoformat()
        }
        message = {key: msg_val
                   if (msg_val := always_fields.pop(val, None)) is not None
                   else getattr(record, val)
                   for key, val in self.fmt_keys.items()
                   }

        message.update(always_fields)

        for key, val in record.__dict__.items():
            if key not in LOG_RECORDS_BUILTIN_ATTRS:
                message[key] = val

        return message


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None, style="%"):
        logging.Formatter.__init__(self, fmt, datefmt, style)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg
