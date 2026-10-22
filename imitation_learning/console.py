"""Progress messages that also work with legacy Windows console encodings."""
import sys


def log(message, *, flush=False):
    """Preserve supported text and escape characters the output stream cannot encode."""
    encoding = getattr(sys.stdout, "encoding", None)
    if encoding:
        message = str(message).encode(encoding, errors="backslashreplace").decode(encoding)
    print(message, flush=flush)
