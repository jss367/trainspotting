"""Redact Discord credentials embedded in public training examples before storage."""

import base64
import re

# Discord bot tokens encode the numeric account id, followed by a timestamp and
# signature. Check the decoded id as well as the shape to avoid masking ordinary
# dotted strings. This is a targeted guard, not a general secret detector.
_DISCORD = re.compile(r"(?<![\w-])([A-Za-z0-9_-]{20,30})\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{25,110}(?![\w-])")
MARKER = "[REDACTED_DISCORD_TOKEN]"


def redact_credentials(text: str) -> str:
    """Mask token-shaped credentials; leave the surrounding example unchanged."""
    def replace(match: re.Match) -> str:
        encoded = match[1]
        try:
            account = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except ValueError:
            return match[0]
        return MARKER if account.isdigit() else match[0]

    return _DISCORD.sub(replace, text)
