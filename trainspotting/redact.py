"""Redact Discord credentials embedded in public training examples before storage."""

import base64
import re

# Discord bot tokens encode the numeric account id, followed by a timestamp and
# signature. Check the decoded id as well as the shape to avoid masking ordinary
# dotted strings. This is a targeted guard, not a general secret detector.
#
# Callers redact serialized JSON, where a token on its own line reads `\n<token>`
# and one after a non-ASCII letter reads `\u00e9<token>`. The escape ends in a
# word character, so a JSON escape also counts as a boundary. A backslash alone
# does not: otherwise the match starts at the escape's `n`, fails to decode, and
# consumes the real token with it.
_BOUNDARY = r"(?:(?<![\w\\-])|(?<=\\[nrtbf\\])|(?<=\\u[0-9a-fA-F]{4}))"
_DISCORD = re.compile(_BOUNDARY + r"([A-Za-z0-9_-]{20,30})\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{25,110}(?![\w-])")
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
