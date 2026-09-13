"""Access control for the kiosk.

The kiosk streams faces to a recogniser that answers with names. Exposed
without a login, that is an identity oracle: anyone holding the URL could hold
up a photograph of a stranger and learn whether they are enrolled and what they
are called - or simply enrol, delete, and read the check-in history. So every
page, API call and websocket sits behind a login, except the login page itself,
a health check, and static assets, none of which reveal anything.

One shared password rather than user accounts. A kiosk is a single device run
by whoever set it up; accounts would add a user table, password hashing and a
reset flow to protect exactly one secret.

Sessions are HMAC-signed expiry timestamps held in a cookie, using only the
standard library and storing nothing server-side. The signing key is derived
from the password, so changing the password signs every device out.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from urllib.parse import urlsplit

COOKIE_NAME = "presence_session"


class Auth:
    def __init__(self, password: str | None, secret: str | None, max_age_s: int) -> None:
        self.password = password or None
        self.max_age_s = max_age_s
        # Without a configured secret, a fresh one per process. Sessions then
        # end whenever the server restarts - an inconvenience, not a hole.
        base = (secret or secrets.token_hex(32)).encode()
        self._key = hmac.new(base, (self.password or "").encode(), hashlib.sha256).digest()

    @property
    def configured(self) -> bool:
        return self.password is not None

    def check_password(self, candidate: str) -> bool:
        if not self.configured:
            return False
        # Comparing fixed-length digests keeps timing from revealing either the
        # content or the length of the password.
        return hmac.compare_digest(
            hashlib.sha256(candidate.encode()).digest(),
            hashlib.sha256(self.password.encode()).digest(),
        )

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode(), hashlib.sha256).hexdigest()

    def issue(self, now: float | None = None) -> str:
        expires = int((time.time() if now is None else now) + self.max_age_s)
        return f"{expires}.{self._sign(str(expires))}"

    def valid(self, token: str | None, now: float | None = None) -> bool:
        if not self.configured or not token or "." not in token:
            return False
        expires, signature = token.split(".", 1)
        if not hmac.compare_digest(signature, self._sign(expires)):
            return False
        try:
            return int(expires) > (time.time() if now is None else now)
        except ValueError:
            return False


def safe_next(target: str | None) -> str:
    """Only same-site paths, so the login form cannot become an open redirect."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target


def same_origin(origin: str | None, host: str | None) -> bool:
    """Whether a websocket handshake came from a page this server served.

    Browsers attach cookies to websocket handshakes but do not apply the
    same-origin policy to them, so without this check any website a signed-in
    kiosk happened to visit could open a socket using the kiosk's session.
    Clients that send no Origin at all are not browsers, and cannot be tricked
    into making the request.
    """
    if origin is None:
        return True
    return host is not None and urlsplit(origin).netloc == host
