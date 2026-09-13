"""Tests for the login gate.

The kiosk answers "who is this face?" with a name, so the properties that
matter are the ones an outsider would probe: that nothing leaks without a
session, that sessions cannot be forged or outlive their expiry, and that a
logged-in kiosk cannot be borrowed by another website.

The app-level tests never enter the app's lifespan, so no model is loaded and
none of them touch a database.
"""

import pytest
from starlette.websockets import WebSocketDisconnect

from server.auth import Auth, safe_next, same_origin


def make(password: str | None = "correct horse") -> Auth:
    return Auth(password, secret="test-secret", max_age_s=3600)


# --- sessions ------------------------------------------------------------


def test_the_right_password_is_accepted_and_others_are_not():
    auth = make()
    assert auth.check_password("correct horse")
    assert not auth.check_password("correct horse ")
    assert not auth.check_password("")


def test_with_no_password_configured_nobody_can_sign_in():
    auth = make(password=None)
    assert not auth.check_password("")
    assert not auth.valid(auth.issue())


def test_an_issued_session_is_valid():
    auth = make()
    assert auth.valid(auth.issue())


def test_a_session_stops_working_when_it_expires():
    auth = make()
    token = auth.issue(now=1_000)
    assert auth.valid(token, now=1_000 + 3599)
    assert not auth.valid(token, now=1_000 + 3601)


def test_extending_the_expiry_by_hand_breaks_the_signature():
    auth = make()
    expires, signature = auth.issue().split(".")
    forged = f"{int(expires) + 10**8}.{signature}"
    assert not auth.valid(forged)


def test_changing_the_password_signs_every_device_out():
    token = Auth("old", secret="test-secret", max_age_s=3600).issue()
    assert not Auth("new", secret="test-secret", max_age_s=3600).valid(token)


@pytest.mark.parametrize("token", [None, "", "no-dot", "abc.def", "1.2.3"])
def test_malformed_tokens_are_rejected(token):
    assert not make().valid(token)


@pytest.mark.parametrize(
    "target, expected",
    [
        ("/enrol", "/enrol"),
        (None, "/"),
        ("https://evil.example", "/"),
        ("//evil.example", "/"),
        ("/\\evil.example", "/"),
    ],
)
def test_login_only_redirects_within_the_site(target, expected):
    assert safe_next(target) == expected


def test_websocket_origin_must_match_the_host():
    assert same_origin("https://kiosk.example", "kiosk.example")
    assert not same_origin("https://evil.example", "kiosk.example")
    assert same_origin(None, "kiosk.example"), "non-browser clients send no Origin"


# --- the app behind the gate ----------------------------------------------


@pytest.fixture
def server(monkeypatch):
    import server.app as module

    monkeypatch.setattr(module, "auth", make())
    monkeypatch.setattr(module, "FAILED_LOGIN_DELAY_S", 0)
    monkeypatch.setattr(module.app.state, "open_access", False, raising=False)
    return module


@pytest.fixture
def client(server):
    from fastapi.testclient import TestClient

    # Deliberately not used as a context manager: that would run the lifespan
    # and load several hundred megabytes of model.
    return TestClient(server.app)


def sign_in(client, password="correct horse", next_path="/"):
    return client.post(
        "/login", data={"password": password, "next": next_path}, follow_redirects=False
    )


def test_pages_send_you_to_sign_in(client):
    response = client.get("/enrol", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login?next=/enrol"


@pytest.mark.parametrize("path", ["/api/roster", "/api/events", "/api/config"])
def test_the_api_refuses_anyone_without_a_session(client, path):
    assert client.get(path).status_code == 401


def test_enrolment_and_deletion_refuse_anyone_without_a_session(client):
    assert client.post("/api/enrol", json={"name": "x", "images": ["y"]}).status_code == 401
    assert client.delete("/api/people/anyone").status_code == 401


def test_sign_in_page_health_check_and_assets_are_public(client):
    assert client.get("/login").status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_a_wrong_password_sets_no_session(client):
    response = sign_in(client, password="guess")
    assert response.status_code == 303
    assert "error=wrong" in response.headers["location"]
    assert "presence_session" not in response.cookies
    assert client.get("/api/config").status_code == 401


def test_signing_in_returns_you_where_you_were_going(client):
    response = sign_in(client, next_path="/history")
    assert response.status_code == 303
    assert response.headers["location"] == "/history"
    assert client.get("/history").status_code == 200
    assert client.get("/api/config").status_code == 200


def test_sign_in_will_not_redirect_off_site(client):
    response = sign_in(client, next_path="https://evil.example")
    assert response.headers["location"] == "/"


def test_the_session_cookie_cannot_be_read_by_scripts(client):
    assert "httponly" in sign_in(client).headers["set-cookie"].lower()


def test_signing_out_ends_the_session(client):
    sign_in(client)
    client.get("/logout")
    assert client.get("/api/config").status_code == 401


def test_the_verification_socket_refuses_anyone_without_a_session(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/verify"):
            pass


def test_another_website_cannot_use_a_signed_in_kiosks_session(client):
    sign_in(client)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/verify", headers={"origin": "https://evil.example"}):
            pass


def test_local_development_needs_no_sign_in(server, client):
    server.app.state.open_access = True
    assert client.get("/enrol").status_code == 200


def test_the_server_will_not_go_public_without_a_password(monkeypatch, server):
    import config

    monkeypatch.setattr(config, "HOST", "0.0.0.0")
    monkeypatch.setattr(config, "PASSWORD", None)
    with pytest.raises(SystemExit, match="PRESENCE_PASSWORD"):
        server.serve()
    assert not server.app.state.open_access
