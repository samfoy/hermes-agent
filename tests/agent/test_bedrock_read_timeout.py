"""Regression tests for the Bedrock read-timeout failure mode.

Observed symptom (turn died after ~5m30s at 0.4 t/s on a reasoning model)::

    Error: AWSHTTPSConnectionPool(host='bedrock-runtime.us-west-2.amazonaws.com',
                                  port=443): Read timed out.

Two independent defects produced it:

1. ``boto3.client("bedrock-runtime", ...)`` was built with **no** botocore
   ``Config``, so it inherited botocore's default ``read_timeout`` of 60s.
   Extended-thinking models routinely pause longer than 60s between wire
   events, so the transport timeout fired before Hermes' own stale-stream
   watchdog (180-300s) could act — inverting the ownership of "is this stream
   dead?" away from the component that has the diagnostics, client eviction,
   and retry/fallback escalation.

2. ``is_stale_connection_error`` did not match a **raw**
   ``urllib3.exceptions.ReadTimeoutError``. botocore only translates urllib3
   exceptions into its own hierarchy inside ``URLLib3Session.send()`` (the
   initial request). Mid-stream EventStream reads happen outside that
   try/except, so the urllib3 error escapes unwrapped — and it inherits from
   ``PoolError``/``HTTPError``, not ``ProtocolError``/``ConnectionError``, so
   the classifier missed it. The cached (now-dead) client was never evicted
   and the failure surfaced as a hard turn error instead of a retry.

The ``AWSHTTPSConnectionPool`` prefix in the message is botocore's urllib3
connection-pool subclass, which is the fingerprint distinguishing defect 2's
mid-stream leak from a plain botocore-wrapped timeout.
"""

import pytest

pytest.importorskip("boto3", reason="boto3 required for Bedrock tests")
pytest.importorskip("botocore", reason="botocore required for Bedrock tests")

from agent import bedrock_adapter as ba  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """Clients are cached per-region; isolate each test."""
    ba.reset_client_cache()
    yield
    ba.reset_client_cache()


# ---------------------------------------------------------------------------
# Defect 1 — botocore's 60s default read_timeout must never be inherited
# ---------------------------------------------------------------------------

class TestBedrockClientTimeoutConfig:
    def test_read_timeout_is_not_botocore_default(self):
        """The 60s botocore default is the bug — it must be overridden."""
        from botocore.config import Config

        assert Config().read_timeout == 60, (
            "botocore default changed; this test's premise needs revisiting"
        )
        client = ba._get_bedrock_runtime_client("us-west-2")
        assert client.meta.config.read_timeout != 60

    def test_read_timeout_clears_max_stale_watchdog(self):
        """Transport timeout must sit above the stale watchdog, not below it.

        ``_derive_stream_stale_timeout`` scales to 300s for >100k-token
        contexts. If the socket read timeout is lower, it fires first and
        steals ownership of dead-stream detection from the watchdog.
        """
        client = ba._get_bedrock_runtime_client("us-west-2")
        assert client.meta.config.read_timeout > 300.0

    def test_connect_timeout_stays_short(self):
        """A long connect timeout only delays failover to a fallback provider."""
        client = ba._get_bedrock_runtime_client("us-west-2")
        connect = client.meta.config.connect_timeout
        assert 0 < connect <= 60.0
        assert connect < client.meta.config.read_timeout

    def test_retries_upgraded_off_legacy_mode(self):
        """legacy mode misses throttling/5xx shapes Bedrock returns under load."""
        client = ba._get_bedrock_runtime_client("us-west-2")
        retries = client.meta.config.retries
        assert retries.get("mode") in ("standard", "adaptive")

    def test_tcp_keepalive_enabled(self):
        """Long thinking pauses are when NAT/proxies silently cull idle sockets."""
        client = ba._get_bedrock_runtime_client("us-west-2")
        assert client.meta.config.tcp_keepalive is True

    def test_control_plane_client_also_configured(self):
        client = ba._get_bedrock_control_client("us-west-2")
        assert client.meta.config.read_timeout != 60

    def test_provider_timeout_override_wins(self):
        """providers.bedrock.request_timeout_seconds must reach the boto client."""
        client = ba._get_bedrock_runtime_client("us-west-2", 1800.0)
        assert client.meta.config.read_timeout == 1800.0

    def test_env_override_respected(self, monkeypatch):
        monkeypatch.setenv("HERMES_BEDROCK_READ_TIMEOUT", "1234")
        ba.reset_client_cache()
        client = ba._get_bedrock_runtime_client("eu-west-1")
        assert client.meta.config.read_timeout == 1234.0

    def test_explicit_arg_beats_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_BEDROCK_READ_TIMEOUT", "1234")
        ba.reset_client_cache()
        client = ba._get_bedrock_runtime_client("eu-west-1", 777.0)
        assert client.meta.config.read_timeout == 777.0

    def test_zero_or_negative_provider_timeout_falls_back(self):
        """A misconfigured 0/-1 must not disable the timeout entirely."""
        for bad in (0, -1, 0.0):
            ba.reset_client_cache()
            client = ba._get_bedrock_runtime_client("us-west-2", bad)
            assert client.meta.config.read_timeout > 300.0


# ---------------------------------------------------------------------------
# Defect 2 — raw urllib3 mid-stream timeouts must classify as stale
# ---------------------------------------------------------------------------

def _raw_urllib3_read_timeout():
    """Build the exact exception a mid-stream EventStream read raises."""
    import urllib3
    from urllib3.exceptions import ReadTimeoutError

    pool = urllib3.HTTPSConnectionPool(
        "bedrock-runtime.us-west-2.amazonaws.com", port=443
    )
    return ReadTimeoutError(pool, "https://bedrock-runtime.us-west-2.amazonaws.com",
                            "Read timed out.")


class TestStaleClassificationOfReadTimeouts:
    def test_raw_urllib3_read_timeout_is_stale(self):
        """The reported error — must be retryable, not a hard turn failure."""
        assert ba.is_stale_connection_error(_raw_urllib3_read_timeout()) is True

    def test_urllib3_read_timeout_is_not_a_connection_error_subclass(self):
        """Documents *why* the original check missed it."""
        from urllib3.exceptions import (
            ReadTimeoutError,
            ProtocolError,
            NewConnectionError,
        )

        assert not issubclass(ReadTimeoutError, ProtocolError)
        assert not issubclass(ReadTimeoutError, NewConnectionError)

    def test_botocore_wrapped_read_timeout_still_stale(self):
        """The initial-request path (already worked) must not regress."""
        from botocore.exceptions import ReadTimeoutError

        exc = ReadTimeoutError(
            endpoint_url="https://bedrock-runtime.us-west-2.amazonaws.com",
            error="Read timed out.",
        )
        assert ba.is_stale_connection_error(exc) is True

    def test_urllib3_connect_timeout_is_stale(self):
        from urllib3.exceptions import ConnectTimeoutError

        assert ba.is_stale_connection_error(ConnectTimeoutError("timed out")) is True

    def test_protocol_error_still_stale(self):
        from urllib3.exceptions import ProtocolError

        assert ba.is_stale_connection_error(ProtocolError("closed")) is True

    def test_application_errors_are_not_stale(self):
        """Guard against over-matching: real bugs must still surface."""
        assert ba.is_stale_connection_error(ValueError("bad input")) is False
        assert ba.is_stale_connection_error(KeyError("missing")) is False
        assert ba.is_stale_connection_error(AssertionError("app-level")) is False

    def test_client_error_is_not_stale(self):
        """Throttling/validation are handled elsewhere, not by client eviction."""
        from botocore.exceptions import ClientError

        exc = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "bad"}}, "Converse"
        )
        assert ba.is_stale_connection_error(exc) is False


# ---------------------------------------------------------------------------
# Eviction — a stale-classified error must yield a fresh pool on retry
# ---------------------------------------------------------------------------

class TestClientEviction:
    def test_stale_error_eviction_produces_new_client(self):
        first = ba._get_bedrock_runtime_client("us-west-2")
        assert ba._get_bedrock_runtime_client("us-west-2") is first, "should cache"

        exc = _raw_urllib3_read_timeout()
        assert ba.is_stale_connection_error(exc) is True
        assert ba.invalidate_runtime_client("us-west-2") is True

        assert ba._get_bedrock_runtime_client("us-west-2") is not first

    def test_evicting_uncached_region_is_noop(self):
        assert ba.invalidate_runtime_client("ap-southeast-2") is False

    def test_eviction_is_per_region(self):
        west = ba._get_bedrock_runtime_client("us-west-2")
        east = ba._get_bedrock_runtime_client("us-east-1")
        ba.invalidate_runtime_client("us-west-2")

        assert ba._get_bedrock_runtime_client("us-east-1") is east
        assert ba._get_bedrock_runtime_client("us-west-2") is not west
