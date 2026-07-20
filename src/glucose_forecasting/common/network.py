#!/usr/bin/env python3
"""Windows TLS workarounds for this dev environment.

Two independent, stackable problems were diagnosed on this machine (both
observed as failures the moment any code creates an ``ssl.SSLContext`` —
directly, or transitively via ``requests``/``httpx``/``huggingface_hub``):
1. **Process crash**: ``OPENSSL_Uplink(...): no OPENSSL_Applink``. Caused by
   Avast/AVG's ``aswMonFltProxy`` network-scanning driver setting the
   ``SSLKEYLOGFILE`` environment variable process-wide (for its own
   HTTPS-inspection feature). The uplink-enabled OpenSSL build shipped in
   the ``uv``-managed ("python-build-standalone") Python 3.12/3.13
   interpreters on Windows crashes natively — bypassing any Python
   ``try``/``except`` — the first time an SSL context is created while that
   variable is set, regardless of its value (named pipe or a plain file
   path both reproduce it). Upstream fix:
   https://github.com/astral-sh/python-build-standalone/pull/1132 (merged
   2026-05-21, builds OpenSSL with ``no-uplink`` for Python 3.12+) — not yet
   reflected in the specific interpreter builds installed here as of this
   writing. This project has no use for TLS key logging (a Wireshark
   debugging feature), so unconditionally clearing the variable before any
   SSL call is safe and has no side effects.
   See also: https://github.com/astral-sh/python-build-standalone/issues/640,
   https://github.com/astral-sh/uv/issues/14333.

2. **Certificate verification failure**: once (1) is fixed, plain
   ``certifi``-based verification then fails with ``[SSL:
   CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate`` —
   the same antivirus's HTTPS-inspection feature re-signs traffic with its
   own locally-installed root CA, which is trusted by Windows' own
   certificate store but isn't in Python's bundled ``certifi`` CA list.
   ``truststore.inject_into_ssl()`` (a small, PyPA-maintained package; the
   same approach ``pip`` itself vendors for this exact scenario) routes
   verification through the OS trust store instead, matching what this
   project's ``pyproject.toml`` already does for ``uv``'s own network calls
   via ``[tool.uv] native-tls = true``.


Call ``apply_windows_tls_workarounds()`` as early as possible in any script
that makes HTTPS requests, before importing anything that might construct an
SSL context (``requests``, ``httpx``, ``huggingface_hub``, ...) — OpenSSL
reads ``SSLKEYLOGFILE`` lazily on first use, so calling this after that
point has no effect. Safe to call unconditionally on any OS/machine; a
no-op where neither problem is present.
"""
from __future__ import annotations

import os


def apply_windows_tls_workarounds() -> None:
    """Clear SSLKEYLOGFILE and route SSL verification through the OS trust store."""
    os.environ.pop("SSLKEYLOGFILE", None)
    import truststore

    truststore.inject_into_ssl()
