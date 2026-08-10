"""Credential storage using Windows DPAPI.

Non-empty credential writes are fail-closed: if Windows DPAPI cannot encrypt
the value, the caller receives an exception instead of a plaintext secret
being written to QSettings. Legacy plaintext values remain readable so old
installations can migrate on their next successful save.
"""

import base64
import ctypes
import ctypes.wintypes as wintypes
import sys
from dataclasses import dataclass


_HAS_DPAPI = sys.platform == "win32"

if _HAS_DPAPI:
    class _DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    _crypt32 = ctypes.windll.crypt32
    _kernel32 = ctypes.windll.kernel32


class DPAPIEncryptionError(RuntimeError):
    """Raised when a credential cannot be protected with Windows DPAPI."""


class DPAPIDecryptionError(RuntimeError):
    """Raised when a DPAPI credential cannot be decrypted."""


@dataclass(frozen=True)
class CredentialReadResult:
    """Credential value plus migration metadata for legacy plaintext data."""

    value: str
    is_legacy_plaintext: bool = False


def read_credential(stored: str) -> CredentialReadResult:
    """Read a stored value while exposing whether it needs migration."""
    if not stored:
        return CredentialReadResult("")
    if not stored.startswith("dpapi:"):
        return CredentialReadResult(stored, is_legacy_plaintext=True)
    return CredentialReadResult(_unprotect_dpapi(stored), is_legacy_plaintext=False)


def protect(plaintext: str) -> str:
    """Encrypt a string with DPAPI and return a prefixed base64 blob.

    Empty values remain empty. All non-empty values must be encrypted; there
    is deliberately no plaintext fallback.
    """
    if not plaintext:
        return plaintext
    if not _HAS_DPAPI:
        raise DPAPIEncryptionError(
            "Cannot protect credential because Windows DPAPI is unavailable."
        )

    blob_out = _DATA_BLOB()
    protected_ptr = None
    try:
        encoded = plaintext.encode("utf-8")
        input_buffer = ctypes.create_string_buffer(encoded, len(encoded))
        blob_in = _DATA_BLOB(
            len(encoded),
            ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_char)),
        )
        ok = _crypt32.CryptProtectData(
            ctypes.byref(blob_in),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(blob_out),
        )
        if not ok:
            raise DPAPIEncryptionError("Windows CryptProtectData failed.")

        protected_ptr = blob_out.pbData
        encrypted = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        return "dpapi:" + base64.b64encode(encrypted).decode("ascii")
    except DPAPIEncryptionError:
        raise
    except Exception as exc:
        raise DPAPIEncryptionError(
            f"Windows DPAPI encryption failed: {exc}"
        ) from exc
    finally:
        if protected_ptr:
            _kernel32.LocalFree(protected_ptr)


def unprotect(stored: str) -> str:
    """Decrypt a stored value, accepting legacy plaintext for migration."""
    return read_credential(stored).value


def _unprotect_dpapi(stored: str) -> str:
    """Decrypt a DPAPI value after its prefix has been validated."""
    if not _HAS_DPAPI:
        raise DPAPIDecryptionError(
            "Cannot decrypt stored credential - DPAPI is only available "
            "on Windows. Please re-enter your API key."
        )

    blob_out = _DATA_BLOB()
    plaintext_ptr = None
    try:
        encrypted = base64.b64decode(stored[6:], validate=True)
        if not encrypted:
            raise ValueError("empty DPAPI payload")
        input_buffer = ctypes.create_string_buffer(encrypted, len(encrypted))
        blob_in = _DATA_BLOB(
            len(encrypted),
            ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_char)),
        )
        ok = _crypt32.CryptUnprotectData(
            ctypes.byref(blob_in),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(blob_out),
        )
        if not ok:
            raise DPAPIDecryptionError(
                "Failed to decrypt stored credential. The key may have been "
                "encrypted by a different Windows user. Please re-enter your API key."
            )

        plaintext_ptr = blob_out.pbData
        return ctypes.string_at(blob_out.pbData, blob_out.cbData).decode("utf-8")
    except DPAPIDecryptionError:
        raise
    except Exception as exc:
        raise DPAPIDecryptionError(
            f"Failed to decrypt stored credential: {exc}. "
            "Please re-enter your API key."
        ) from exc
    finally:
        if plaintext_ptr:
            _kernel32.LocalFree(plaintext_ptr)
