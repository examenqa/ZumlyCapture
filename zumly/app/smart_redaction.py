"""Local, suggestion-only OCR scanning for sensitive on-screen text.

The scanner owns one long-lived FFmpeg image pipe and runs OCR only for source
frames that survive the edited timeline. Recognized plaintext exists only in
short-lived local variables inside the worker and is never logged or persisted.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from io import BytesIO
import hashlib
import math
import re
import subprocess
import sys
import threading
from typing import Callable, Iterable, Protocol, Sequence

from PIL import Image
from PySide6.QtCore import QThread, Signal

from .models import (
    OverlayGeometry,
    OverlayTiming,
    RedactionDetectionType,
    RedactionSuggestion,
    SceneSpace,
)
from .timeline import TimelineSpan
from .utils import ffmpeg_exe


SCAN_INTERVAL_MS = 1000.0
MAX_OCR_WIDTH = 1280
FRAME_HASH_DISTANCE = 4
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class SmartRedactionUnavailableError(RuntimeError):
    """Local OCR cannot run on this Windows installation."""


def public_scan_error(exc: BaseException) -> str:
    """Return a privacy-safe UI message without serializing OCR internals."""
    if isinstance(exc, (SmartRedactionUnavailableError, ImportError, ModuleNotFoundError)):
        return (
            "Local OCR is unavailable. Install or repair the Windows OCR "
            "components and add at least one Windows OCR language pack."
        )
    if isinstance(exc, FileNotFoundError):
        return "FFmpeg is unavailable, so the video could not be scanned."
    return "The local OCR scan could not be completed. No recognized text was retained."

_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{5,}\d)(?!\w)")


@dataclass(frozen=True)
class OcrWord:
    text: str
    x: float
    y: float
    width: float
    height: float


class OcrBackend(Protocol):
    def recognize(self, image: Image.Image) -> list[list[OcrWord]]: ...

    def close(self) -> None: ...


class WindowsOcrBackend:
    """Windows.Media.Ocr adapter instantiated inside the scan worker thread."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise SmartRedactionUnavailableError(
                "Smart Redaction requires Windows OCR."
            )
        try:
            from winrt.windows.media.ocr import OcrEngine
        except (ImportError, ModuleNotFoundError) as exc:
            raise SmartRedactionUnavailableError(
                "Windows OCR runtime components are missing."
            ) from exc

        self._engine = OcrEngine.try_create_from_user_profile_languages()
        if self._engine is None:
            raise SmartRedactionUnavailableError(
                "No supported Windows OCR language pack is installed."
            )
        self._loop = asyncio.new_event_loop()

    async def _recognize_async(self, bitmap):
        return await self._engine.recognize_async(bitmap)

    def recognize(self, image: Image.Image) -> list[list[OcrWord]]:
        from winrt.windows.graphics.imaging import (
            BitmapAlphaMode,
            BitmapPixelFormat,
            SoftwareBitmap,
        )
        from winrt.windows.storage.streams import Buffer

        rgba = image.convert("RGBA")
        raw = rgba.tobytes("raw", "BGRA")
        buffer = Buffer(len(raw))
        buffer.length = len(raw)
        memoryview(buffer).cast("B")[:] = raw
        bitmap = SoftwareBitmap(
            BitmapPixelFormat.BGRA8,
            rgba.width,
            rgba.height,
            BitmapAlphaMode.IGNORE,
        )
        try:
            bitmap.copy_from_buffer(buffer)
            result = self._loop.run_until_complete(self._recognize_async(bitmap))
            lines: list[list[OcrWord]] = []
            for line in result.lines:
                words: list[OcrWord] = []
                for word in line.words:
                    rect = word.bounding_rect
                    words.append(
                        OcrWord(
                            str(word.text),
                            float(rect.x),
                            float(rect.y),
                            float(rect.width),
                            float(rect.height),
                        )
                    )
                if words:
                    lines.append(words)
            return lines
        finally:
            bitmap.close()

    def close(self) -> None:
        if not self._loop.is_closed():
            self._loop.close()


@dataclass(frozen=True)
class _DetectedRegion:
    detection_type: RedactionDetectionType
    geometry: OverlayGeometry
    confidence: float
    fingerprint: bytes


@dataclass
class _AccumulatedSuggestion:
    detection_type: RedactionDetectionType
    clip_id: str
    start_ms: float
    end_ms: float
    geometry: OverlayGeometry
    confidence: float
    fingerprint: bytes


def _read_png(stream) -> bytes | None:
    """Read one complete PNG from an FFmpeg image2pipe stream."""
    signature = stream.read(len(PNG_SIGNATURE))
    if not signature:
        return None
    if signature != PNG_SIGNATURE:
        raise RuntimeError("FFmpeg returned an invalid OCR frame stream.")
    payload = bytearray(signature)
    while True:
        header = stream.read(8)
        if len(header) != 8:
            raise RuntimeError("FFmpeg ended in the middle of an OCR frame.")
        length = int.from_bytes(header[:4], "big")
        chunk_type = header[4:]
        body = stream.read(length + 4)
        if len(body) != length + 4:
            raise RuntimeError("FFmpeg ended in the middle of an OCR frame chunk.")
        payload.extend(header)
        payload.extend(body)
        if chunk_type == b"IEND":
            return bytes(payload)


def _difference_hash(image: Image.Image) -> int:
    resampling = getattr(Image, "Resampling", Image)
    small = image.convert("L").resize((9, 8), resample=resampling.LANCZOS)
    flattened = getattr(small, "get_flattened_data", None)
    pixels = list(flattened() if callable(flattened) else small.getdata())
    value = 0
    for row in range(8):
        base = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[base + column] > pixels[base + column + 1])
    return value


def _frame_is_similar(previous: int | None, current: int) -> bool:
    return previous is not None and (previous ^ current).bit_count() <= FRAME_HASH_DISTANCE


def _iou(left: OverlayGeometry, right: OverlayGeometry) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union > 0.0 else 0.0


def _matched_word_geometry(
    words: Sequence[OcrWord],
    match_start: int,
    match_end: int,
    image_size: tuple[int, int],
    *,
    separator_length: int = 1,
) -> OverlayGeometry | None:
    cursor = 0
    selected: list[OcrWord] = []
    for index, word in enumerate(words):
        start = cursor
        end = start + len(word.text)
        if start < match_end and end > match_start:
            selected.append(word)
        cursor = end + (separator_length if index < len(words) - 1 else 0)
    if not selected:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return None
    padding = max(3.0, min(width, height) * 0.004)
    left = max(0.0, min(word.x for word in selected) - padding)
    top = max(0.0, min(word.y for word in selected) - padding)
    right = min(float(width), max(word.x + word.width for word in selected) + padding)
    bottom = min(float(height), max(word.y + word.height for word in selected) + padding)
    return OverlayGeometry(
        left / width,
        top / height,
        max(1.0, right - left) / width,
        max(1.0, bottom - top) / height,
        SceneSpace.VIDEO,
    )


def detect_sensitive_regions(
    lines: Sequence[Sequence[OcrWord]], image_size: tuple[int, int]
) -> list[_DetectedRegion]:
    """Classify OCR lines without returning or retaining recognized plaintext."""
    detected: list[_DetectedRegion] = []
    for words in lines:
        line_text = " ".join(word.text for word in words)
        compact_text = "".join(word.text for word in words)
        matches: list[tuple[RedactionDetectionType, re.Match[str], float, int]] = []
        matches.extend(
            (RedactionDetectionType.EMAIL, match, 0.94, 0)
            for match in _EMAIL_RE.finditer(compact_text)
        )
        for match in _PHONE_RE.finditer(line_text):
            digits = sum(character.isdigit() for character in match.group(0))
            if 7 <= digits <= 15:
                matches.append((RedactionDetectionType.PHONE, match, 0.88, 1))
        for detection_type, match, confidence, separator_length in matches:
            geometry = _matched_word_geometry(
                words,
                match.start(),
                match.end(),
                image_size,
                separator_length=separator_length,
            )
            if geometry is None:
                continue
            normalized = re.sub(r"\s+", "", match.group(0)).casefold()
            fingerprint = hashlib.sha256(normalized.encode("utf-8")).digest()[:12]
            detected.append(
                _DetectedRegion(detection_type, geometry, confidence, fingerprint)
            )
        # Drop the only plaintext aggregate as soon as classification completes.
        line_text = ""
        compact_text = ""
    return detected


class SmartRedactionScanner:
    """Synchronous scanner intended to run inside one cancellable QThread."""

    def __init__(
        self,
        video_path: str,
        source_duration_ms: float,
        visible_spans: Iterable[TimelineSpan],
        *,
        backend_factory: Callable[[], OcrBackend] = WindowsOcrBackend,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> None:
        self.video_path = str(video_path)
        self.source_duration_ms = max(0.0, float(source_duration_ms))
        self.visible_spans = tuple(
            span for span in visible_spans if span.kind == "source" and span.source_end_ms > span.source_start_ms
        )
        self._backend_factory = backend_factory
        self._process_factory = process_factory
        self._progress_callback = progress_callback
        self._cancel_event = threading.Event()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen | None = None

    def cancel(self) -> None:
        self._cancel_event.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _visible_spans_at(self, source_ms: float) -> list[TimelineSpan]:
        return [
            span
            for span in self.visible_spans
            if span.source_start_ms <= source_ms < span.source_end_ms
        ]

    def _merged_visible_ranges(self) -> list[tuple[float, float]]:
        ordered = sorted(
            (
                max(0.0, span.source_start_ms),
                min(self.source_duration_ms, span.source_end_ms),
            )
            for span in self.visible_spans
        )
        merged: list[list[float]] = []
        for start_ms, end_ms in ordered:
            if end_ms <= start_ms:
                continue
            if merged and start_ms <= merged[-1][1] + 0.5:
                merged[-1][1] = max(merged[-1][1], end_ms)
            else:
                merged.append([start_ms, end_ms])
        return [(start, end) for start, end in merged]

    @staticmethod
    def _sample_times(ranges: Sequence[tuple[float, float]]) -> list[float]:
        samples: list[float] = []
        for start_ms, end_ms in ranges:
            count = max(1, int(math.ceil((end_ms - start_ms) / SCAN_INTERVAL_MS)))
            samples.extend(
                min(end_ms - 0.001, start_ms + index * SCAN_INTERVAL_MS)
                for index in range(count)
            )
        return samples

    @staticmethod
    def _visible_span_filter(ranges: Sequence[tuple[float, float]]) -> str:
        if len(ranges) == 1:
            start_ms, end_ms = ranges[0]
            return (
                f"[0:v]trim=start={start_ms / 1000.0:.6f}:"
                f"end={end_ms / 1000.0:.6f},setpts=PTS-STARTPTS,fps=1[ocrbase];"
                f"[ocrbase]scale='min({MAX_OCR_WIDTH},iw)':-2:flags=lanczos[ocr]"
            )
        split_outputs = "".join(f"[ocrin{index}]" for index in range(len(ranges)))
        lines = [f"[0:v]split={len(ranges)}{split_outputs}"]
        for index, (start_ms, end_ms) in enumerate(ranges):
            lines.append(
                f"[ocrin{index}]trim=start={start_ms / 1000.0:.6f}:"
                f"end={end_ms / 1000.0:.6f},setpts=PTS-STARTPTS,fps=1[ocrspan{index}]"
            )
        inputs = "".join(f"[ocrspan{index}]" for index in range(len(ranges)))
        lines.append(f"{inputs}concat=n={len(ranges)}:v=1:a=0[ocrbase]")
        lines.append(
            f"[ocrbase]scale='min({MAX_OCR_WIDTH},iw)':-2:flags=lanczos[ocr]"
        )
        return ";".join(lines)

    def _accumulate(
        self,
        accumulators: list[_AccumulatedSuggestion],
        detection: _DetectedRegion,
        span: TimelineSpan,
        source_ms: float,
    ) -> None:
        end_ms = min(span.source_end_ms, source_ms + SCAN_INTERVAL_MS)
        candidate = next(
            (
                item
                for item in reversed(accumulators)
                if item.clip_id == span.item_id
                and item.detection_type is detection.detection_type
                and item.fingerprint == detection.fingerprint
                and source_ms <= item.end_ms + SCAN_INTERVAL_MS * 0.75
                and _iou(item.geometry, detection.geometry) >= 0.45
            ),
            None,
        )
        if candidate is None:
            accumulators.append(
                _AccumulatedSuggestion(
                    detection.detection_type,
                    span.item_id,
                    source_ms,
                    max(source_ms + 250.0, end_ms),
                    detection.geometry,
                    detection.confidence,
                    detection.fingerprint,
                )
            )
            return
        candidate.end_ms = max(candidate.end_ms, end_ms)
        candidate.confidence = max(candidate.confidence, detection.confidence)
        candidate.geometry = detection.geometry

    def run(self) -> list[RedactionSuggestion]:
        if self.cancelled or not self.visible_spans or self.source_duration_ms <= 0.0:
            return []
        visible_ranges = self._merged_visible_ranges()
        sample_times = self._sample_times(visible_ranges)
        if not sample_times:
            return []
        backend: OcrBackend | None = None
        process: subprocess.Popen | None = None
        try:
            backend = self._backend_factory()
        except (ImportError, ModuleNotFoundError) as exc:
            raise SmartRedactionUnavailableError(
                "Windows OCR runtime components are missing."
            ) from exc
        if self.cancelled:
            backend.close()
            return []
        command = [
            ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            self.video_path,
            "-filter_complex",
            self._visible_span_filter(visible_ranges),
            "-map",
            "[ocr]",
            "-an",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]
        try:
            process = self._process_factory(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            backend.close()
            raise
        with self._process_lock:
            self._process = process
        if self.cancelled and process.poll() is None:
            process.terminate()
        if process.stdout is None:
            backend.close()
            if process.poll() is None:
                process.terminate()
            with self._process_lock:
                self._process = None
            raise RuntimeError("FFmpeg OCR frame pipe did not open.")
        accumulators: list[_AccumulatedSuggestion] = []
        last_hash: int | None = None
        last_detections: list[_DetectedRegion] = []
        total = len(sample_times)
        frame_index = 0
        try:
            while not self.cancelled:
                payload = _read_png(process.stdout)
                if payload is None:
                    break
                if frame_index >= total:
                    break
                source_ms = sample_times[frame_index]
                frame_index += 1
                spans = self._visible_spans_at(source_ms)
                if spans:
                    with Image.open(BytesIO(payload)) as decoded:
                        image = decoded.convert("RGB")
                    frame_hash = _difference_hash(image)
                    if _frame_is_similar(last_hash, frame_hash):
                        detections = last_detections
                    else:
                        detections = detect_sensitive_regions(
                            backend.recognize(image), image.size
                        )
                        last_hash = frame_hash
                        last_detections = detections
                    for span in spans:
                        for detection in detections:
                            self._accumulate(accumulators, detection, span, source_ms)
                if self._progress_callback is not None:
                    self._progress_callback(
                        min(100, int(frame_index * 100 / total)),
                        min(frame_index, total),
                        total,
                    )
            if self.cancelled:
                return []
            return_code = process.wait(timeout=3.0)
            if return_code != 0:
                raise RuntimeError("FFmpeg could not decode the video for local OCR.")
            return [
                RedactionSuggestion(
                    id="redaction-" + hashlib.sha256(
                        (
                            f"{item.detection_type.value}|{item.clip_id}|"
                            f"{round(item.start_ms)}|{round(item.geometry.x, 4)}|"
                            f"{round(item.geometry.y, 4)}|{round(item.geometry.width, 4)}|"
                            f"{round(item.geometry.height, 4)}"
                        ).encode("utf-8")
                    ).hexdigest()[:24],
                    detection_type=item.detection_type,
                    timing=OverlayTiming(item.start_ms, item.end_ms, item.clip_id),
                    geometry=item.geometry,
                    confidence=item.confidence,
                )
                for item in accumulators
            ]
        finally:
            if backend is not None:
                backend.close()
            if process is not None:
                if process.poll() is None:
                    try:
                        process.terminate() if self.cancelled else process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                try:
                    process.wait(timeout=1.0)
                except (subprocess.TimeoutExpired, OSError):
                    pass
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
            with self._process_lock:
                self._process = None


class SmartRedactionWorker(QThread):
    """Exactly one cancellable background worker for one project scan."""

    progress = Signal(int, int, int)
    completed = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        video_path: str,
        source_duration_ms: float,
        visible_spans: Iterable[TimelineSpan],
        parent=None,
        *,
        scanner_factory: Callable[..., SmartRedactionScanner] = SmartRedactionScanner,
    ) -> None:
        super().__init__(parent)
        self._scanner = scanner_factory(
            video_path,
            source_duration_ms,
            visible_spans,
            progress_callback=self.progress.emit,
        )

    def cancel(self) -> None:
        self._scanner.cancel()

    def run(self) -> None:
        try:
            suggestions = self._scanner.run()
            if self._scanner.cancelled:
                self.cancelled.emit()
            else:
                self.completed.emit(suggestions)
        except Exception as exc:
            if self._scanner.cancelled:
                self.cancelled.emit()
            else:
                self.failed.emit(public_scan_error(exc))
