"""Analyze mouse + click activity to auto-generate zoom keyframes.

Click clusters are the only remaining automatic zoom signal. The
``key_events`` argument is retained for backward-compatible call sites but
is intentionally ignored now that the keystroke feature has been removed.
Mouse settlements (cursor resting) are **not** used as zoom triggers.
"""

import logging
import math
from typing import List, Optional, Sequence, Tuple

from .models import MousePosition, KeyEvent, ClickEvent, VideoSegment, ZoomKeyframe

logger = logging.getLogger(__name__)


# ── Tuning constants ────────────────────────────────────────────────

WINDOW_MS = 500           # time window for averaging (ms)
MIN_GAP_MS = 4000         # minimum gap between separate clusters (was 2500)
ZOOM_LEVEL = 1.5          # zoom factor for auto-keyframes (overridden by depth setting)
ZOOM_HOLD_TYPING_MS = 3000 # hold longer for typing (user is still working)
ZOOM_HOLD_CLICK_MS = 2000  # hold for click clusters
TRANSITION_MS = 800       # easing duration (zoom-in)
ZOOM_PRE_ROLL_MS = 500    # start generated zoom keyframes before the planned camera move
PAN_TRANSITION_MS = 400   # base duration for panning to new target while zoomed
PAN_TRANSITION_MAX_MS = 700  # cap pan duration even for large distances
PAN_MERGE_GAP_MS = 2200  # use pan only when clicks are too tight for a clean zoom-out/in cycle
MAX_CLUSTER_DURATION_MS = 8000  # split clusters that exceed this total span
ANTICIPATION_MS = 400     # arrive this many ms *before* action starts so the viewer sees the trigger

# Thresholds
TYPING_MIN_KPS = 1.0      # minimum keys-per-second to count as typing
MOUSE_STILL_PX_MS = 0.5   # mouse speed below this = "still" (px/ms)
MOUSE_TYPING_PX_MS = 3.0  # mouse speed below this = "slow enough for typing" (px/ms)
DECEL_MIN_RATIO = 3.0     # speed must drop by at least this factor to count
CLICK_WINDOW_MS = 3000    # sliding window for click-cluster detection
CLICK_MIN_COUNT = 1       # minimum clicks in window to trigger zoom

# Signal weights (higher = preferred when ranking mixed clusters)
WEIGHT_TYPING = 1.0
WEIGHT_CLICK = 1.2

# Spatial-aware clustering: merge same-type peaks that are close in space
CLICK_MERGE_GAP_MS = 8000    # merge click peaks within 8s if spatially close
TYPING_MERGE_GAP_MS = 6000   # merge typing peaks within 6s if spatially close
SPATIAL_MERGE_DIST = 0.15    # normalized distance threshold for spatial proximity

# Pan dampening: don't drag viewport center all the way to the target
PAN_VIEWPORT_MARGIN = 0.15   # margin fraction within viewport edge


def _dampen_pan(
    target_x: float, target_y: float, zoom: float,
    margin: float = PAN_VIEWPORT_MARGIN,
    from_x: float = 0.5, from_y: float = 0.5,
) -> Tuple[float, float]:
    """Compute pan to keep *target* visible within the zoomed viewport.

    Starts from (*from_x*, *from_y*) — the viewport center before this
    move — and shifts the minimum amount needed so the target lands
    inside the visible area with a small margin from the edge.

    At low zoom levels most positions are already visible and no
    panning is needed at all.
    """
    if zoom <= 1.0:
        return 0.5, 0.5

    # Half of viewport extent in normalised coords
    half_vw = 0.5 / zoom
    half_vh = 0.5 / zoom

    # Shrink by margin so the target isn't right at the edge
    eff_hw = half_vw * (1.0 - margin)
    eff_hh = half_vh * (1.0 - margin)

    pan_x, pan_y = from_x, from_y

    # Shift only if the target falls outside the effective visible band
    if target_x < pan_x - eff_hw:
        pan_x = target_x + eff_hw
    elif target_x > pan_x + eff_hw:
        pan_x = target_x - eff_hw

    if target_y < pan_y - eff_hh:
        pan_y = target_y + eff_hh
    elif target_y > pan_y + eff_hh:
        pan_y = target_y - eff_hh

    # Clamp so the viewport doesn't fly off the edge of the source
    pan_x = max(half_vw, min(1.0 - half_vw, pan_x))
    pan_y = max(half_vh, min(1.0 - half_vh, pan_y))

    return pan_x, pan_y


def analyze_activity(
    mouse_track: List[MousePosition],
    monitor_rect: dict,
    key_events: Optional[List[KeyEvent]] = None,
    click_events: Optional[List[ClickEvent]] = None,
    max_clusters: int | None = None,
    zoom_level: float = ZOOM_LEVEL,
    follow_cursor: bool = True,
    min_gap_ms: int = MIN_GAP_MS,
    visible_segments: Sequence[VideoSegment] | None = None,
) -> List[ZoomKeyframe]:
    """Detect activity clusters from mouse + click data.

    Args:
        zoom_level: Zoom factor for auto-keyframes (e.g. 1.25, 1.5, 2.0).
        follow_cursor: If True, pan follows cursor; if False, zoom to center.
        max_clusters: Optional explicit compatibility cap. Auto Zoom leaves
            this unset so every eligible cluster is processed.
        min_gap_ms: Minimum gap between separate clusters (ms).
        visible_segments: Ordered source clips that remain in the edited
            timeline. Clicks in deleted source ranges are ignored.

    Returns zoom-in / zoom-out keyframe pairs sorted by timestamp.
    """
    if len(mouse_track) < 10:
        return []

    if zoom_level <= 0.0:
        zoom_level = 1.0

    mon_left = monitor_rect.get("left", 0)
    mon_top = monitor_rect.get("top", 0)
    mon_w = max(monitor_rect.get("width", 1), 1)
    mon_h = max(monitor_rect.get("height", 1), 1)

    ignored_key_count = len(key_events or [])
    click_list = list(click_events or [])
    if visible_segments:
        visible_ranges = [
            (float(segment.start_ms), float(segment.end_ms))
            for segment in visible_segments
            if float(segment.end_ms) > float(segment.start_ms)
        ]
        click_list = [
            click
            for click in click_list
            if any(
                start_ms <= float(click.timestamp) < end_ms
                for start_ms, end_ms in visible_ranges
            )
        ]
    duration = max(float(sample.timestamp) for sample in mouse_track)

    logger.info(
        "Analyzing: %d mouse samples, %d ignored legacy key events, %d click events, duration=%.0fms",
        len(mouse_track), ignored_key_count, len(click_list),
        duration,
    )
    if ignored_key_count:
        logger.debug("Ignoring removed keystroke activity during auto-zoom analysis")

    # ── 1. Per-sample mouse velocity + normalized position ──────────
    samples: List[Tuple[float, float, float, float]] = []
    for i in range(1, len(mouse_track)):
        prev, curr = mouse_track[i - 1], mouse_track[i]
        dt = max(curr.timestamp - prev.timestamp, 1.0)
        dx = curr.x - prev.x
        dy = curr.y - prev.y
        # A resumed cursor sample begins a new continuity span. Treating the
        # paused movement as velocity would create a false activity peak.
        speed = 0.0 if curr.resume_boundary else math.sqrt(dx * dx + dy * dy) / dt
        nx = max(0.0, min(1.0, (curr.x - mon_left) / mon_w))
        ny = max(0.0, min(1.0, (curr.y - mon_top) / mon_h))
        samples.append((curr.timestamp, speed, nx, ny))

    if not samples:
        return []

    # ── 2. Find click-driven peaks ──────────────────────────────────
    WindowInfo = Tuple[float, float, float, float, str]  # (time, score, x, y, label)
    peaks: List[WindowInfo] = []

    # Click peaks: emit one peak per click. Spatial clustering below merges
    # repeated same-area clicks, while far-apart clicks remain separate so the
    # camera can pan instead of aiming at a burst centroid.
    if len(click_list) >= CLICK_MIN_COUNT:
        sorted_clicks = sorted(click_list, key=lambda c: c.timestamp)
        for click in sorted_clicks:
            nx = max(0.0, min(1.0, (click.x - mon_left) / mon_w))
            ny = max(0.0, min(1.0, (click.y - mon_top) / mon_h))
            peaks.append((float(click.timestamp), WEIGHT_CLICK, nx, ny, "click"))

    if not peaks:
        return []  # no click activity → no zoom keyframes

    # Log peak breakdown
    peak_types = {}
    for p in peaks:
        peak_types[p[4]] = peak_types.get(p[4], 0) + 1
    logger.info("Peaks: %s", peak_types)

    # ── 4. Cluster nearby peaks (spatial-aware) ────────────────────
    #
    # Two peaks merge into the same cluster when EITHER:
    #   a) they are within min_gap_ms of any peak already in the cluster, OR
    #   b) they are click peaks that are spatially close
    #      (< SPATIAL_MERGE_DIST), and within an extended time threshold.
    #
    # This prevents repeated zoom-out → zoom-in cycles when the user
    # clicks in the same area with small pauses.
    peaks.sort(key=lambda p: p[0])
    clusters: List[List[WindowInfo]] = []
    current_cluster: List[WindowInfo] = [peaks[0]]

    for p in peaks[1:]:
        should_merge = False
        for cp in current_cluster:
            gap = p[0] - cp[0]
            # (a) merge within the base time gap only when click targets are
            # spatially close. Far-apart clicks should become separate
            # clusters so the camera can pan instead of aiming at a centroid.
            if gap < min_gap_ms:
                if p[4] == cp[4] and p[4] == "click":
                    dist = math.sqrt((p[2] - cp[2]) ** 2 + (p[3] - cp[3]) ** 2)
                    should_merge = dist < SPATIAL_MERGE_DIST
                else:
                    should_merge = True
                if should_merge:
                    break
            # (b) extended merge for same-type, spatially close click peaks
            if p[4] == cp[4] and p[4] == "click":
                dist = math.sqrt((p[2] - cp[2]) ** 2 + (p[3] - cp[3]) ** 2)
                if dist < SPATIAL_MERGE_DIST and gap < CLICK_MERGE_GAP_MS:
                    should_merge = True
                    break

        if should_merge:
            current_cluster.append(p)
        else:
            clusters.append(current_cluster)
            current_cluster = [p]
    clusters.append(current_cluster)

    # Split clusters that exceed MAX_CLUSTER_DURATION_MS so a single
    # zoom block never spans the whole video.
    split_clusters: List[List[WindowInfo]] = []
    for cluster in clusters:
        cluster.sort(key=lambda p: p[0])
        span = cluster[-1][0] - cluster[0][0]
        if span <= MAX_CLUSTER_DURATION_MS:
            split_clusters.append(cluster)
            continue
        # Walk through peaks; start a new sub-cluster whenever adding
        # the next peak would exceed the duration limit.
        sub: List[WindowInfo] = [cluster[0]]
        for p in cluster[1:]:
            if p[0] - sub[0][0] > MAX_CLUSTER_DURATION_MS:
                split_clusters.append(sub)
                sub = [p]
            else:
                sub.append(p)
        if sub:
            split_clusters.append(sub)
    clusters = split_clusters

    # Auto Zoom processes every eligible cluster. Keep the explicit cap only
    # for backward-compatible callers that deliberately request one.
    def cluster_peak_score(c: List[WindowInfo]) -> float:
        return max(p[1] for p in c)

    if max_clusters is not None and int(max_clusters) > 0:
        clusters.sort(key=cluster_peak_score, reverse=True)
        clusters = clusters[:int(max_clusters)]
        clusters.sort(key=lambda c: c[0][0])

    # ── 5. Generate keyframe pairs ──────────────────────────────────
    #
    # For each cluster we zoom in at the *start* of its activity and
    # zoom out after the *end* + a hold period.  The pan target is the
    # score-weighted centroid of the cluster, dampened so the viewport
    # doesn't jump excessively.
    #
    # When consecutive clusters are close in time, we stay zoomed in
    # and smoothly **pan** to the new location instead of zooming out
    # and back in.  This avoids disorienting zoom-out / zoom-in cycles
    # when actions happen across different parts of the screen.
    keyframes: List[ZoomKeyframe] = []

    # Pre-compute cluster info for boundary clamping
    cluster_info: List[dict] = []
    for cluster in clusters:
        best = max(cluster, key=lambda p: p[1])
        cluster_start = min(p[0] for p in cluster)
        cluster_end = max(p[0] for p in cluster)

        # Score-weighted centroid position (raw, un-dampened)
        total_score = sum(p[1] for p in cluster)
        if follow_cursor and total_score > 0:
            raw_x = sum(p[2] * p[1] for p in cluster) / total_score
            raw_y = sum(p[3] * p[1] for p in cluster) / total_score
        elif follow_cursor:
            raw_x, raw_y = best[2], best[3]
        else:
            raw_x, raw_y = 0.5, 0.5

        label = best[4]

        # Center viewport on the target position, clamped so the
        # viewport stays within source bounds.
        safe_zoom = zoom_level if zoom_level > 0.0 else 1.0
        half_vw = 0.5 / safe_zoom
        half_vh = 0.5 / safe_zoom
        pan_x = max(half_vw, min(1.0 - half_vw, raw_x))
        pan_y = max(half_vh, min(1.0 - half_vh, raw_y))
        # Anticipation: arrive ANTICIPATION_MS *before* the action starts,
        # so the viewer sees the trigger.  The transition must complete at
        # (cluster_start - ANTICIPATION_MS), so it starts one TRANSITION_MS
        # before that.
        zoom_in_time = max(0.0, cluster_start - TRANSITION_MS - ANTICIPATION_MS)
        zoom_in_time = max(0.0, zoom_in_time - ZOOM_PRE_ROLL_MS)

        hold_ms = ZOOM_HOLD_CLICK_MS
        reason = "Click cluster detected"

        cluster_info.append({
            "start": cluster_start,
            "end": cluster_end,
            "zoom_in_time": zoom_in_time,
            "pan_x": pan_x,
            "pan_y": pan_y,
            "raw_x": raw_x,
            "raw_y": raw_y,
            "label": label,
            "hold_ms": hold_ms,
            "reason": reason,
        })

    # ── Build chains of clusters to pan between ────────────────────
    #
    # A "chain" is a sequence of clusters that are close enough to
    # stay zoomed in and just pan.  The camera zooms in at the start
    # of the first cluster and zooms out after the last.
    chains: List[List[int]] = []  # each chain is a list of cluster indices
    current_chain: List[int] = [0] if cluster_info else []

    for ci_idx in range(1, len(cluster_info)):
        prev_ci = cluster_info[current_chain[-1]]
        curr_ci = cluster_info[ci_idx]
        # Gap between actual activity end and next activity start
        # (hold period is just camera dwell — doesn't affect chaining)
        gap = curr_ci["start"] - prev_ci["end"]

        if gap < PAN_MERGE_GAP_MS:
            # Close enough → stay zoomed and pan. Every eligible click cluster
            # remains in the same continuous chain; there is no arbitrary cap.
            current_chain.append(ci_idx)
        else:
            chains.append(current_chain)
            current_chain = [ci_idx]
    if current_chain:
        chains.append(current_chain)

    # ── Prevent chain-level overlap ────────────────────────────────
    #
    # A chain's visual span runs from its zoom-in start to its
    # zoom-out end (last cluster end + hold + zoom-out duration).
    # If two consecutive chains overlap, the timeline shows two
    # stacked segments — one hides the other until one is deleted.
    #
    # Fix by reducing hold time first, then pushing the next chain's
    # zoom-in later if still overlapping.
    zoom_out_dur = TRANSITION_MS * 2
    for ch_idx in range(1, len(chains)):
        prev_last_ci = cluster_info[chains[ch_idx - 1][-1]]
        curr_first_ci = cluster_info[chains[ch_idx][0]]

        prev_chain_end = (
            prev_last_ci["end"] + prev_last_ci["hold_ms"] + zoom_out_dur
        )
        overlap = prev_chain_end - curr_first_ci["zoom_in_time"]
        if overlap <= 0:
            continue

        # Strategy 1: reduce previous chain's hold time
        hold_reduction = min(overlap, prev_last_ci["hold_ms"])
        prev_last_ci["hold_ms"] -= int(hold_reduction)
        overlap -= hold_reduction
        if overlap <= 0:
            continue

        # Strategy 2: push current chain's zoom-in later
        curr_first_ci["zoom_in_time"] += int(overlap) + 50

    # ── Generate keyframes from chains ─────────────────────────────
    for chain in chains:
        first_ci = cluster_info[chain[0]]
        last_ci = cluster_info[chain[-1]]

        # Zoom-in at the start of the first cluster
        zoom_in_time = first_ci["zoom_in_time"]
        kf_in = ZoomKeyframe.create(
            timestamp=zoom_in_time,
            zoom=zoom_level,
            x=first_ci["pan_x"],
            y=first_ci["pan_y"],
            duration=TRANSITION_MS,
            reason=first_ci["reason"],
            is_auto_generated=True,
        )
        keyframes.append(kf_in)

        # Pan keyframes between consecutive clusters in the chain.
        # Each pan target is computed relative to the previous cluster's
        # pan position so the camera only moves as far as needed.
        prev_pan_x = first_ci["pan_x"]
        prev_pan_y = first_ci["pan_y"]

        for i in range(1, len(chain)):
            prev_ci = cluster_info[chain[i - 1]]
            curr_ci = cluster_info[chain[i]]

            # Center viewport on the new cluster target, clamped to bounds
            half_vw = 0.5 / zoom_level
            half_vh = 0.5 / zoom_level
            curr_pan_x = max(half_vw, min(1.0 - half_vw, curr_ci["raw_x"]))
            curr_pan_y = max(half_vh, min(1.0 - half_vh, curr_ci["raw_y"]))
            # Store updated pan for the next iteration
            curr_ci["pan_x"] = curr_pan_x
            curr_ci["pan_y"] = curr_pan_y

            # Duration proportional to pan distance, with min/max bounds
            dx = curr_pan_x - prev_pan_x
            dy = curr_pan_y - prev_pan_y
            dist = math.sqrt(dx * dx + dy * dy)
            pan_dur = min(
                PAN_TRANSITION_MAX_MS,
                max(PAN_TRANSITION_MS, int(dist * 1200)),
            )

            # Pan should complete ANTICIPATION_MS before the activity starts
            desired_arrive = curr_ci["start"] - ANTICIPATION_MS
            pan_time = max(
                prev_ci["end"],  # never start before the previous action ends
                desired_arrive - pan_dur,
            )

            # safety: don't go earlier than the chain zoom-in completion
            pan_time = max(zoom_in_time + TRANSITION_MS, pan_time)

            kf_pan = ZoomKeyframe.create(
                timestamp=pan_time,
                zoom=zoom_level,
                x=curr_pan_x,
                y=curr_pan_y,
                duration=pan_dur,
                reason=f"Pan to: {curr_ci['reason'].lower()}",
                is_auto_generated=True,
            )
            keyframes.append(kf_pan)
            prev_pan_x, prev_pan_y = curr_pan_x, curr_pan_y

        # Zoom-out after the last cluster's hold period
        zoom_out_time = last_ci["end"] + last_ci["hold_ms"]
        zoom_out_dur = TRANSITION_MS * 2  # zoom-out is slower

        # Clamp so zoom-out start + transition doesn't exceed duration
        max_zoom_out_start = duration - zoom_out_dur
        if max_zoom_out_start > 0:
            zoom_out_time = min(zoom_out_time, max_zoom_out_start)
        if zoom_out_time > duration:
            zoom_out_time = duration

        kf_out = ZoomKeyframe.create(
            timestamp=zoom_out_time,
            zoom=1.0,
            x=0.5,
            y=0.5,
            duration=zoom_out_dur,
            reason=f"Zoom out after: {last_ci['reason'].lower()}",
            is_auto_generated=True,
        )
        keyframes.append(kf_out)

    keyframes.sort(key=lambda k: k.timestamp)

    # ── 6. Prevent overlap between consecutive zoom sections ─────
    #
    # Extract zoom segments from the sorted keyframe list, then
    # ensure each segment's zoom-out completes before the next
    # segment's zoom-in starts.  This is a safety net in addition
    # to the chain-level overlap prevention above.
    segments: list[tuple[int, int]] = []  # (zoom_in_idx, zoom_out_idx)
    seg_start_idx: int | None = None
    for idx, kf in enumerate(keyframes):
        if kf.zoom > 1.01 and seg_start_idx is None:
            seg_start_idx = idx
        elif kf.zoom <= 1.01 and seg_start_idx is not None:
            segments.append((seg_start_idx, idx))
            seg_start_idx = None

    for s_idx in range(len(segments) - 1):
        _, out_idx = segments[s_idx]
        next_in_idx, _ = segments[s_idx + 1]

        out_kf = keyframes[out_idx]
        in_kf = keyframes[next_in_idx]

        out_end = out_kf.timestamp + out_kf.duration
        if out_end <= in_kf.timestamp:
            continue

        available = in_kf.timestamp - out_kf.timestamp
        if available > 100:
            # Shorten the zoom-out transition
            keyframes[out_idx] = ZoomKeyframe.create(
                timestamp=out_kf.timestamp,
                zoom=out_kf.zoom,
                x=out_kf.x,
                y=out_kf.y,
                duration=max(100, int(available) - 50),
                reason=out_kf.reason,
                is_auto_generated=True,
            )
        else:
            # Not enough room — push the zoom-in later
            keyframes[next_in_idx] = ZoomKeyframe.create(
                timestamp=out_end + 50,
                zoom=in_kf.zoom,
                x=in_kf.x,
                y=in_kf.y,
                duration=in_kf.duration,
                reason=in_kf.reason,
                is_auto_generated=True,
            )

    return keyframes


def detect_chapters(
    mouse_events: List[MousePosition],
    key_events: List[KeyEvent] | None,
    click_events: List[ClickEvent] | None,
    duration_ms: float,
    monitor_rect: Optional[dict] = None,
) -> List["Chapter"]:
    """Auto-detect chapter boundaries based on mouse and click activity.

    Uses heuristic analysis (no AI API calls) to identify scene boundaries:
    - Extended inactivity (>3s idle between mouse/click events)
    - Major mouse position jumps (cursor moves >30% of monitor distance)
    - Significant time gaps between activity bursts

    ``monitor_rect`` should be the captured monitor bounds. Without it, jump
    detection is disabled rather than guessing a screen size from mouse motion.

    Returns a sorted list of Chapter objects with auto-generated names.
    """
    from .models import Chapter

    # Thresholds
    IDLE_GAP_MS = 3000  # 3s idle = scene boundary
    JUMP_THRESHOLD = 0.3  # 30% of screen distance
    MIN_CHAPTER_LENGTH_MS = 5000  # Don't create chapters shorter than 5s

    # Collect all event timestamps
    event_times = []

    # Add mouse activity timestamps (sample every 500ms to reduce noise)
    if mouse_events:
        last_added = -1000
        for m in mouse_events:
            if m.timestamp - last_added >= 500:
                event_times.append(m.timestamp)
                last_added = m.timestamp

    if key_events:
        logger.debug("Ignoring removed keystroke activity during chapter detection")

    # Add click timestamps
    if click_events:
        event_times.extend(c.timestamp for c in click_events)

    if not event_times:
        # No activity — single chapter for entire recording
        return [Chapter(timestamp_ms=0, name="Chapter 1", auto_detected=True)]

    event_times.sort()

    # Detect large gaps in activity
    boundaries = [0]  # Always start with timestamp 0
    for i in range(1, len(event_times)):
        gap = event_times[i] - event_times[i - 1]
        if gap >= IDLE_GAP_MS:
            boundaries.append(int(event_times[i]))

    # Detect major mouse jumps only against the actual capture dimensions.
    # A mouse-track bounding box is not a monitor proxy: a small movement area
    # makes the threshold implausibly tiny and creates false chapters.
    screen_w = float((monitor_rect or {}).get("width", 0) or 0)
    screen_h = float((monitor_rect or {}).get("height", 0) or 0)
    if mouse_events and len(mouse_events) > 1 and screen_w > 0 and screen_h > 0:
        diag = math.hypot(screen_w, screen_h)
        jump_threshold_px = diag * JUMP_THRESHOLD

        for i in range(1, len(mouse_events)):
            prev = mouse_events[i - 1]
            curr = mouse_events[i]
            if curr.resume_boundary:
                continue
            dist = math.hypot(curr.x - prev.x, curr.y - prev.y)
            if dist >= jump_threshold_px:
                # Big jump — potential scene boundary
                boundaries.append(int(curr.timestamp))
    elif mouse_events and len(mouse_events) > 1:
        logger.debug("Skipping jump-based chapter detection without monitor dimensions")

    # Deduplicate and sort boundaries
    boundaries = sorted(set(boundaries))

    # Filter out boundaries that create too-short chapters
    filtered = [boundaries[0]]
    for b in boundaries[1:]:
        if b - filtered[-1] >= MIN_CHAPTER_LENGTH_MS:
            filtered.append(b)

    # Generate Chapter objects with auto-generated names
    chapters = []
    for i, ts in enumerate(filtered):
        # Clamp chapter timestamp to video duration if known
        if duration_ms > 0 and ts > duration_ms:
            continue
        chapters.append(
            Chapter(
                timestamp_ms=ts,
                name=f"Chapter {i + 1}",
                auto_detected=True,
            )
        )

    return chapters
