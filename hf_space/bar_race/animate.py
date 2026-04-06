"""Build keyframes from normalised data and interpolate between them.

Each *keyframe* corresponds to one unique date in the data.  Between
consecutive keyframes we generate ``frames_per_step`` intermediate frames.

Value interpolation is **linear** so bar widths grow at a constant rate
with zero micro-pauses.  Only **rank changes** (vertical position swaps)
use cubic ease-in-out for a smooth sliding feel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------

def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out: smooth start and stop in [0, 1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out: fast start, smooth deceleration in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BarState:
    """Snapshot of a single bar in one frame."""

    player: str
    value: float
    rank: float  # may be fractional during interpolation
    team: str = ""

    # Visual state flags
    entering: bool = False
    exiting: bool = False
    alpha: float = 1.0  # 0→1 entering, 1→0 exiting

    # Overlay data
    tenure: int = 0           # keyframes this player has been in top N


@dataclass
class FrameState:
    """Complete state for a single rendered frame."""

    bars: list[BarState]
    date_label: str
    progress: float  # 0‥1 overall animation progress
    max_value: float

    # Overlay state (populated by post-processing).
    leader: str = ""                        # current #1 player name
    reign_history: list[str] = field(default_factory=list)
    gap_pct: float = 0.0
    show_gap: bool = False
    players_seen: int = 0

    # Bottom panel columns
    tenure_leaders: list[str] = field(default_factory=list)    # top 3 by tenure
    milestone_records: list[str] = field(default_factory=list) # fastest-to records


@dataclass
class Keyframe:
    """One per unique date — raw snapshot from the data."""

    date: datetime
    entries: list[BarState]  # sorted descending by value
    label: str = ""  # original label (e.g. "18" for age-based data)


# ---------------------------------------------------------------------------
# Keyframe building
# ---------------------------------------------------------------------------

def build_keyframes(df: pd.DataFrame, top_n: int = 10) -> list[Keyframe]:
    """Build one :class:`Keyframe` per unique date, keeping *top_n* entries."""
    label_map: dict = df.attrs.get("date_label_map", {})
    keyframes: list[Keyframe] = []

    for date, grp in df.groupby("date"):
        # Aggregate duplicates (same player on same date).
        agg = grp.groupby("player", as_index=False).agg(
            {"value": "max", "team": "first"}
        )
        agg = agg.sort_values("value", ascending=False).head(top_n).reset_index(drop=True)

        entries = [
            BarState(
                player=row["player"],
                value=float(row["value"]),
                rank=float(i),
                team=str(row["team"]),
            )
            for i, row in agg.iterrows()
        ]
        ts = pd.Timestamp(date).to_pydatetime()
        label = label_map.get(pd.Timestamp(date), "")
        keyframes.append(Keyframe(date=ts, entries=entries, label=label))

    keyframes.sort(key=lambda k: k.date)
    return keyframes


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interpolate_date(
    d1: datetime,
    d2: datetime,
    t: float,
    label_a: str = "",
    label_b: str = "",
) -> str:
    """Interpolate between two dates and return a label string.

    For numeric labels (ages, years) the label stays fixed for the
    entire segment so it doesn't flicker mid-transition.
    """
    if label_a and label_b:
        try:
            float(label_a)
            float(label_b)
            # Keep current label for entire segment; only show next
            # label on the very last frame (t=1.0) so the transition
            # is seamless with the next segment's first frame.
            return label_a if t < 1.0 else label_b
        except ValueError:
            return label_a if t < 0.5 else label_b

    delta = (d2 - d1).total_seconds()
    result = d1 + timedelta(seconds=delta * t)
    return result.strftime("%b %d, %Y")


# ---------------------------------------------------------------------------
# Entry / exit timing
# ---------------------------------------------------------------------------

_ENTRY_FRACTION = 0.42   # entry uses last ~42 % of the step (~0.5 s at 60 fps)
_EXIT_FRACTION = 0.42    # exit uses first ~42 % of the step (~0.5 s at 60 fps)


def interpolate_frames(
    keyframes: list[Keyframe],
    total_frames: int,
    top_n: int = 10,
) -> list[FrameState]:
    """Generate *total_frames* :class:`FrameState` objects by interpolating.

    Bars that appear in the next keyframe but not the current one *enter*
    from below; bars disappearing *exit* downward.
    """
    if len(keyframes) < 2:
        raise ValueError("Need at least 2 keyframes to interpolate.")

    n_steps = len(keyframes) - 1
    frames_per_step = max(1, total_frames // n_steps)

    # Ensure minimum frames per step for smooth animation.
    _MIN_FPT = 20
    if n_steps <= 50 and frames_per_step < _MIN_FPT:
        frames_per_step = _MIN_FPT

    frames: list[FrameState] = []

    for step_idx in range(n_steps):
        kf_a = keyframes[step_idx]
        kf_b = keyframes[step_idx + 1]

        map_a = {b.player: b for b in kf_a.entries}
        map_b = {b.player: b for b in kf_b.entries}
        all_players = list(dict.fromkeys(
            [b.player for b in kf_a.entries] + [b.player for b in kf_b.entries]
        ))

        if step_idx < n_steps - 1:
            n_frames = frames_per_step
        else:
            # Last step: use at least frames_per_step frames.
            n_frames = max(frames_per_step, total_frames - len(frames))
        n_frames = max(n_frames, 1)

        for fi in range(n_frames):
            raw_t = fi / max(n_frames - 1, 1)
            t_rank = ease_in_out_cubic(raw_t)
            t_value = raw_t  # LINEAR — constant bar growth, no micro-pauses

            bars: list[BarState] = []

            for player in all_players:
                a = map_a.get(player)
                b = map_b.get(player)

                entering = a is None
                exiting = b is None
                team = (a.team if a else b.team) if (a or b) else ""  # type: ignore[union-attr]

                if entering:
                    # --- slide in from bottom with correct value ---
                    entry_start = 1.0 - _ENTRY_FRACTION
                    if raw_t < entry_start:
                        continue  # not visible yet
                    et = ease_in_out_cubic(
                        (raw_t - entry_start) / _ENTRY_FRACTION
                    )
                    rank = _lerp(float(top_n) + 0.5, b.rank, et)
                    value = b.value  # show correct value immediately
                    alpha = min(1.0, et * 2.0)

                elif exiting:
                    # --- slide out downward smoothly ---
                    if raw_t > _EXIT_FRACTION:
                        continue  # already gone
                    et = ease_in_out_cubic(raw_t / _EXIT_FRACTION)
                    rank = _lerp(a.rank, float(top_n) + 0.5, et)
                    value = a.value  # keep actual value while sliding out
                    alpha = max(0.0, 1.0 - et * 2.0)

                else:
                    # --- normal interpolation ---
                    rank = _lerp(a.rank, b.rank, t_rank)
                    value = _lerp(a.value, b.value, t_value)
                    alpha = 1.0

                bars.append(BarState(
                    player=player,
                    value=value,
                    rank=rank,
                    team=team,
                    entering=entering,
                    exiting=exiting,
                    alpha=alpha,
                ))

            # Keep only visible bars (rank < top_n + 1 to allow exit/entry).
            bars = [b for b in bars if b.rank < top_n + 1]
            bars.sort(key=lambda b: b.rank)

            date_label = _interpolate_date(
                kf_a.date, kf_b.date, raw_t,
                label_a=kf_a.label, label_b=kf_b.label,
            )

            max_value = max((b.value for b in bars), default=1.0)

            overall_progress = (step_idx + raw_t) / n_steps

            frames.append(FrameState(
                bars=bars,
                date_label=date_label,
                progress=overall_progress,
                max_value=max_value,
            ))

    return frames


# ---------------------------------------------------------------------------
# Leader tracking (post-processing pass)
# ---------------------------------------------------------------------------

@dataclass
class ReignPeriod:
    """Track a #1 leader's reign."""
    player: str
    start_label: str
    end_label: str = ""


def _abbrev(name: str, max_len: int = 18) -> str:
    """Abbreviate long names: 'Kareem Abdul-Jabbar' → 'K. Abdul-Jabbar'."""
    if len(name) <= max_len:
        return name
    parts = name.split()
    if len(parts) >= 2:
        short = f"{parts[0][0]}. {' '.join(parts[1:])}"
        if len(short) <= max_len + 2:
            return short
        return short[:max_len] + "…"
    return name[:max_len] + "…"


def _detect_milestones(max_val: float) -> list[int]:
    """Choose milestone thresholds based on data scale."""
    if max_val < 2_000:
        return [100, 200, 300, 400, 500, 600, 700, 800, 900,
                1_000, 1_100, 1_200, 1_300, 1_400, 1_500, 1_600]
    if max_val < 10_000:
        return [500, 1_000, 2_000, 3_000, 5_000, 7_500]
    return [5_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000]


@dataclass
class SoundEvent:
    """A sound cue at a specific frame index."""
    frame: int
    kind: str    # "whoosh", "ding", "boom"
    intensity: float = 1.0  # 0‥1


def populate_leader_overlays(
    frames: list[FrameState],
    fps: int = 60,
    gap_threshold: float = 0.10,
    gap_hysteresis: float = 0.08,
) -> tuple[list[ReignPeriod], list[SoundEvent]]:
    """Fill in leader/reign/gap/tenure/milestone fields. Returns (reigns, sound_events)."""
    if not frames:
        return [], []

    reigns: list[ReignPeriod] = []
    sound_events: list[SoundEvent] = []
    prev_leader = ""
    gap_active = False
    all_players_seen: set[str] = set()

    # Tenure: count keyframes per player (increment once per step, not per frame).
    tenure_counts: dict[str, int] = {}
    n_steps = 0
    prev_date_label = ""

    # Milestones: track fastest player to each milestone.
    max_val = max((b.value for f in frames for b in f.bars), default=0)
    milestones = _detect_milestones(max_val)
    # "First to" milestones: first player ever to cross each threshold.
    first_to: dict[int, tuple[str, str]] = {}  # m → (player, date_label)
    reached: dict[str, set[int]] = {}          # player → set of passed milestones
    first_appearance: dict[str, int] = {}      # player → keyframe index of first appearance

    # Track previous ranks for whoosh detection.
    prev_ranks: dict[str, float] = {}

    for i, f in enumerate(frames):
        # Track unique players.
        for b in f.bars:
            all_players_seen.add(b.player)
        f.players_seen = len(all_players_seen)

        # Tenure + milestones: only update at keyframe boundaries.
        is_new_keyframe = (f.date_label != prev_date_label)
        if is_new_keyframe:
            n_steps += 1
            for b in f.bars:
                tenure_counts[b.player] = tenure_counts.get(b.player, 0) + 1
                # Track first appearance.
                if b.player not in first_appearance:
                    first_appearance[b.player] = n_steps

            # Milestones: check at keyframe boundaries only.
            for b in f.bars:
                player_reached = reached.setdefault(b.player, set())
                for m in milestones:
                    if m not in player_reached and b.value >= m:
                        player_reached.add(m)
                        sound_events.append(SoundEvent(frame=i, kind="ding"))
                        # Record the first player ever to reach this milestone.
                        if m not in first_to:
                            first_to[m] = (b.player, f.date_label)

            prev_date_label = f.date_label

        # Apply tenure to bars.
        for b in f.bars:
            b.tenure = tenure_counts.get(b.player, 0)

        # Build milestone records column (most recent milestones first).
        if first_to:
            records: list[str] = []
            for m in sorted(first_to, reverse=True):
                player, when = first_to[m]
                records.append(f"{m:,}: {_abbrev(player)} ({when})")
            f.milestone_records = records[:3]

        # Build tenure leaderboard column (top 3 by tenure).
        if tenure_counts:
            top_tenure = sorted(tenure_counts.items(), key=lambda x: -x[1])[:3]
            f.tenure_leaders = [
                f"{_abbrev(p)}: {t}"
                for p, t in top_tenure
            ]

        # Determine current leader.
        leader = ""
        leader_val = 0.0
        second_val = 0.0
        sorted_bars = sorted(f.bars, key=lambda b: b.rank)
        if sorted_bars:
            leader = sorted_bars[0].player
            leader_val = sorted_bars[0].value
        if len(sorted_bars) > 1:
            second_val = sorted_bars[1].value

        f.leader = leader

        # Detect leader change.
        if leader and leader != prev_leader and prev_leader != "":
            if reigns:
                reigns[-1].end_label = f.date_label
            reigns.append(ReignPeriod(player=leader, start_label=f.date_label))
            sound_events.append(SoundEvent(frame=i, kind="boom"))
        elif leader and not reigns:
            reigns.append(ReignPeriod(player=leader, start_label=f.date_label))

        # Whoosh: detect 3+ position gains.
        for b in f.bars:
            pr = prev_ranks.get(b.player)
            if pr is not None and pr - b.rank >= 3.0:
                sound_events.append(SoundEvent(
                    frame=i, kind="whoosh",
                    intensity=min(1.0, (pr - b.rank) / 5.0),
                ))
        prev_ranks = {b.player: b.rank for b in f.bars}
        prev_leader = leader

        # Reign history log (abbreviated names).
        if reigns:
            history: list[str] = []
            for r in reversed(reigns[-4:]):
                end = r.end_label if r.end_label else f.date_label
                history.append(f"{_abbrev(r.player)} ({r.start_label}\u2014{end})")
            f.reign_history = history

        # Gap percentage.
        if leader_val > 0 and second_val > 0:
            gap = (leader_val - second_val) / second_val
            f.gap_pct = gap
            if gap_active:
                gap_active = gap >= gap_hysteresis
            else:
                gap_active = gap >= gap_threshold
            f.show_gap = gap_active
        else:
            f.gap_pct = 0.0
            f.show_gap = False

    # Close final reign.
    if reigns and frames:
        reigns[-1].end_label = frames[-1].date_label

    return reigns, sound_events
