"""
Hand Distance Volume Control
- Tracks thumb (landmark 4) and index fingertip (landmark 8) with MediaPipe Hands
- Uses normalized distance (0..~0.4) to control system volume (0..100%)
- Smoothing via EMA + simple hysteresis/deadzone to avoid jitter
- Hotkeys: Q (quit), C (calibrate), M (mute toggle), T (volume test)
"""

from __future__ import annotations

import math
import json
import time
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# -------------------- Configuration --------------------
CFG_PATH = Path("./hand_control_config.json")


@dataclass
class Settings:
    # Camera
    cam_index: int = 0

    # Normalized distance bounds (NOT pixels) for mapping 0..100
    min_dist_norm: float = 0.035  # pinch closed
    max_dist_norm: float = 0.22   # pinch open

    # Output range
    out_min: float = 0.0
    out_max: float = 100.0

    # Signal conditioning
    ema_alpha: float = 0.25
    hysteresis_pct: float = 1.0
    deadzone_pct: float = 1.0

    # UI / behavior
    draw: bool = True
    control_system_volume: bool = True
    hand_prefer_right: bool = True


def save_cfg(cfg: Settings) -> None:
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)


def load_cfg() -> Settings:
    if CFG_PATH.exists():
        try:
            raw = json.load(open(CFG_PATH, "r", encoding="utf-8"))
            # Backward compatibility: old keys may be named *px* and >1.0 (pixels).
            # Normalize to modern keys/meaning.
            if "min_dist_px" in raw or "max_dist_px" in raw:
                raw["min_dist_norm"] = 0.035
                raw["max_dist_norm"] = 0.22
                raw.pop("min_dist_px", None)
                raw.pop("max_dist_px", None)
            s = Settings(**{**Settings().__dict__, **raw})
            # Guard if someone put pixel-like numbers by mistake
            if s.min_dist_norm > 1.0 or s.max_dist_norm > 1.0:
                s.min_dist_norm, s.max_dist_norm = 0.035, 0.22
            return s
        except Exception:
            pass
    s = Settings()
    save_cfg(s)
    return s


CFG = load_cfg()


# -------------------- Volume control backends --------------------
class VolumeController:
    """Cross-platform system volume control (Windows/macOS/Linux)."""

    def __init__(self, enable: bool) -> None:
        self.enable = enable
        self.os = platform.system()
        self.muted = False
        self.backend: Optional[str] = None
        self._volume_iface = None  # Windows: POINTER(IAudioEndpointVolume)
        self._init_backend()

    def _init_backend(self) -> None:
        if not self.enable:
            print("Volume: disabled in config.")
            return

        if self.os == "Windows":
            try:
                from ctypes import POINTER, cast
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                # IMPORTANT: cast the COM interface
                self._volume_iface = cast(interface, POINTER(IAudioEndpointVolume))
                # Probe once
                _ = self._volume_iface.GetMasterVolumeLevelScalar()
                self.backend = "pycaw"
                print("Volume backend: pycaw (GetSpeakers)")
            except Exception as e:
                print("Volume backend init failed (pycaw):", e)
                self.backend = None

        elif self.os == "Darwin":
            self.backend = "osascript"
            print("Volume backend: osascript (macOS)")

        elif self.os == "Linux":
            self.backend = "pactl"
            print("Volume backend: pactl (Linux)")

        else:
            self.backend = None
            print("Volume: unsupported OS.")

    def set_volume_percent(self, pct: float) -> None:
        pct = max(0.0, min(100.0, pct))
        if not self.enable or self.backend is None:
            return
        try:
            if self.backend == "pycaw":
                self._volume_iface.SetMasterVolumeLevelScalar(pct / 100.0, None)
            elif self.backend == "osascript":
                subprocess.run(
                    ["osascript", "-e", f"set volume output volume {int(pct)}"],
                    check=False,
                )
            elif self.backend == "pactl":
                subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(pct)}%"],
                    check=False,
                )
        except Exception as e:
            print("set_volume_percent error:", e)

    def get_volume_percent(self) -> Optional[float]:
        if not self.enable or self.backend is None:
            return None
        try:
            if self.backend == "pycaw":
                return float(self._volume_iface.GetMasterVolumeLevelScalar() * 100.0)
            elif self.backend == "osascript":
                out = subprocess.check_output(
                    ["osascript", "-e", "output volume of (get volume settings)"],
                    text=True,
                )
                return float(out.strip())
            elif self.backend == "pactl":
                out = subprocess.check_output(
                    ["pactl", "get-sink-volume", "@DEFAULT_SINK@"], text=True
                )
                import re

                m = re.search(r"(\d+)%", out)
                return float(m.group(1)) if m else None
        except Exception:
            return None

    def toggle_mute(self) -> None:
        self.muted = not self.muted
        if not self.enable or self.backend is None:
            return
        try:
            if self.backend == "pycaw":
                self._volume_iface.SetMute(self.muted, None)
            elif self.backend == "osascript":
                subprocess.run(
                    ["osascript", "-e", f"set volume output muted {str(self.muted).lower()}"],
                    check=False,
                )
            elif self.backend == "pactl":
                subprocess.run(
                    ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
                    check=False,
                )
        except Exception as e:
            print("toggle_mute error:", e)


# -------------------- Utilities --------------------
def map_range(x: float, a1: float, a2: float, b1: float, b2: float) -> float:
    """Clamp x to [a1,a2] and map to [b1,b2]."""
    x = max(min(x, a2), a1)
    if a2 == a1:
        return b1
    return b1 + (x - a1) * (b2 - b1) / (a2 - a1)


class EMA:
    """Exponential Moving Average filter."""

    def __init__(self, alpha: float, init: Optional[float] = None) -> None:
        self.a = alpha
        self.y = init

    def update(self, x: float) -> float:
        self.y = x if self.y is None else (self.a * x + (1 - self.a) * self.y)
        return self.y


# -------------------- MediaPipe setup --------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def pick_hand(results, prefer_right: bool = True):
    """Return the chosen hand landmarks + handedness label ('Right'/'Left') if available."""
    if not results.multi_hand_landmarks:
        return None, None
    if len(results.multi_hand_landmarks) == 1:
        label = None
        if results.multi_handedness:
            label = results.multi_handedness[0].classification[0].label
        return results.multi_hand_landmarks[0], label

    pairs = list(zip(results.multi_hand_landmarks, results.multi_handedness))

    def is_right(h):
        try:
            return h.classification[0].label.lower().startswith("right")
        except Exception:
            return False

    sorted_pairs = (
        sorted(pairs, key=lambda p: (0 if is_right(p[1]) else 1))
        if prefer_right
        else sorted(pairs, key=lambda p: (0 if not is_right(p[1]) else 1))
    )
    label = sorted_pairs[0][1].classification[0].label if sorted_pairs[0][1] else None
    return sorted_pairs[0][0], label


# -------------------- Main loop --------------------
def main() -> None:
    cfg = CFG
    vol = VolumeController(cfg.control_system_volume)
    ema = EMA(cfg.ema_alpha, None)

    # Open camera (CAP_DSHOW improves stability on Windows)
    if platform.system() == "Windows":
        cap = cv2.VideoCapture(cfg.cam_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cfg.cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    # FPS tracking
    t0 = time.time()
    frames = 0
    fps_display = 0

    help_text = "Q: Quit | C: Calibrate | M: Mute | H: Help | T: Test volume"

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        last_applied: Optional[float] = None

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            value_pct: Optional[float] = None
            dist_norm: Optional[float] = None
            handed = None

            if results.multi_hand_landmarks:
                hand_lms, handed = pick_hand(results, cfg.hand_prefer_right)
                if hand_lms:
                    # Landmarks: 4 = thumb tip, 8 = index tip
                    p1 = hand_lms.landmark[4]
                    p2 = hand_lms.landmark[8]
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)

                    # Normalized distance (0..~0.4)
                    dx, dy = (p2.x - p1.x), (p2.y - p1.y)
                    dist_norm = math.hypot(dx, dy)

                    # Map to 0..100 and smooth
                    raw = map_range(
                        dist_norm, cfg.min_dist_norm, cfg.max_dist_norm, cfg.out_min, cfg.out_max
                    )
                    value_pct = ema.update(raw)

                    if cfg.draw:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2),
                        )
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 3)
                        cv2.circle(frame, (x1, y1), 7, (0, 255, 0), -1)
                        cv2.circle(frame, (x2, y2), 7, (0, 255, 0), -1)

            # FPS update every ~0.5s
            frames += 1
            if time.time() - t0 >= 0.5:
                fps_display = int(frames / (time.time() - t0))
                t0 = time.time()
                frames = 0

            # HUD
            if cfg.draw:
                bar_x, bar_y, bar_w, bar_h = 30, 80, 26, 320
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
                if value_pct is not None:
                    filled = int(np.interp(value_pct, [cfg.out_min, cfg.out_max], [bar_h, 0]))
                    cv2.rectangle(
                        frame, (bar_x, bar_y + filled), (bar_x + bar_w, bar_y + bar_h), (0, 255, 120), -1
                    )
                    cv2.putText(
                        frame,
                        f"{int(value_pct)}%",
                        (bar_x - 2, bar_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 120),
                        2,
                    )

                cv2.putText(
                    frame, f"FPS: {fps_display}", (w - 140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                cv2.putText(
                    frame, help_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2
                )
                if handed:
                    cv2.putText(
                        frame, f"Hand: {handed}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 255, 255), 2
                    )
                if dist_norm is not None:
                    cv2.putText(
                        frame, f"Dist: {dist_norm:.3f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
                    )

            # Apply to system volume with hysteresis/deadzone
            if value_pct is not None and cfg.control_system_volume:
                if last_applied is None:
                    last_applied = value_pct
                    vol.set_volume_percent(value_pct)
                else:
                    diff = abs(value_pct - last_applied)
                    thr = max(cfg.hysteresis_pct, cfg.deadzone_pct)
                    if diff >= thr:
                        target = round(value_pct)  # integer %
                        vol.set_volume_percent(target)
                        last_applied = target

            # Show
            cv2.imshow("Hand Distance Volume Control", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("m"), ord("M")):
                vol.toggle_mute()
            elif key in (ord("c"), ord("C")):
                min_meas, max_meas = calibrate(cap, hands)
                if (
                    (min_meas is not None)
                    and (max_meas is not None)
                    and (max_meas > min_meas + 0.01)
                ):
                    cfg.min_dist_norm = float(min_meas)
                    cfg.max_dist_norm = float(max_meas)
                    save_cfg(cfg)
                    ema.y = None
                    last_applied = None
            elif key in (ord("t"), ord("T")):
                # quick volume jump test: 10% <-> 90%
                curr = vol.get_volume_percent()
                print(f"[TEST] Current volume: {curr}%")
                vol.set_volume_percent(10 if (curr is None or curr > 50) else 90)
                time.sleep(0.2)
                print(f"[TEST] New volume: {vol.get_volume_percent()}%")

    cap.release()
    cv2.destroyAllWindows()


def calibrate(cap, hands, seconds_each: float = 2.0) -> Tuple[Optional[float], Optional[float]]:
    """
    Two phases with normalized distance:
      1) Pinch closed  (min)
      2) Pinch open    (max)
    Hold each pose for ~seconds_each.
    """
    def measure(label: str, want_min: bool = True) -> Optional[float]:
        t_start = time.time()
        best: Optional[float] = None
        frame = None
        while time.time() - t_start < seconds_each:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                hand, _ = pick_hand(res, CFG.hand_prefer_right)
                if hand:
                    p1 = hand.landmark[4]
                    p2 = hand.landmark[8]
                    dx, dy = (p2.x - p1.x), (p2.y - p1.y)
                    dist = math.hypot(dx, dy)  # normalized
                    best = dist if best is None else (min(best, dist) if want_min else max(best, dist))

                    # draw overlay
                    x1, y1 = int(p1.x * w), int(p1.y * h)
                    x2, y2 = int(p2.x * w), int(p2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 255), 3)
                    cv2.circle(frame, (x1, y1), 7, (0, 255, 0), -1)
                    cv2.circle(frame, (x2, y2), 7, (0, 255, 0), -1)
                    cv2.putText(
                        frame, f"{label}: {best:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2
                    )

            if frame is not None:
                cv2.putText(
                    frame, "Calibrating... Hold the pose", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    return None
        return best

    min_norm = measure("MIN (pinch closed)", want_min=True)
    if min_norm is None:
        cv2.destroyWindow("Calibration")
        return None, None
    max_norm = measure("MAX (pinch open)", want_min=False)
    cv2.destroyWindow("Calibration")
    if max_norm is None:
        return None, None
    return min_norm, max_norm


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        raise
