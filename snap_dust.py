#!/usr/bin/env python3
"""
SNAP TO DUST - a Thanos-snap effect for your webcam in Google Meet (or Zoom / Teams).

  1. When it starts, it photographs your room WITHOUT you (step out of view when it beeps).
  2. It sends your camera to a virtual webcam ("OBS Virtual Camera") - pick that in Meet.
  3. Snap your fingers where the camera can see your hand -> you crumble into dust that blows
     away, and the exact spot where you were fills in with the room behind you. The rest of
     the picture stays live. Snap again -> the dust flies back and you return.

The snap is SEEN by the camera (thumb + middle finger together, then snap) and also HEARD
by the microphone when the mic lets the sound through.
It keeps checking that the room still matches the photo and asks you to retake it if not.

Hotkeys (work even while Chrome / Meet is focused):
    F8  snap / un-snap          F9  retake the empty-room photo
Preview window keys:  SPACE snap   B retake room   M mic on/off   H hide text   Q or Esc quit

Run  python snap_dust.py --help  for all options.
"""
import argparse
import math
import os
import sys
import threading
import time
import urllib.request
import zipfile
from collections import deque

import cv2
import numpy as np

from hand_tracker import HandTracker

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
SEG_MODEL = os.path.join(MODEL_DIR, "selfie_segmenter.tflite")
PALM_MODEL = os.path.join(MODEL_DIR, "hand_detector.tflite")
LANDMARK_MODEL = os.path.join(MODEL_DIR, "hand_landmarks.tflite")
SEG_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
           "selfie_segmenter/float16/latest/selfie_segmenter.tflite")
HAND_BUNDLE_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/latest/hand_landmarker.task")
ROOM_FILE = os.path.join(HERE, "empty_room.png")
WINDOW = "Snap to Dust - preview (this is what Meet sees)"

ASH = np.array([38, 44, 52], np.float32)        # BGR: dark grey-brown "Thanos dust"
ASH_AGED = np.array([62, 67, 74], np.float32)   # dust lightens a little as it drifts

K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K_FILL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))   # ~10 px extra around you


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def beep(freq=880, ms=120, times=1, gap=0.06):
    """Audible cues - you can't see the screen while you're out of the shot."""
    if sys.platform != "win32":
        return

    def run():
        import winsound
        for _ in range(times):
            winsound.Beep(int(freq), int(ms))
            time.sleep(gap)
    threading.Thread(target=run, daemon=True).start()


# --------------------------------------------------------------------------- models

def ensure_models(need_hand):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(SEG_MODEL):
        log("Downloading the person-segmentation model (250 KB) ...")
        urllib.request.urlretrieve(SEG_URL, SEG_MODEL + ".part")
        os.replace(SEG_MODEL + ".part", SEG_MODEL)
    if need_hand and not (os.path.exists(PALM_MODEL) and os.path.exists(LANDMARK_MODEL)):
        log("Downloading the hand-tracking models (8 MB) ...")
        tmp = os.path.join(MODEL_DIR, "hand_landmarker.task")
        urllib.request.urlretrieve(HAND_BUNDLE_URL, tmp)
        with zipfile.ZipFile(tmp) as z:
            for name, dest in (("hand_detector.tflite", PALM_MODEL),
                               ("hand_landmarks_detector.tflite", LANDMARK_MODEL)):
                with open(dest, "wb") as f:
                    f.write(z.read(name))
        os.remove(tmp)


class PersonSegmenter:
    """MediaPipe selfie segmenter, run through OpenCV's DNN module. Returns 0..1 'is person'."""

    def __init__(self, path):
        self.net = cv2.dnn.readNet(path)

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, 1 / 255.0, (256, 256), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward().reshape(256, 256)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- camera

class Camera:
    """Grabs webcam frames on a background thread so we always process the newest one."""

    def __init__(self, index, width, height, fps):
        indices = [index] if index is not None else [0, 1, 2]
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
        fallback = None
        self.cap = None
        for idx in indices:
            for be in backends:
                cap = cv2.VideoCapture(idx, be)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                frame = None
                for _ in range(8):                      # let it warm up
                    ok, f = cap.read()
                    if ok and f is not None:
                        frame = f
                if frame is None:
                    cap.release()
                    continue
                # Skip infrared (black & white) cameras like "HP IR Camera".
                colorful = np.abs(frame[..., 0].astype(np.int16) - frame[..., 2]).mean() > 1.0
                if colorful or index is not None:
                    self.cap, self.index = cap, idx
                    break
                if fallback is None:
                    fallback = (cap, idx, frame)
                else:
                    cap.release()
            if self.cap is not None:
                break
        if self.cap is None and fallback is not None:
            self.cap, self.index, frame = fallback
        elif fallback is not None and fallback[0] is not self.cap:
            fallback[0].release()
        if self.cap is None:
            raise RuntimeError("Could not open a webcam. Close other apps using it "
                               "(e.g. switch Meet to 'OBS Virtual Camera' first) or try --camera 1")
        self.h, self.w = frame.shape[:2]
        self.frame, self.seq = frame, 1
        self.cond = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            if frame.shape[:2] != (self.h, self.w):
                frame = cv2.resize(frame, (self.w, self.h))
            with self.cond:
                self.frame = frame
                self.seq += 1
                self.cond.notify_all()

    def read(self, last_seq, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout)
            return self.frame, self.seq

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


# --------------------------------------------------------------------------- snap listener

class SnapDetector:
    """Listens to the microphone for a finger snap: a sudden, loud burst of high-pitched
    sound (2-9 kHz) that dies away within ~50 ms. Talking, music and humming don't do that."""

    BLOCK = 256
    # Some laptop mics (e.g. Intel Smart Sound) run noise filtering that outputs pure silence
    # between sounds and turns snaps way down. So nothing here depends on absolute loudness -
    # only on how much a sound sticks out, how sharp it is and how fast it dies away.
    FLOOR_MIN = 1e-6                           # treat digital silence as about -100 dBFS

    def __init__(self, sensitivity=5, device=None, samplerate=None, start=True):
        s = float(np.clip(sensitivity, 1, 10))
        self.th_db = 26.0 - 2.0 * s            # how far above the room's background noise
        self.min_peak = 0.0
        self.stream = None
        self.name = "test"
        if start:
            import sounddevice as sd
            info = sd.query_devices(device, "input")
            self.name = info["name"]
            samplerate = samplerate or int(info["default_samplerate"]) or 48000
        self.sr = int(samplerate or 48000)
        n = self.BLOCK * 2
        self.win = np.hanning(n).astype(np.float32)
        f = np.fft.rfftfreq(n, 1.0 / self.sr)
        self.hf_band = (f >= 2000) & (f <= 9000)
        self.lf_band = (f >= 80) & (f <= 1200)
        self.prev = np.zeros(self.BLOCK, np.float32)
        self.buf = np.zeros(0, np.float32)
        self.hist = deque(maxlen=300)
        self.recent = deque(maxlen=4)
        self.floor = None
        self.pending = None
        self.blocks = 0
        self.cool_until = 0.0
        self.level_db = 0.0
        self.enabled = True
        self.last_info = ""
        self._fired = 0
        self._lock = threading.Lock()
        if start:
            self.stream = sd.InputStream(device=device, channels=1, samplerate=self.sr,
                                         blocksize=self.BLOCK, dtype="float32", callback=self._cb)
            self.stream.start()

    def _cb(self, indata, frames, tinfo, status):
        x = indata[:, 0]
        if len(x) == self.BLOCK and self.buf.size == 0:
            self.process(x.copy())
            return
        self.buf = np.concatenate([self.buf, x])
        while self.buf.size >= self.BLOCK:
            self.process(self.buf[:self.BLOCK].copy())
            self.buf = self.buf[self.BLOCK:]

    def process(self, x, now=None):
        now = time.monotonic() if now is None else now
        seg = np.concatenate([self.prev, x])
        self.prev = x
        spec = np.abs(np.fft.rfft(seg * self.win)) ** 2
        hf = float(spec[self.hf_band].sum()) + 1e-12
        lf = float(spec[self.lf_band].sum()) + 1e-12
        peak = float(np.abs(x).max())
        self.blocks += 1
        self.hist.append(hf)
        if self.floor is None or self.blocks % 25 == 0:
            self.floor = max(float(np.percentile(self.hist, 40)), self.FLOOR_MIN)
        db = 10.0 * math.log10(hf / self.floor)
        self.level_db = max(db, self.level_db - 0.6)

        if self.pending is not None:
            p = self.pending
            p["e"].append(hf)
            p["peak"] = max(p["peak"], peak)
            if len(p["e"]) <= 3:
                p["lf"] = max(p["lf"], lf)
            if len(p["e"]) >= 12:
                onset = max(p["e"][:3])
                decay = (sum(p["e"][8:12]) / 4.0) / onset
                hilo = onset / p["lf"]
                ok = decay < 0.15 and p["peak"] >= self.min_peak and hilo > 0.35
                self.last_info = (f"{p['db']:.0f} dB over room noise, peak {p['peak']:.2f}, "
                                  f"hi/lo {hilo:.1f}, decay {decay:.2f} -> "
                                  + ("SNAP!" if ok else "not a snap"))
                if ok:
                    with self._lock:
                        self._fired += 1
                    self.cool_until = now + 0.8
                self.pending = None
        elif self.enabled and now > self.cool_until and self.blocks > 60:
            base = sum(self.recent) / len(self.recent) if self.recent else hf
            if db > self.th_db and hf > 6.0 * base:
                self.pending = {"e": [hf], "peak": peak, "lf": lf, "db": db}
        self.recent.append(hf)

    def poll(self):
        with self._lock:
            n, self._fired = self._fired, 0
        return n > 0

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


# --------------------------------------------------------------------------- what's behind you

class Aligner:
    """Works out how the camera's current view relates to the room picture (shift, tilt,
    perspective) by matching distinctive points - furniture edges, wall panel lines, patterns -
    on the part of the room that's visible around you. Plain walls give few such points, so
    the picture's contrast is boosted first (that turns ~20 usable points into ~1000)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        try:
            self.det = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.01)
            self.bf = cv2.BFMatcher(cv2.NORM_L2)
        except AttributeError:
            self.det = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=6, edgeThreshold=15,
                                      fastThreshold=8)
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.lock = threading.Lock()
        self.kp, self.des = [], None

    def features(self, gray, mask=None):
        return self.det.detectAndCompute(self.clahe.apply(gray), mask)

    def set_reference(self, gray, mask=None):
        kp, des = self.features(gray, mask)
        with self.lock:
            self.kp, self.des = kp, des

    def estimate(self, gray, mask):
        """Homography taking reference coordinates -> this image's, and how many points agreed.
        Returns (None, n) when it isn't sure."""
        with self.lock:
            ref_kp, ref_des = self.kp, self.des
        if ref_des is None or len(ref_kp) < 15:
            return None, 0
        kp, des = self.features(gray, mask)
        if des is None or len(kp) < 15:
            return None, 0
        pairs = self.bf.knnMatch(ref_des, des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 15:
            return None, len(good)
        src = np.float32([ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 2.0, maxIters=3000, confidence=0.995)
        if Hm is None:
            return None, 0
        n = int(inl.sum())
        if n < 15 or n < 0.2 * len(good) or not self.plausible(Hm):
            return None, n
        return Hm, n

    def plausible(self, Hm):
        c = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]]).reshape(-1, 1, 2)
        m = cv2.perspectiveTransform(c, Hm).reshape(-1, 2)
        area = cv2.contourArea(m) / float(self.w * self.h)
        shift = float(np.linalg.norm(m.mean(0) - c.reshape(-1, 2).mean(0))) / self.w
        return 0.6 < area < 1.6 and shift < 0.35 and cv2.isContourConvex(m.reshape(-1, 1, 2))

    def corner_shift(self, H1, H2=None):
        """Biggest corner movement (pixels) between two homographies (or vs. no change)."""
        c = np.float32([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]]).reshape(-1, 1, 2)
        a = cv2.perspectiveTransform(c, H1)
        b = c if H2 is None else cv2.perspectiveTransform(c, H2)
        return float(np.abs(a - b).max())


def visible_room(mask, grow):
    """uint8 255 where the room is clearly visible (not you, not right next to you)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
    return cv2.erode(((mask < 0.03) * 255).astype(np.uint8), k)


class Background:
    """What the room behind you looks like, lined up with the camera's current view.

    - The room photo is warped to fit the current view exactly (the laptop may have moved a
      little since the photo), re-fitted twice a second and right when you snap.
    - Its brightness / colour is matched locally to the live picture all around you, and that
      correction is carried smoothly into the area you're hiding, so the patch blends in.
    - It also notices when the photo doesn't match the room any more.
    The heavy lifting runs on a background thread; the video loop just swaps in the result."""

    def __init__(self, w, h, sw, sh):
        self.w, self.h, self.sw, self.sh = w, h, sw, sh
        self.qw, self.qh = sw // 2, sh // 2
        self.tw, self.th = 160, max(2, int(round(160 * h / w)))
        self.aligner = Aligner(sw, sh)
        self.ready = False
        self.mismatch = False
        self.aligned = False
        self.info = ""
        self.banner = ""
        self._bad = self._good = 0
        self._job = self._result = None
        self.last_fit_t, self.fit_resid = -1e9, 0.0
        rng = np.random.default_rng()
        self.noise = []
        for _ in range(4):
            n = rng.normal(0, 1.8, (h, w)).round()
            pos = np.clip(n, 0, 255).astype(np.uint8)
            neg = np.clip(-n, 0, 255).astype(np.uint8)
            self.noise.append((cv2.merge([pos] * 3), cv2.merge([neg] * 3)))

    def set_room(self, img):
        self._wait_job()
        if img.shape[:2] != (self.h, self.w):
            img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.room_src = np.ascontiguousarray(img)
        small = cv2.resize(img, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
        self.aligner.set_reference(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        self.src_q_gray = cv2.cvtColor(cv2.resize(img, (self.qw, self.qh), interpolation=cv2.INTER_AREA),
                                       cv2.COLOR_BGR2GRAY).astype(np.float32)
        self._g_started = False
        self.mismatch, self.aligned, self._bad, self._good, self.info = False, False, 0, 0, ""
        self._install(self._build(np.eye(3), np.ones((self.th, self.tw, 3), np.float32)))
        self.ready = True

    # -- building the matched room picture (heavy: background thread) -----------------------
    def _warp(self, Hs):
        if np.allclose(Hs, np.eye(3)):
            room = self.room_src.copy()
            covered = np.ones((self.qh, self.qw), bool)
        else:
            S = np.diag([self.w / self.sw, self.h / self.sh, 1.0])
            room = cv2.warpPerspective(self.room_src, S @ Hs @ np.linalg.inv(S), (self.w, self.h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            # which part of the current view the photo actually covers
            Q = np.diag([self.qw / self.sw, self.qh / self.sh, 1.0])
            covered = cv2.warpPerspective(np.ones((self.qh, self.qw), np.uint8), Q @ Hs @ np.linalg.inv(Q),
                                          (self.qw, self.qh), flags=cv2.INTER_NEAREST) > 0
        return {"Hs": Hs, "room": room, "covered_q": covered,
                "room_tiny": cv2.resize(room, (self.tw, self.th), interpolation=cv2.INTER_AREA).astype(np.float32),
                "room_q_gray": cv2.cvtColor(cv2.resize(room, (self.qw, self.qh), interpolation=cv2.INTER_AREA),
                                            cv2.COLOR_BGR2GRAY).astype(np.float32)}

    def _build(self, Hs, G, warped=None):
        b = dict(warped) if warped is not None else self._warp(Hs)
        g = cv2.resize(G, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        room_m = cv2.multiply(b["room"], g, dtype=cv2.CV_8U)               # colour-matched
        b.update(G=G, room_m=room_m,
                 room_q=cv2.resize(room_m, (self.qw, self.qh), interpolation=cv2.INTER_AREA),
                 grainy=[cv2.subtract(cv2.add(room_m, p), n) for p, n in self.noise])
        return b

    def _install(self, b):
        for k, v in b.items():
            setattr(self, k, v)

    def _gain_field(self, small, you, room_tiny, G_old, started):
        """Match brightness/colour locally: measure live/photo all around you, then spread that
        smoothly into the hidden area (coarse first, then finer where there's data)."""
        tiny = cv2.resize(small, (self.tw, self.th), interpolation=cv2.INTER_AREA).astype(np.float32)
        c = cv2.resize(you, (self.tw, self.th), interpolation=cv2.INTER_AREA)
        vis = cv2.erode((c < 0.03).astype(np.uint8), K5).astype(np.float32)
        if vis.mean() < 0.03:
            return G_old, started
        ratio = (tiny + 4.0) / (room_tiny + 4.0)
        glob = np.median(ratio[vis > 0], axis=0)
        ratio = np.clip(ratio, glob * 0.6, glob * 1.6)
        G = np.empty_like(ratio)
        G[:] = glob
        for sigma in (40, 14, 5):
            den = cv2.GaussianBlur(vis, (0, 0), sigma)
            num = cv2.GaussianBlur(ratio * vis[..., None], (0, 0), sigma)
            est = num / np.maximum(den, 1e-4)[..., None]
            wgt = np.clip(den / 0.35, 0, 1)[..., None]
            G = G * (1 - wgt) + est * wgt
        G = G.astype(np.float32)
        return (G if not started else 0.6 * G_old + 0.4 * G), True

    # -- lining the photo up with the camera ------------------------------------------------
    def _estimate(self, small, you):
        """Where does the photo sit in this view? Returns ('none' | 'fail' | 'ok', homography)."""
        vis = visible_room(you, 9)
        if (vis > 0).mean() < 0.05:
            return "none", None                       # you fill the view - nothing to go on
        Hm, _ = self.aligner.estimate(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), vis)
        if Hm is None:
            Hm = self._ecc_estimate(small, you)      # few features, e.g. bare walls
        return ("ok", Hm) if Hm is not None else ("fail", None)

    def _ecc_estimate(self, small, you):
        live = cv2.cvtColor(cv2.resize(small, (self.qw, self.qh), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        vis = visible_room(cv2.resize(you, (self.qw, self.qh), interpolation=cv2.INTER_AREA), 7)
        try:
            _, warp = cv2.findTransformECC(self.src_q_gray, live, np.eye(2, 3, dtype=np.float32),
                                           cv2.MOTION_EUCLIDEAN,
                                           (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-4),
                                           vis, 5)
        except cv2.error:
            return None
        S = np.diag([self.sw / self.qw, self.sh / self.qh, 1.0])
        Hm = S @ np.vstack([warp, [0, 0, 1]]).astype(np.float64) @ np.linalg.inv(S)
        if not self.aligner.plausible(Hm) or self.aligner.corner_shift(Hm) > 0.15 * self.sw:
            return None
        return Hm

    def _work(self, small, you, cur, G, started, fit=True):
        """One refresh: re-fit the photo (if asked - it's the slow part), re-match the
        lighting, rebuild if needed."""
        st, Hm = self._estimate(small, you) if fit else ("skip", None)
        Hs = cur["Hs"]
        warped = cur
        if st == "ok" and self.aligner.corner_shift(Hm, Hs) > 0.6:
            warped = self._warp(Hm)
        G2, started2 = self._gain_field(small, you, warped["room_tiny"], G, started)
        rebuild = warped is not cur or np.abs(G2 - G).max() > 0.004
        return st, (self._build(warped["Hs"], G2, warped) if rebuild else None), started2

    def _snapshot(self):
        return {k: getattr(self, k) for k in ("Hs", "room", "covered_q", "room_tiny", "room_q_gray")}

    def _wait_job(self):
        if self._job is not None:
            self._job.join()
        self._job = self._result = None

    def align(self, small, you):
        """Re-fit right now (used at the moment you snap). True = fitted, False = couldn't
        match, None = too little of the room visible to try."""
        self._wait_job()
        st, b, self._g_started = self._work(small, you, self._snapshot(), self.G, self._g_started)
        if b is not None:
            self._install(b)
        return {"none": None, "fail": False, "ok": True}[st]

    def residual(self, small, you):
        """Cheap (~1 ms): how different does the visible room look from the lined-up picture?
        Jumps when the camera moves, so we know when a full re-fit is worth it."""
        live = cv2.cvtColor(cv2.resize(small, (self.tw, self.th), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref = cv2.cvtColor(self.room_tiny.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = visible_room(cv2.resize(you, (self.tw, self.th), interpolation=cv2.INTER_AREA), 3) > 0
        if m.mean() < 0.05:
            return 0.0
        a, b = live[m], ref[m]
        return float(np.median(np.abs(a / (np.median(a) + 1.0) - b / (np.median(b) + 1.0))))

    def start_check(self, small, you, fit=True):
        """Kick off a refresh + 'is this still the same room?' check in the background."""
        if self._job is not None and self._job.is_alive():
            return
        small, you = small.copy(), you.copy()
        cur, G, started = self._snapshot(), self.G, self._g_started

        def job():
            try:
                self._result = (*self._work(small, you, cur, G, started, fit), small, you)
            except Exception as e:
                self._result = ("fail", None, started, small, you)
                log(f"(background check hiccup: {e})")

        self._job = threading.Thread(target=job, daemon=True)
        self._job.start()

    def poll_check(self):
        """Swap in a finished refresh. Returns 'mismatch' / 'match' when the verdict on the
        room photo changes, else None."""
        r = self._result
        if r is None:
            return None
        self._result = None
        st, b, self._g_started, small, you = r
        if b is not None:
            self._install(b)
        if st in ("none", "skip"):
            return None
        fitted = st == "ok"
        self.aligned = fitted
        self.last_fit_t = time.monotonic()
        self.fit_resid = self.residual(small, you)
        live = cv2.cvtColor(cv2.resize(small, (self.qw, self.qh), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = visible_room(cv2.resize(you, (self.qw, self.qh), interpolation=cv2.INTER_AREA), 7) > 0
        a, bb = live[m], self.room_q_gray[m]
        rel = np.abs(a / (np.median(a) + 1.0) - bb / (np.median(bb) + 1.0))
        diff = float((rel > 0.18).mean())
        # did the camera move so much that the photo had to be bent a lot / doesn't reach you?
        moved = self.aligner.corner_shift(self.Hs) > 0.15 * self.sw
        you_q = cv2.resize(you, (self.qw, self.qh), interpolation=cv2.INTER_AREA) > 0.05
        uncovered = float((you_q & ~self.covered_q).sum()) / max(1, int(you_q.sum()))
        bad = diff > 0.35 or (not fitted and diff > 0.22) or moved or uncovered > 0.08
        if bad:
            self.banner = ("The camera moved since the room picture - press F9 to retake it from here"
                           if fitted and (moved or uncovered > 0.08) else
                           "This doesn't look like your room picture - press F9 to take a new one")
        self.info = (f"{diff * 100:.0f}% of the visible room looks different"
                     + ("" if fitted else ", and I couldn't line the photo up with the camera")
                     + (", and the camera has moved a lot since the photo" if moved else "")
                     + (f", and the photo misses {uncovered * 100:.0f}% of the spot where you are"
                        if uncovered > 0.08 else ""))
        if bad:
            self._bad, self._good = self._bad + 1, 0
        else:
            self._good, self._bad = self._good + 1, 0
        if not self.mismatch and self._bad >= 2:
            self.mismatch = True
            return "mismatch"
        if self.mismatch and self._good >= 2:
            self.mismatch = False
            return "match"
        return None

    def check(self, small, you):
        """start_check + poll_check, but waits (used by the tests)."""
        self.start_check(small, you)
        self._job.join()
        return self.poll_check()

    def fill(self, sl, k):
        """The room behind you for the full-size region sl, with a little sensor grain."""
        return self.grainy[k % len(self.grainy)][sl]


def compose(frame, fill_mask, bg, k):
    """Your live camera, with only the area under fill_mask (you, your outline and shadow)
    replaced by the room behind you."""
    out = frame.copy()
    sh, sw = fill_mask.shape
    x, y, w, h = cv2.boundingRect((fill_mask > 0.02).view(np.uint8))
    if w == 0 or h == 0:
        return out
    H, W = frame.shape[:2]
    fx0, fy0 = x * W // sw, y * H // sh
    fx1, fy1 = min(W, -(-(x + w) * W // sw)), min(H, -(-(y + h) * H // sh))
    sl = (slice(fy0, fy1), slice(fx0, fx1))
    m = cv2.resize(np.ascontiguousarray(fill_mask[y:y + h, x:x + w]), (fx1 - fx0, fy1 - fy0),
                   interpolation=cv2.INTER_LINEAR)
    out[sl] = cv2.blendLinear(bg.fill(sl, k), frame[sl], m, 1.0 - m)
    return out


def build_mask(conf, small, room_q, use_diff=True):
    """Person mask at preview resolution. Starts from the AI segmentation, then (your idea)
    adds anything near you that differs from the empty-room photo - things you're holding,
    hair edges, your shadow on the wall - so they turn to dust with you."""
    m = conf
    if use_diff and room_q is not None:
        qh, qw = room_q.shape[:2]
        a = cv2.resize(small, (qw, qh), interpolation=cv2.INTER_AREA)
        d = cv2.absdiff(cv2.blur(a, (3, 3)), cv2.blur(room_q, (3, 3))).max(axis=2)
        dm = np.clip((d.astype(np.float32) - 30.0) * (1.0 / 20.0), 0, 1)
        cq = cv2.resize(conf, (qw, qh), interpolation=cv2.INTER_AREA)
        k = max(3, int(qw * 0.05) | 1)
        near = cv2.dilate((cq > 0.5).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        dm = cv2.morphologyEx(dm * near, cv2.MORPH_OPEN, K3)
        m = np.maximum(m, cv2.resize(dm, (conf.shape[1], conf.shape[0]), interpolation=cv2.INTER_LINEAR))
    m = cv2.dilate(m, K5)
    return cv2.GaussianBlur(m, (0, 0), 1.6)


# --------------------------------------------------------------------------- the effect

def fractal_noise(h, w, rng, octaves=6, base=3):
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh = base * 2 ** o
        gw = max(2, int(round(gh * w / h)))
        grid = rng.random((gh + 1, gw + 1)).astype(np.float32)
        out += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.55
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


class DustEffect:
    """Each pixel of you gets a 'crumble time' (a sweep across your body + organic noise).
    Near that time it chars dark, then breaks off as a dust particle that drifts, swirls,
    accelerates downwind and fades. Particle motion is a closed-form function of time,
    so playing time backwards makes the dust fly back and re-form you."""

    def __init__(self, w, h, sw, sh, duration=3.0, wind="right", seed=None):
        """w, h: output size.  sw, sh: size of the person mask (the 'small' working size)."""
        self.W, self.H, self.SW, self.SH = w, h, sw, sh
        self.Td = max(0.5, float(duration))
        self.sc = w / 1280.0
        self.step = 2
        gy, gx = np.mgrid[0:h - 1:self.step, 0:w - 1:self.step]
        self.gx = gx.ravel().astype(np.int32)
        self.gy = gy.ravel().astype(np.int32)
        self.sgx = np.minimum(self.gx * sw // w, sw - 1).astype(np.int32)   # same points in mask coords
        self.sgy = np.minimum(self.gy * sh // h, sh - 1).astype(np.int32)
        cap = self.gx.size
        self.px = np.zeros(cap, np.float32)
        self.py = np.zeros(cap, np.float32)
        self.ts = np.zeros(cap, np.float32)
        self.speed = np.zeros(cap, np.float32)
        self.life = np.zeros(cap, np.float32)
        self.phase = np.zeros(cap, np.float32)
        self.shrink = np.zeros(cap, np.float32)
        self.spread = np.zeros(cap, np.float32)
        self.liftk = np.zeros(cap, np.float32)
        self.col = np.zeros((cap, 3), np.float32)
        self.n = 0
        self.max_life = 2.6
        self.total = self.Td + self.max_life + 0.1
        self.density = 0.5            # fraction of pixels that become flying flakes
        self.wind = wind
        self.wx, self.wy = {"right": (1.0, 0.0), "left": (-1.0, 0.0), "up": (0.0, -1.0)}[wind]
        self.rng = np.random.default_rng(seed)
        # motion tuning, in pixels for a 1280-wide frame
        self.v0, self.acc, self.lift, self.turb = 30.0, 260.0, 70.0, 60.0
        self.ash_img = np.empty((h, w, 3), np.uint8)
        self.ash_img[:] = ASH.astype(np.uint8)
        self.Ds = None
        self.spawned_to = 0.0

    def start(self, mask):
        """mask: person mask at the small working size (sh x sw)."""
        hs, ws = self.SH, self.SW
        small = mask
        ys, xs = np.nonzero(small > 0.5)
        if xs.size < 200:
            return False
        x0, x1 = np.percentile(xs, [1, 99])
        y0, y1 = np.percentile(ys, [1, 99])
        yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32)
        u = np.clip((xx - x0) / max(1.0, x1 - x0), 0, 1)
        v = np.clip((yy - y0) / max(1.0, y1 - y0), 0, 1)
        along = {"right": 1 - u, "left": u, "up": v}[self.wind]   # downwind side goes first
        d = 0.55 * along + 0.45 * fractal_noise(hs, ws, self.rng)
        lo, hi = np.percentile(d[small > 0.5], [0.5, 99.5])
        self.Ds = np.clip((d - float(lo)) / max(1e-3, float(hi - lo)), 0, 1).astype(np.float32)
        self.Dg = self.Ds[self.sgy, self.sgx]
        self.n = 0
        self.spawned_to = 0.0
        return True

    def _spawn(self, s_lo, s_hi, frame, mask):
        self.spawned_to = s_hi
        sel = np.flatnonzero((self.Dg >= s_lo) & (self.Dg < s_hi))
        if sel.size == 0:
            return
        gx, gy = self.gx[sel], self.gy[sel]
        keep = (mask[self.sgy[sel], self.sgx[sel]] > 0.45) & (self.rng.random(sel.size) < self.density)
        sel, gx, gy = sel[keep], gx[keep], gy[keep]
        k = min(sel.size, self.px.size - self.n)
        if k <= 0:
            return
        sel, gx, gy = sel[:k], gx[:k], gy[:k]
        i = slice(self.n, self.n + k)
        live = frame[gy, gx].astype(np.float32)
        self.col[i] = live + (ASH - live) * 0.62           # same char colour as the crumbling edge
        self.px[i], self.py[i] = gx, gy
        self.ts[i] = self.Dg[sel] * self.Td
        r = self.rng
        self.speed[i] = r.uniform(0.55, 1.45, k)
        self.life[i] = r.uniform(1.3, self.max_life, k)
        self.phase[i] = r.uniform(0, 2 * np.pi, k)
        self.shrink[i] = r.uniform(0.15, 0.9, k)
        self.spread[i] = r.normal(0, 0.28, k)
        self.liftk[i] = r.uniform(0.3, 1.7, k)
        self.n += k

    def render(self, t, frame, mask, out, spawn=True):
        """frame: live camera (uint8, full size).  mask: person mask at the small size.
        out: uint8 full-size image of the empty room (contiguous); you + dust are drawn into it."""
        s = t / self.Td
        if spawn and s > self.spawned_to:
            self._spawn(self.spawned_to, s, frame, mask)
        if s < 1.02:
            x, y, w, h = cv2.boundingRect((mask > 0.02).view(np.uint8))
            if w > 0 and h > 0:
                # maths at the small size, then upscale the two weight maps to the full-size region
                diff = self.Ds[y:y + h, x:x + w] - s
                alive = np.clip(diff * (1 / 0.012), 0, 1)
                w_ash = 0.62 - 0.62 * np.clip(diff * (1 / 0.11), 0, 1)   # charring near the edge
                a = mask[y:y + h, x:x + w] * alive
                fx0, fy0 = x * self.W // self.SW, y * self.H // self.SH
                fx1 = min(self.W, -(-(x + w) * self.W // self.SW))
                fy1 = min(self.H, -(-(y + h) * self.H // self.SH))
                size = (fx1 - fx0, fy1 - fy0)
                a = cv2.resize(a, size, interpolation=cv2.INTER_LINEAR)
                w_ash = cv2.resize(w_ash, size, interpolation=cv2.INTER_LINEAR)
                sl = (slice(fy0, fy1), slice(fx0, fx1))
                person = cv2.blendLinear(frame[sl], self.ash_img[sl], 1.0 - w_ash, w_ash)
                out[sl] = cv2.blendLinear(person, out[sl], a, 1.0 - a)
        self._draw_particles(out, t)
        return out

    def _draw_particles(self, out, t):
        n = self.n
        if n == 0:
            return
        age = t - self.ts[:n]
        idx = np.flatnonzero((age >= 0) & (age < self.life[:n]))
        if idx.size == 0:
            return
        a = age[idx]
        sp, ph = self.speed[idx], self.phase[idx]
        x0, y0 = self.px[idx], self.py[idx]
        sc = self.sc
        dist = (self.v0 * a + 0.5 * self.acc * a * a) * sp * sc
        rise = 0.5 * self.lift * a * a * sp * sc * self.liftk[idx]
        side = dist * self.spread[idx]               # fan the cloud out as it travels
        wob = self.turb * sc * a
        x = (x0 + self.wx * dist - self.wy * side + wob * np.sin(y0 * 0.021 + a * 2.3 + ph)
             + 0.35 * wob * np.sin(x0 * 0.063 + a * 4.1 + 2 * ph))
        y = (y0 + self.wy * dist + self.wx * side - rise + wob * np.cos(x0 * 0.017 + a * 1.9 + ph)
             + 0.35 * wob * np.cos(y0 * 0.058 + a * 3.7 + 2 * ph))
        ok = (x >= 0) & (x < self.W - 1) & (y >= 0) & (y < self.H - 1)
        if not ok.any():
            return
        x, y, a, idx = x[ok], y[ok], a[ok], idx[ok]
        alpha = (0.95 * (1.0 - a / self.life[idx]) ** 1.8)[:, None]
        col = self.col[idx]
        col = col + (ASH_AGED - col) * (np.clip(a * 1.4, 0, 1) * 0.45)[:, None]
        lin = y.astype(np.int64) * self.W + x.astype(np.int64)
        flat = out.reshape(-1, 3)

        def blend(li, c, al):
            src = flat[li].astype(np.float32)
            flat[li] = (src + (c - src) * al).astype(np.uint8)

        blend(lin, col, alpha)
        big = a < self.shrink[idx]                   # fresh flakes are 2x2, older ones crumble to 1px
        lb, cb, ab = lin[big], col[big], alpha[big]
        for off in (1, self.W, self.W + 1):
            blend(lb + off, cb, ab)


# --------------------------------------------------------------------------- room capture

class RoomCapture:
    """Builds the empty-room picture from every bit of the room the camera gets to see.

    Step out of the shot - or, if getting up would move your laptop, just lean far to one side
    and then the other. It stitches together everything it has seen behind you (lining each
    frame up first in case the camera wobbles), and counts spots seen far away from you more,
    so your outline and shadow don't get baked in. Shows how much of the room it has so far."""

    MAX_SECONDS = 40

    def __init__(self, delay, w, h, sw, sh):
        self.delay = delay
        self.w, self.h, self.sw, self.sh = w, h, sw, sh
        self.t0 = time.monotonic()
        self.beeped = set()
        self.started = False
        self.sum = np.zeros((h, w, 3), np.float32)
        self.wsum = np.zeros((h, w), np.float32)
        self.sum_s = np.zeros((sh, sw, 3), np.float32)      # small copy, for brightness matching
        self.wsum_s = np.zeros((sh, sw), np.float32)
        self.aligner = Aligner(sw, sh)
        self.ref_set = False
        self.Hm, self.have_fit, self.fails = None, False, 0
        self.added = 0
        self.best, self.best_t = 0.0, time.monotonic()
        self.moving_until = 0.0
        self.coverage = 0.0
        self.n = 0
        log(f"Taking the EMPTY ROOM picture in {delay:.0f} s: lean far left and then far right "
            "(or step out of the shot), until it reaches 100%. Try not to move the laptop.")

    def _set_reference(self, small, conf):
        """The view right now - you sitting where you'll be in the meeting - is what the room
        picture gets built for. Every later frame is lined up to it, so if the laptop tips
        while you move, the picture still fits your seated view."""
        you = cv2.dilate((conf > 0.03).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        self.aligner.set_reference(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (1 - you) * 255)
        self.ref_set = True

    def _scale(self):
        return np.diag([self.w / self.sw, self.h / self.sh, 1.0])

    def update(self, frame, small, conf):
        """Returns (message for the preview, room image or None)."""
        now = time.monotonic()
        el = now - self.t0
        if not self.ref_set:
            self._set_reference(small, conf)
        if el < self.delay:
            left = int(math.ceil(self.delay - el))
            if left not in self.beeped:
                self.beeped.add(left)
                beep(880, 110)
            return f"Get out of the way in {left}...", None
        if not self.started:
            self.started = True
            beep(1320, 350)
        self.n += 1
        if self.n % 2 == 0:
            self._add(frame, small, conf)
        self.coverage = float((self.wsum > 1.5).mean())
        el -= self.delay
        if self.coverage > self.best + 0.005:
            self.best, self.best_t = self.coverage, now
        stalled = self.coverage >= 0.9 and now - self.best_t > 8   # nothing new seen for 8 s
        if self.coverage >= 0.995 or stalled or el > self.MAX_SECONDS:
            return "Got it!", self._finish()
        pct = f"{self.coverage * 100:.0f}%"
        if now < self.moving_until:
            return f"The laptop moved! Lean instead of getting up  {pct}", None
        if el > 10 and self.coverage < 0.9:
            return f"Still can't see behind you - step out of the shot  {pct}", None
        return f"Lean far left, then far right (or step out)  {pct}", None

    def _add(self, frame, small, conf):
        you = cv2.dilate((conf > 0.03).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        vis = (1 - you).astype(np.uint8)
        if vis.mean() < 0.002:
            return
        # how much to trust each visible spot: little right next to you, fully further away
        dist = cv2.distanceTransform(vis, cv2.DIST_L2, 3)
        w_s = (np.clip(dist / (0.06 * self.sw), 0.05, 1.0) * vis).astype(np.float32)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        f_s = small.astype(np.float32)
        f = frame
        w = cv2.resize(w_s, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        # where is the camera pointing now, compared with your seated view?
        if self.n % 6 == 0 or not self.have_fit:  # matching points is slow: not every time
            Hm, _ = self.aligner.estimate(gray, vis * 255)
            if Hm is not None:
                self.Hm, self.have_fit, self.fails = Hm, True, 0
                if self.aligner.corner_shift(Hm) > 0.04 * self.sw:
                    self.moving_until = time.monotonic() + 2.0
            else:
                self.have_fit, self.fails = False, self.fails + 1
        if not self.have_fit and self.fails < 8:
            return                                # can't line this frame up: don't use it
        Hm = self.Hm if self.have_fit else None   # (bare walls, nothing to match: assume still)
        if Hm is not None and self.aligner.corner_shift(Hm) > 0.5:
            # the camera has tipped: bring this frame back into your seated view
            S = self._scale()
            Hf = S @ Hm @ np.linalg.inv(S)
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            f = cv2.warpPerspective(f, Hf, (self.w, self.h), flags=flags)
            # trim the frame's own edge so its black border can't bleed in as a dark line
            w = cv2.erode(cv2.warpPerspective(w, Hf, (self.w, self.h), flags=flags), K5, iterations=2)
            f_s = cv2.warpPerspective(f_s, Hm, (self.sw, self.sh), flags=flags)
            w_s = cv2.erode(cv2.warpPerspective(w_s, Hm, (self.sw, self.sh), flags=flags), K3, iterations=2)
        # the webcam re-adjusts brightness as you move: match this frame to what we have
        gain = np.ones(3, np.float32)
        both = (self.wsum_s > 0.5) & (w_s > 0.5)
        if both.sum() > 500:
            have = self.sum_s[both] / self.wsum_s[both][:, None]
            gain = np.median((have + 4) / (f_s[both] + 4), axis=0).astype(np.float32)
        w3 = cv2.merge([w * gain[0], w * gain[1], w * gain[2]])
        cv2.accumulateProduct(f.astype(np.float32), w3, self.sum)
        cv2.accumulate(w, self.wsum)
        cv2.accumulateProduct(f_s, cv2.merge([w_s * gain[0], w_s * gain[1], w_s * gain[2]]), self.sum_s)
        cv2.accumulate(w_s, self.wsum_s)
        self.added += 1
        if self.added % 4 == 0:
            # from now on, match against everything seen so far (already in your seated view's
            # coordinates), not just the strips visible around you at the start
            seen = cv2.erode(((self.wsum_s > 0.8) * 255).astype(np.uint8), K5)
            so_far = (self.sum_s / np.maximum(self.wsum_s, 1e-3)[..., None]).clip(0, 255).astype(np.uint8)
            self.aligner.set_reference(cv2.cvtColor(so_far, cv2.COLOR_BGR2GRAY), seen)

    def _finish(self):
        room = (self.sum / np.maximum(self.wsum, 1e-3)[..., None]).clip(0, 255).astype(np.uint8)
        holes = (self.wsum < 0.3).astype(np.uint8)
        if holes.mean() > 0.0005:             # never seen at all: paint in from around it
            small = cv2.resize(room, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
            hs = cv2.dilate(cv2.resize(holes, (self.sw, self.sh), interpolation=cv2.INTER_NEAREST), K3)
            filled = cv2.resize(cv2.inpaint(small, hs, 7, cv2.INPAINT_TELEA), (self.w, self.h))
            room[holes > 0] = filled[holes > 0]
            log(f"{holes.mean() * 100:.1f}% of the room was never visible - filled that in by guessing.")
        beep(1568, 120, times=2)
        return room

    def progress_image(self, width):
        """Little preview of the room picture so far; red = not seen yet."""
        hgt = int(round(width * self.sh / self.sw))
        img = (self.sum_s / np.maximum(self.wsum_s, 1e-3)[..., None]).clip(0, 255).astype(np.uint8)
        img = cv2.resize(img, (width, hgt), interpolation=cv2.INTER_AREA)
        seen = cv2.resize((self.wsum_s > 1.5).astype(np.uint8), (width, hgt), interpolation=cv2.INTER_NEAREST)
        img[seen == 0] = (40, 40, 200)
        return img


# --------------------------------------------------------------------------- UI bits

def draw_hud(img, state, info):
    h, w = img.shape[:2]
    s = w / 1280.0

    def text(t, x, y, scale=0.7, color=(255, 255, 255), thick=2):
        x, y = int(x), int(y)
        th = max(1, int(round(thick * s)))
        # outline = the same text nudged around in black (a thicker stroke changes glyph widths)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
            cv2.putText(img, t, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale * s, (0, 0, 0), th, cv2.LINE_AA)
        cv2.putText(img, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * s, color, th, cv2.LINE_AA)

    tr = info["tracker"]
    if tr is not None and info["show_hand"]:
        tr.draw(img, w / info["frame_w"])

    labels = {
        "capture": ("TAKING EMPTY-ROOM PHOTO", (0, 200, 255)),
        "idle": ("READY - snap to turn to dust", (120, 255, 120)),
        "dusting": ("I don't feel so good...", (80, 170, 255)),
        "gone": ("GONE - snap again to come back", (220, 220, 220)),
        "returning": ("COMING BACK...", (255, 210, 120)),
    }
    label, color = labels[state]
    text(label, 20 * s, 42 * s, 0.9, color)
    vc = "virtual cam ON" if info["vcam"] else "virtual cam OFF - preview only"
    right = f"{info['fps']:.0f} fps   {vc}"
    rw = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, max(1, int(2 * s)))[0][0]
    text(right, w - rw - 20 * s, 40 * s, 0.6, (180, 255, 180) if info["vcam"] else (120, 160, 255))
    if tr is not None and state != "capture":
        if tr.ready_recently(0.3):
            t, c = "hand: READY - snap!", (0, 220, 255)
        elif tr.seen_recently(0.3):
            t, c = "hand: seen - touch thumb + middle finger to get ready", (120, 255, 120)
        else:
            t, c = "hand: not seen", (170, 170, 170)
        text(t, 20 * s, 76 * s, 0.6, c)

    if info["mismatch"]:
        y0 = int(96 * s)
        cv2.rectangle(img, (0, y0), (w, y0 + int(44 * s)), (30, 30, 160), -1)
        text(info["mismatch"], 20 * s, y0 + 30 * s, 0.7, (255, 255, 255))

    # mic meter
    x0, y0, bw, bh = int(20 * s), int(h - 70 * s), int(300 * s), int(16 * s)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (40, 40, 40), -1)
    if info["mic"] is not None:
        lvl = float(np.clip(info["mic"].level_db / 45.0, 0, 1))
        thr = float(np.clip(info["mic"].th_db / 45.0, 0, 1))
        on = info["mic"].enabled
        cv2.rectangle(img, (x0, y0), (x0 + int(bw * lvl), y0 + bh),
                      (60, 200, 255) if lvl >= thr else (120, 200, 120), -1)
        cv2.line(img, (x0 + int(bw * thr), y0 - 4), (x0 + int(bw * thr), y0 + bh + 4), (0, 0, 255), 2)
        text("mic" if on else "mic OFF (M)", x0 + bw + 10 * s, y0 + bh, 0.55,
             (255, 255, 255) if on else (120, 120, 255))
    else:
        text("no mic - use F8 / SPACE", x0 + bw + 10 * s, y0 + bh, 0.55, (120, 160, 255))
    text("F8/SPACE snap   F9/B retake room   M mic on/off   H hide text   Q quit",
         20 * s, h - 20 * s, 0.55, (230, 230, 230), 1)

    if info["big"]:
        sc = 1.6
        size = cv2.getTextSize(info["big"], cv2.FONT_HERSHEY_SIMPLEX, sc * s, max(1, int(4 * s)))[0]
        if size[0] > w * 0.94:                       # long message: shrink to fit
            sc *= w * 0.94 / size[0]
            size = cv2.getTextSize(info["big"], cv2.FONT_HERSHEY_SIMPLEX, sc * s, max(1, int(4 * s)))[0]
        text(info["big"], (w - size[0]) / 2, h / 2, sc, (0, 220, 255), 4 if sc > 1.2 else 3)
    return img


def open_virtual_cam(w, h, fps):
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        log(f"Virtual camera ON: '{cam.device}'  <- choose this camera in Google Meet")
        return cam
    except Exception as e:
        log(f"Virtual camera not available: {e}")
        log("  -> Install OBS Studio once (free, obsproject.com); it adds 'OBS Virtual Camera'.")
        log("  Running in preview-only mode for now.")
        return None


def start_hotkeys(events):
    try:
        from pynput import keyboard
    except Exception:
        log("pynput not installed - global F8/F9 hotkeys off (preview-window keys still work).")
        return None

    def on_press(key):
        if key == keyboard.Key.f8:
            events.append("snap")
        elif key == keyboard.Key.f9:
            events.append("room")

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    return listener


def list_devices():
    import sounddevice as sd
    print("\nMicrophones (use --mic <number>):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i:3d}  {d['name']}")
    print("\nCameras (use --camera <number>):")
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if cap.isOpened():
            ok, f = cap.read()
            if ok and f is not None:
                bw = np.abs(f[..., 0].astype(np.int16) - f[..., 2]).mean() <= 1.0
                print(f"  {i:3d}  {f.shape[1]}x{f.shape[0]}" + ("  (black & white - probably IR)" if bw else ""))
        cap.release()


def parse_args():
    p = argparse.ArgumentParser(description="Thanos-snap yourself to dust on your webcam.")
    p.add_argument("--camera", type=int, default=None, help="webcam number (default: first colour camera)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--mic", default=None, help="microphone number or name (see --list-devices)")
    p.add_argument("--sensitivity", type=float, default=5, help="snap sensitivity 1 (strict) .. 10 (twitchy)")
    p.add_argument("--no-mic", action="store_true", help="don't listen for snaps (use F8 / SPACE)")
    p.add_argument("--no-hand-check", action="store_true",
                   help="no hand tracking: only the snap SOUND (or F8) triggers, hand visible or not")
    p.add_argument("--room", default=None, help="use this photo of the empty room instead of taking one")
    p.add_argument("--retake", action="store_true", help="ignore the saved empty-room photo and take a new one")
    p.add_argument("--room-delay", type=float, default=4, help="seconds before it starts building the room picture")
    p.add_argument("--duration", type=float, default=3.0, help="seconds it takes you to crumble")
    p.add_argument("--delay", type=float, default=0.6, help="pause between the snap and the crumbling")
    p.add_argument("--return-speed", type=float, default=1.6, help="speed of the come-back animation")
    p.add_argument("--wind", choices=["right", "left", "up"], default="right", help="where the dust blows")
    p.add_argument("--seg-only", action="store_true", help="use only the AI cut-out, not the room difference")
    p.add_argument("--no-vcam", action="store_true", help="preview window only")
    p.add_argument("--show-hand", action="store_true", help="draw the tracked hand points in the preview")
    p.add_argument("--list-devices", action="store_true", help="list cameras and microphones, then exit")
    return p.parse_args()


# --------------------------------------------------------------------------- main

def main():
    args = parse_args()
    if args.list_devices:
        list_devices()
        return
    print(__doc__)
    ensure_models(need_hand=not args.no_hand_check)

    cam = Camera(args.camera, args.width, args.height, args.fps)
    W, H = cam.w, cam.h
    SW = min(640, W)
    SH = max(2, int(round(H * SW / W / 2)) * 2)
    PW, PH = 960, max(2, int(round(960 * H / W)))       # preview window size
    log(f"Webcam #{cam.index}: {W}x{H}")

    seg = PersonSegmenter(SEG_MODEL)
    tracker = None
    if not args.no_hand_check:
        tracker = HandTracker(PALM_MODEL, LANDMARK_MODEL)
        tracker.start()
        log("Hand tracking on: touch thumb + middle finger together, then snap.")

    snap = None
    if not args.no_mic:
        mic = args.mic
        if mic is not None and str(mic).isdigit():
            mic = int(mic)
        try:
            snap = SnapDetector(args.sensitivity, mic)
            log(f"Also listening for the snap sound on: {snap.name}")
        except Exception as e:
            log(f"Microphone unavailable ({e}) - the camera / F8 / SPACE still work.")

    vcam = None if args.no_vcam else open_virtual_cam(W, H, args.fps)
    events = deque()
    hotkeys = start_hotkeys(events)

    bg = Background(W, H, SW, SH)
    effect = DustEffect(W, H, SW, SH, args.duration, args.wind)

    state, capture = "idle", None
    if args.room:
        img = cv2.imread(args.room)
        if img is None:
            sys.exit(f"Could not read the room photo: {args.room}")
        bg.set_room(img)
        log(f"Using your empty-room photo: {args.room}")
    elif os.path.exists(ROOM_FILE) and not args.retake:
        img = cv2.imread(ROOM_FILE)
        if img is not None:
            bg.set_room(img)
            log("Using the saved empty-room photo - I'll check it still matches the room.")
    if not bg.ready:
        state, capture = "capture", RoomCapture(args.room_delay, W, H, SW, SH)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, PW, PH)

    t, seq, frame_i = 0.0, 0, 0
    prev_conf = hold = None
    last = time.monotonic()
    fps = float(args.fps)
    flash_text, flash_until = "", 0.0
    printed_info, last_print = "", 0.0
    snap_cool = 0.0
    show_hud = True
    try:
        while True:
            frame, seq = cam.read(seq)
            now = time.monotonic()
            dt = min(0.1, now - last)
            last = now
            fps = 0.92 * fps + 0.08 / max(dt, 1e-3)
            frame_i += 1
            if tracker is not None:
                tracker.submit(frame)

            small = cv2.resize(frame, (SW, SH), interpolation=cv2.INTER_AREA)
            conf = seg(small)
            if prev_conf is not None:
                conf = 0.65 * conf + 0.35 * prev_conf
            prev_conf = conf

            # ---- where are you, and what's behind you
            mask = None
            if bg.ready and state != "capture":
                mask = build_mask(conf, small, bg.room_q, not args.seg_only)
                # 'hold' bridges a frame or two where the cut-out misses an arm or a hand
                hold = mask if hold is None else np.maximum(mask, hold * 0.8)
                # twice a second, in the background: re-fit the photo to the camera (laptops on
                # beds wobble), re-match the lighting, and check it's still the same room
                if frame_i % 15 == 5:
                    # the full re-fit is CPU-heavy: only when the view looks shifted, or every 4 s
                    fit = (now - bg.last_fit_t > 4.0
                           or bg.residual(small, hold) > bg.fit_resid * 1.4 + 0.015)
                    bg.start_check(small, hold, fit)
                verdict = bg.poll_check()
                if verdict == "mismatch":
                    log(bg.banner + "  (" + bg.info + ")")
                    beep(520, 200, times=2)
                elif verdict == "match":
                    log("The room matches your photo again.")

            # ---- inputs
            want_snap = want_room = False
            if tracker is not None and tracker.poll_snap():
                want_snap = True
                log("Saw your snap (camera).")
            if snap is not None:
                if snap.poll():
                    if tracker is None or tracker.ready_recently(1.2):
                        want_snap = True
                        log("Heard your snap (mic).")
                    else:
                        log("Heard a snap-like sound, but didn't see your hand in the snap pose - ignored.")
                if snap.last_info != printed_info and (now - last_print > 2 or "SNAP" in snap.last_info):
                    printed_info, last_print = snap.last_info, now
                    log("mic: " + snap.last_info)
            while events:
                e = events.popleft()
                want_snap |= e == "snap"
                want_room |= e == "room"
                if e == "mic" and snap is not None:
                    snap.enabled = not snap.enabled
                    log("Snap listening " + ("ON" if snap.enabled else "OFF"))
            if want_snap:                      # camera + mic can both catch the same snap
                if now < snap_cool:
                    want_snap = False
                else:
                    snap_cool = now + 1.0

            if want_room:
                state, capture, hold = "capture", RoomCapture(args.room_delay, W, H, SW, SH), None
            elif want_snap and state != "capture":
                if state == "idle":
                    if mask is not None and effect.start(mask):
                        bg.align(small, hold)            # freshest possible fit of the photo
                        state, t = "dusting", -args.delay
                        log("*SNAP*  ...I don't feel so good.")
                        if bg.mismatch:
                            log("  (the room photo doesn't match - the spot where you were may look off)")
                    else:
                        log("Snap! ...but I can't see anyone to turn to dust.")
                elif state in ("gone", "dusting"):
                    if state == "gone":
                        t = effect.Td + effect.max_life   # last moment any dust was still visible
                    state = "returning"
                    log("*SNAP*  Bringing you back...")
                elif state == "returning":
                    state = "dusting"

            # ---- advance time
            if state == "dusting":
                t += dt
                if t >= effect.total:
                    state = "gone"
                    log("You've been dusted. Snap again (or F8) to come back.")
            elif state == "returning":
                t -= dt * args.return_speed
                if t <= 0:
                    state = "idle"
                    log("You're back!")

            # ---- render
            big = ""
            if state == "capture":
                big, room = capture.update(frame, small, conf)
                out = frame
                if room is not None:
                    bg.set_room(room)
                    cv2.imwrite(ROOM_FILE, room)
                    state, capture = "idle", None
                    prev_conf = hold = None
                    log("Empty room saved. Come back and sit down - snap whenever you're ready!")
                    flash_text, flash_until = "Got it! Come back in.", now + 2.5
            elif state == "idle" or (state == "dusting" and t < 0):
                out = frame
            else:
                # live camera everywhere, except the spot where you are -> the room behind you
                # generous: the AI cut-out is a bit coarse, and a sliver of your edge (hair,
                # shoulders) left behind reads as an outline. The room picture is lined up and
                # colour-matched, so filling a little extra around you doesn't show.
                fill = cv2.GaussianBlur(cv2.dilate(hold, K_FILL), (0, 0), 2.0)
                out = compose(frame, fill, bg, frame_i)
                if state != "gone":
                    out = effect.render(t, frame, mask, out, spawn=(state == "dusting"))

            if vcam is not None:
                vcam.send(out)

            if not big and now < flash_until:
                big = flash_text
            preview = draw_hud(cv2.resize(out, (PW, PH), interpolation=cv2.INTER_AREA), state, {
                "fps": fps, "vcam": vcam is not None, "mic": snap, "tracker": tracker,
                "frame_w": W, "mismatch": bg.banner if bg.mismatch and state != "capture" else "",
                "big": big, "show_hand": args.show_hand}) if show_hud else \
                cv2.resize(out, (PW, PH), interpolation=cv2.INTER_AREA)
            if state == "capture" and capture is not None and capture.started:
                inset = capture.progress_image(PW // 3)          # room picture so far, red = unseen
                ih, iw = inset.shape[:2]
                x0, y0 = PW - iw - 12, 56
                preview[y0:y0 + ih, x0:x0 + iw] = inset
                cv2.rectangle(preview, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih), (255, 255, 255), 1)
            cv2.imshow(WINDOW, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                events.append("snap")
            elif key == ord("b"):
                events.append("room")
            elif key == ord("m"):
                events.append("mic")
            elif key == ord("h"):
                show_hud = not show_hud          # clean preview, to watch the effect
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        log("Shutting down.")
        if hotkeys is not None:
            hotkeys.stop()
        if tracker is not None:
            tracker.stop()
        if snap is not None:
            snap.close()
        if vcam is not None:
            vcam.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
