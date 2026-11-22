"""
Hand tracking + visual finger-snap detection, using MediaPipe's hand models run through
OpenCV's DNN module (no MediaPipe install needed).

  1. Palm detector finds a hand in the whole frame.
  2. Landmark model gives 21 points on that hand (wrist, knuckles, fingertips).
  3. While tracking, each new frame is cropped around last frame's hand (fast).

Snap = thumb tip and middle fingertip pressed together ("ready"), then within a fraction of a
second they fly apart AND the middle finger lands curled in the palm.
"""
import math
import threading
import time

import cv2
import numpy as np

WRIST, THUMB_TIP, INDEX_MCP, MIDDLE_MCP, MIDDLE_TIP = 0, 4, 5, 9, 12
HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], np.float32)


def _palm_anchors():
    anchors = []
    for stride, per_cell in ((8, 2), (16, 6)):
        n = 192 // stride
        for y in range(n):
            for x in range(n):
                anchors += [((x + 0.5) / n, (y + 0.5) / n)] * per_cell
    return np.array(anchors, np.float32)


class HandTracker(threading.Thread):
    def __init__(self, palm_model, landmark_model, palm_threshold=0.5):
        super().__init__(daemon=True)
        self.palm = cv2.dnn.readNet(palm_model)
        self.lmk = cv2.dnn.readNet(landmark_model)
        self.palm_out = self.palm.getUnconnectedOutLayersNames()
        self.lmk_out = self.lmk.getUnconnectedOutLayersNames()
        self.anchors = _palm_anchors()
        self.palm_threshold = palm_threshold
        self.lock = threading.Lock()
        self.pending = None
        self.running = True
        self.roi = None                 # (center xy, size, angle) of the hand crop
        self.landmarks = None           # 21x2 in frame pixels, or None
        self.last_seen = -1e9
        self.pinch = self.curl = float("nan")
        self.ready_until = -1e9         # snap pose ("ready") seen until this time
        self.ready_frames = 0
        self.ready_curl = 0.0
        self.last_ready = -1e9
        self.cool_until = 0.0
        self._snaps = 0
        self.fps = 0.0

    # ------------------------------------------------------------------ public
    def submit(self, frame):
        with self.lock:
            self.pending = frame

    def seen_recently(self, seconds=1.5):
        return time.monotonic() - self.last_seen < seconds

    def ready_recently(self, seconds=1.0):
        return time.monotonic() - self.last_ready < seconds

    def poll_snap(self):
        with self.lock:
            n, self._snaps = self._snaps, 0
        return n > 0

    def stop(self):
        self.running = False

    # ------------------------------------------------------------------ models
    def detect_palm(self, img):
        h, w = img.shape[:2]
        s = 192.0 / max(h, w)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        ox, oy = (192 - nw) // 2, (192 - nh) // 2
        pad = np.zeros((192, 192, 3), np.uint8)
        pad[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        self.palm.setInput(cv2.dnn.blobFromImage(pad, 1 / 255.0, (192, 192), swapRB=True))
        outs = self.palm.forward(self.palm_out)
        reg = [o for o in outs if o.shape[-1] == 18][0][0]
        sc = _sigmoid([o for o in outs if o.shape[-1] == 1][0][0, :, 0])
        i = int(np.argmax(sc))
        if sc[i] < self.palm_threshold:
            return None
        r, a = reg[i], self.anchors[i]

        def to_px(nx, ny):
            return np.array([(nx * 192 - ox) / s, (ny * 192 - oy) / s], np.float32)

        center = to_px(a[0] + r[0] / 192, a[1] + r[1] / 192)
        bw, bh = r[2] / s, r[3] / s
        kp0 = to_px(a[0] + r[4] / 192, a[1] + r[5] / 192)      # wrist
        kp2 = to_px(a[0] + r[8] / 192, a[1] + r[9] / 192)      # middle finger knuckle
        ang = math.pi / 2 - math.atan2(-(kp2[1] - kp0[1]), kp2[0] - kp0[0])
        # MediaPipe's palm -> hand crop: shift towards the fingers, scale 2.6, make square
        center = center + np.array([0.5 * bh * math.sin(ang), -0.5 * bh * math.cos(ang)], np.float32)
        return center, max(bw, bh) * 2.6, ang, float(sc[i])

    def hand_landmarks(self, img, roi):
        center, size, ang = roi
        R = _rot(ang)
        A = R * (size / 224.0)
        b = center - R @ np.array([size / 2, size / 2], np.float32)
        M = np.hstack([A, b[:, None]]).astype(np.float32)
        crop = cv2.warpAffine(img, M, (224, 224), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT)
        self.lmk.setInput(cv2.dnn.blobFromImage(crop, 1 / 255.0, (224, 224), swapRB=True))
        outs = dict(zip(self.lmk_out, self.lmk.forward(self.lmk_out)))
        flat = [o for o in outs.values() if o.size == 63]
        scal = [o for o in outs.values() if o.size == 1]
        lm = flat[0].reshape(21, 3)[:, :2]
        presence = float(scal[0].ravel()[0])
        if presence < 0 or presence > 1:
            presence = float(_sigmoid(presence))
        pts = lm @ A.T + b
        return pts.astype(np.float32), presence

    @staticmethod
    def roi_from_landmarks(L):
        p0, p9 = L[WRIST], L[MIDDLE_MCP]
        ang = math.pi / 2 - math.atan2(-(p9[1] - p0[1]), p9[0] - p0[0])
        pts = L[[0, 1, 2, 5, 9, 13, 17]]
        c0 = (pts.min(0) + pts.max(0)) / 2
        q = (pts - c0) @ _rot(-ang).T
        qmin, qmax = q.min(0), q.max(0)
        w, h = qmax - qmin
        center = c0 + _rot(ang) @ ((qmin + qmax) / 2)
        center = center + np.array([0.1 * h * math.sin(ang), -0.1 * h * math.cos(ang)], np.float32)
        return center.astype(np.float32), float(max(w, h) * 2.0), ang

    # ------------------------------------------------------------------ snap gesture
    def _update_gesture(self, L, now):
        hs = float(np.linalg.norm(L[WRIST] - L[MIDDLE_MCP])) + 1e-6
        self.pinch = float(np.linalg.norm(L[THUMB_TIP] - L[MIDDLE_TIP])) / hs
        self.curl = float(np.linalg.norm(L[MIDDLE_TIP] - L[WRIST])) / hs
        if self.pinch < 0.38:                                   # thumb + middle finger together
            self.ready_frames += 1
            self.ready_curl = self.curl if self.ready_frames == 1 else 0.7 * self.ready_curl + 0.3 * self.curl
            self.last_ready = now
            if self.ready_frames >= 2:
                self.ready_until = now + 0.5
            return
        self.ready_frames = 0
        if (now < self.ready_until and now > self.cool_until and self.pinch > 0.55
                and self.curl < self.ready_curl - 0.22):        # ...then apart, middle finger in palm
            with self.lock:
                self._snaps += 1
            self.ready_until = -1e9
            self.cool_until = now + 1.0

    def process(self, img, now=None):
        """Run one frame (also used directly by the tests)."""
        now = time.monotonic() if now is None else now
        L = None
        if self.roi is not None:
            L, presence = self.hand_landmarks(img, self.roi)
            if presence < 0.5:
                L, self.roi = None, None
        if L is None:
            det = self.detect_palm(img)
            if det is not None:
                L, presence = self.hand_landmarks(img, det[:3])
                if presence < 0.5:
                    L = None
        if L is None:
            self.landmarks, self.roi = None, None
            self.ready_frames = 0
            return None
        self.roi = self.roi_from_landmarks(L)
        self.landmarks = L
        self.last_seen = now
        self._update_gesture(L, now)
        return L

    def _interval(self, now):
        """Go easy on the CPU unless a snap could be coming: look for a hand 10x a second,
        follow a relaxed hand 15x a second, and track every frame once fingers come together."""
        if now < self.ready_until:
            return 0.0                    # a snap may be happening right now
        if self.landmarks is None:
            return 0.1
        if self.pinch < 0.6:
            return 0.0
        return 1 / 15

    def run(self):
        last = time.monotonic()
        while self.running:
            now = time.monotonic()
            if now - last < self._interval(now):
                time.sleep(0.004)
                continue
            with self.lock:
                img, self.pending = self.pending, None
            if img is None:
                time.sleep(0.004)
                continue
            try:
                self.process(img)
            except Exception:
                self.roi, self.landmarks = None, None
            now = time.monotonic()
            self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now

    def draw(self, img, scale):
        """Draw the tracked hand on the preview (scale = preview width / frame width)."""
        L = self.landmarks
        if L is None or not self.seen_recently(0.3):
            return
        P = (L * scale).astype(np.int32)
        ready = time.monotonic() < self.ready_until
        col = (0, 220, 255) if ready else (200, 200, 200)
        for a, b in HAND_LINES:
            cv2.line(img, tuple(P[a]), tuple(P[b]), col, 1, cv2.LINE_AA)
        for i in (THUMB_TIP, MIDDLE_TIP):
            cv2.circle(img, tuple(P[i]), 5, (0, 220, 255) if ready else (255, 255, 255), -1, cv2.LINE_AA)
