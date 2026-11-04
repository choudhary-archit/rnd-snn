"""Run SpikeYOLO on a video in real time, with mpv as the player.

mpv plays the video normally (audio, seeking, pause, fullscreen all work). This
process talks to mpv over its JSON IPC socket: it reads the playback position,
decodes the matching frame with PyAV, runs the model and pushes the boxes back
to mpv as an ASS OSD overlay.

The model is slower than the video's frame rate on CPU, so detection runs on a
"latest frame" basis: each pass picks the frame mpv will be showing when
inference finishes (position + measured latency), detects on it, and draws the
boxes exactly when playback reaches that frame. Boxes stay on screen until the
next pass replaces them.

Usage:  python video_detect.py VIDEO [--model int|bin] [--imgsz 640] [--conf 0.25]
        [any extra --mpv-option=value is forwarded to mpv, e.g. --loop-file=inf]
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import av
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
WEIGHTS = {
    "int": ROOT / "weights/int_23M/best.pt",  # integer-valued I-LIF checkpoint
    "bin": ROOT / "weights/bin_23M/best.pt",  # binary-inference checkpoint (slower on CPU)
}


def load_model(kind):
    if kind == "bin":
        # The bin checkpoint pickles modules under `yolo_spikformer`; alias that
        # name to the binary implementation before unpickling.
        import ultralytics.nn.modules.yolo_spikformer_bin as binmod
        sys.modules["ultralytics.nn.modules.yolo_spikformer"] = binmod
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS[kind]))


class MpvIPC:
    """Minimal synchronous client for mpv's JSON IPC."""

    def __init__(self, path, timeout=10.0):
        deadline = time.time() + timeout
        while True:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(path)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                self.sock.close()
                if time.time() > deadline:
                    raise RuntimeError("could not connect to mpv IPC socket")
                time.sleep(0.05)
        self.buf = b""
        self.req = 0
        self.seeked = False  # set when mpv reports a seek event

    def command(self, cmd):
        self.req += 1
        msg = {"command": cmd, "request_id": self.req}
        self.sock.sendall(json.dumps(msg).encode() + b"\n")
        while True:
            while b"\n" not in self.buf:
                chunk = self.sock.recv(65536)
                if not chunk:
                    raise ConnectionError("mpv closed")
                self.buf += chunk
            line, self.buf = self.buf.split(b"\n", 1)
            reply = json.loads(line)
            if reply.get("event") == "seek":
                self.seeked = True
            elif reply.get("request_id") == self.req:
                return reply

    def get(self, prop):
        reply = self.command(["get_property", prop])
        return reply.get("data") if reply.get("error") == "success" else None


class FrameReader:
    """Returns the video frame at a given timestamp, decoding forward when possible.

    Uses PyAV rather than OpenCV: OpenCV's bundled ffmpeg has no software AV1
    decoder, while PyAV ships dav1d.
    """

    def __init__(self, path):
        try:
            self.container = av.open(str(path))
            self.stream = self.container.streams.video[0]
        except (av.FFmpegError, IndexError):
            raise SystemExit(f"cannot open a video stream in {path}")
        self.stream.thread_type = "AUTO"
        rate = self.stream.average_rate or self.stream.guessed_rate
        self.frame_dt = 1.0 / float(rate) if rate else 1.0 / 30
        # mpv's time-pos starts at 0 even when the file's timestamps don't.
        self.t0 = (self.container.start_time or 0) / av.time_base
        self.frames = self.container.decode(self.stream)
        self.frame = None
        self.pos = -1.0  # timestamp (s) of self.frame

    def at(self, t):
        if t < self.pos - self.frame_dt / 2 or t > self.pos + 3.0:
            ts = int((max(0.0, t - self.frame_dt) + self.t0) / self.stream.time_base)
            self.container.seek(ts, stream=self.stream, backward=True)
            self.frames = self.container.decode(self.stream)
            self.frame, self.pos = None, -1.0
        while self.frame is None or self.pos < t - self.frame_dt / 2:
            try:
                f = next(self.frames)
            except (StopIteration, av.FFmpegError):
                break
            if f.time is not None:
                self.frame, self.pos = f, f.time - self.t0
        if self.frame is None:
            return None
        img = self.frame.to_ndarray(format="bgr24")
        # Apply the display-matrix rotation (phone videos) the way mpv does.
        k = round((self.frame.rotation or 0) / 90) % 4
        return np.ascontiguousarray(np.rot90(img, k)) if k else img


def ass_color(bgr):
    b, g, r = (int(c) for c in bgr)
    return f"&H{b:02X}{g:02X}{r:02X}&"


def boxes_to_ass(result, names, frame_shape, osd, status):
    """Build ASS events drawing `result` boxes in OSD (window) coordinates."""
    from ultralytics.utils.plotting import colors

    fh, fw = frame_shape[:2]
    ow, oh = osd["w"], osd["h"]
    # Area of the window the video occupies (handles letterboxing, zoom, aspect).
    vx, vy = osd["ml"], osd["mt"]
    sx = (ow - osd["ml"] - osd["mr"]) / fw
    sy = (oh - osd["mt"] - osd["mb"]) / fh
    fs = max(12, round(oh / 40))
    lw = max(1.5, oh / 400)

    events = []
    if result is not None:
        for (x1, y1, x2, y2), conf, cls in zip(result.boxes.xyxy.tolist(),
                                               result.boxes.conf.tolist(),
                                               result.boxes.cls.tolist()):
            c = ass_color(colors(int(cls), bgr=True))
            x1, x2 = vx + x1 * sx, vx + x2 * sx
            y1, y2 = vy + y1 * sy, vy + y2 * sy
            events.append(
                f"{{\\an7\\pos(0,0)\\bord{lw:.1f}\\shad0\\1a&HFF&\\3c{c}\\p1}}"
                f"m {x1:.0f} {y1:.0f} l {x2:.0f} {y1:.0f} l {x2:.0f} {y2:.0f} l {x1:.0f} {y2:.0f}")
            events.append(
                f"{{\\an1\\pos({x1:.0f},{y1:.0f})\\fs{fs}\\bord{fs / 6:.1f}\\shad0"
                f"\\1c&HFFFFFF&\\3c{c}}}{names[int(cls)]} {conf:.2f}")
    if status:
        events.append(f"{{\\an9\\pos({ow - 8},8)\\fs{fs * 0.7:.0f}\\bord1\\shad0"
                      f"\\1c&HFFFFFF&\\3c&H000000&}}{status}")
    return "\n".join(events)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("video", type=Path)
    p.add_argument("--model", choices=WEIGHTS, default="int")
    p.add_argument("--imgsz", type=int, default=640, help="inference size (320 is ~3x faster)")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--no-status", action="store_true", help="hide the stats line")
    args, mpv_extra = p.parse_known_args()
    if not args.video.is_file():
        p.error(f"no such file: {args.video}")

    reader = FrameReader(args.video)
    print(f"Loading {args.model} model ...", flush=True)
    model = load_model(args.model)
    first = reader.at(0.0)
    if first is None:
        raise SystemExit("could not decode the first frame")
    t0 = time.time()
    model.predict(first, imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)
    latency = time.time() - t0  # warm-up time; refined by an EMA during playback

    sock_path = os.path.join(tempfile.mkdtemp(prefix="spikeyolo-"), "mpv.sock")
    mpv = subprocess.Popen(["mpv", f"--input-ipc-server={sock_path}", "--force-window=yes",
                            "--osd-level=1", *mpv_extra, str(args.video)])
    overlay = {"name": "osd-overlay", "id": 1, "format": "ass-events"}
    try:
        ipc = MpvIPC(sock_path)
        last_t, det_rate = None, 0.0
        while True:
            pos = ipc.get("time-pos")
            osd = ipc.get("osd-dimensions")
            if pos is None or not osd or not osd.get("w"):
                time.sleep(0.05)  # file still loading
                continue
            paused = ipc.get("pause") or ipc.get("seeking")
            speed = ipc.get("speed") or 1.0
            if ipc.seeked:  # drop boxes from before the seek right away
                ipc.seeked = False
                ipc.command({**overlay, "data": ""})
            if paused and last_t is not None and abs(pos - last_t) < reader.frame_dt / 2:
                time.sleep(0.05)
                continue

            # Detect on the frame that will be on screen when inference is done.
            target = pos if paused else pos + latency * speed
            start = time.time()
            frame = reader.at(target)
            if frame is None:  # past the last frame
                time.sleep(0.05)
                continue
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                   device="cpu", verbose=False)[0]
            took = time.time() - start
            latency = 0.7 * latency + 0.3 * took
            det_rate = 0.7 * det_rate + 0.3 / took if det_rate else 1 / took

            # Wait for playback to reach the detected frame, then draw.
            while True:
                now = ipc.get("time-pos")
                if now is None or ipc.get("pause"):
                    break
                ahead = target - now
                if ahead <= reader.frame_dt / 2:
                    break
                time.sleep(min(ahead / speed, 0.02))
            now = now if now is not None else target
            stale = abs(now - target) > max(1.0, 2 * latency * speed)  # user seeked
            status = None if args.no_status else (
                f"SpikeYOLO {args.model} @{args.imgsz} | {det_rate:.1f} det/s | "
                f"{len(result.boxes)} boxes")
            osd = ipc.get("osd-dimensions") or osd  # window may have been resized
            data = boxes_to_ass(None if stale else result, model.names, frame.shape, osd, status)
            ipc.command({**overlay, "data": data, "res_x": osd["w"], "res_y": osd["h"]})
            last_t = target
    except (ConnectionError, BrokenPipeError):
        pass  # mpv exited
    except KeyboardInterrupt:
        mpv.terminate()
    finally:
        mpv.wait()
        try:
            os.unlink(sock_path)
            os.rmdir(os.path.dirname(sock_path))
        except OSError:
            pass


if __name__ == "__main__":
    main()
