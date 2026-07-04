from __future__ import annotations

from collections import deque

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
RED = "#e34948"

_SURFACE_RGB = np.array([252, 252, 251], np.uint8)
_BLUE_RGB = np.array([42, 120, 214], np.uint8)
_AQUA_RGB = np.array([27, 175, 122], np.uint8)


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_title(title, fontsize=9, color=INK, pad=4)


class Dashboard:

    def __init__(self, dt: float, target_speed: float,
                 width: int = 320, height: int = 720,
                 speed_window: float = 8.0, gait_window: float = 4.0,
                 fig: Figure | None = None):
        self.dt = dt
        self.target = target_speed
        self._n_speed = int(speed_window / dt)
        self._n_gait = int(gait_window / dt)
        self.gait_window = gait_window

        self.trail: list[tuple[float, float]] = []
        self.steps: tuple[list, list] = ([], [])
        self.speed_t: deque = deque(maxlen=self._n_speed)
        self.speed_v: deque = deque(maxlen=self._n_speed)
        self.gait: deque = deque(maxlen=self._n_gait)
        self._t = 0.0

        self.fig = fig or Figure(figsize=(width / 100, height / 100),
                                 dpi=100)
        self.fig.set_facecolor(SURFACE)
        self.canvas = FigureCanvasAgg(self.fig) if fig is None else fig.canvas

        self.ax_map = self.fig.add_axes([0.17, 0.50, 0.79, 0.45])
        _style(self.ax_map, "live map — trajectory & footsteps")
        self.ax_map.set_aspect("equal", adjustable="datalim")
        (self.l_trail,) = self.ax_map.plot([], [], color=INK_2, lw=1.4)
        (self.l_steps_l,) = self.ax_map.plot([], [], "o", color=BLUE, ms=3.5)
        (self.l_steps_r,) = self.ax_map.plot([], [], "o", color=AQUA, ms=3.5)
        self.q_head = self.ax_map.quiver(
            [0], [0], [1], [0], color=INK, scale=9, width=0.014, zorder=5)
        self.q_push = self.ax_map.quiver(
            [0], [0], [0], [0], color=RED, scale=5, width=0.02, zorder=6)
        self.ax_map.legend(handles=[self.l_steps_l, self.l_steps_r],
                           labels=["left", "right"], loc="upper left",
                           fontsize=7, frameon=False)

        self.ax_speed = self.fig.add_axes([0.17, 0.275, 0.79, 0.15])
        _style(self.ax_speed, "forward speed (m/s)")
        (self.l_speed,) = self.ax_speed.plot([], [], color=BLUE, lw=1.6)
        self.ax_speed.axhline(self.target, color=MUTED, ls="--", lw=1.0)
        self.ax_speed.set_ylim(-0.4, 1.6)

        self.ax_gait = self.fig.add_axes([0.17, 0.06, 0.79, 0.13])
        _style(self.ax_gait, "gait — stance bands")
        img = np.full((2, self._n_gait, 3), _SURFACE_RGB, np.uint8)
        self.im_gait = self.ax_gait.imshow(
            img, aspect="auto", interpolation="nearest",
            extent=(-gait_window, 0, -0.5, 1.5))
        self.ax_gait.set_yticks([1, 0], ["L", "R"], fontsize=7)
        self.ax_gait.grid(visible=False)

    def update(self, t: float, xy, heading, v_fwd: float, contacts,
               push=None, touchdowns=()) -> None:
        self._t = t
        self.trail.append((float(xy[0]), float(xy[1])))
        for foot, x, y in touchdowns:
            self.steps[foot].append((x, y))
        self.speed_t.append(t)
        self.speed_v.append(v_fwd)
        self.gait.append((bool(contacts[0]), bool(contacts[1])))
        self._heading = np.asarray(heading, float)
        self._push = push

    def render(self) -> np.ndarray:
        tr = np.asarray(self.trail)
        self.l_trail.set_data(tr[:, 0], tr[:, 1])
        for line, pts in ((self.l_steps_l, self.steps[0]),
                          (self.l_steps_r, self.steps[1])):
            if pts:
                p = np.asarray(pts)
                line.set_data(p[:, 0], p[:, 1])
        finite = tr[np.isfinite(tr[:, 0])]
        x, y = finite[-1]
        self.q_head.set_offsets([[x, y]])
        self.q_head.set_UVC(self._heading[0], self._heading[1])
        if self._push is not None:
            f = np.asarray(self._push)
            f = f / (np.linalg.norm(f) + 1e-9)
            self.q_push.set_offsets([[x, y]])
            self.q_push.set_UVC(f[0], f[1])
            self.q_push.set_visible(True)
        else:
            self.q_push.set_visible(False)
        m = 1.0
        self.ax_map.set_xlim(finite[:, 0].min() - m,
                             max(finite[:, 0].max(), 2) + m)
        self.ax_map.set_ylim(finite[:, 1].min() - m,
                             max(finite[:, 1].max(), 1) + m)

        self.l_speed.set_data(np.asarray(self.speed_t),
                              np.asarray(self.speed_v))
        self.ax_speed.set_xlim(max(0.0, self._t - 8.0), max(8.0, self._t))

        g = np.asarray(self.gait, bool)
        img = np.full((2, self._n_gait, 3), _SURFACE_RGB, np.uint8)
        img[0, -len(g):][g[:, 0]] = _BLUE_RGB
        img[1, -len(g):][g[:, 1]] = _AQUA_RGB
        self.im_gait.set_data(img)
        self.im_gait.set_extent((self._t - self.gait_window, self._t,
                                 -0.5, 1.5))
        self.ax_gait.set_xlim(self._t - self.gait_window, self._t)

        self.canvas.draw()
        buf = np.asarray(self.canvas.buffer_rgba())
        return buf[:, :, :3].copy()


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def hud(frame: np.ndarray, lines: list[str],
        push: tuple | None = None, badge: str | None = None) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(19)
    pad, lh = 10, 24
    w = max(int(d.textlength(s, font=f)) for s in lines) + 2 * pad
    h = lh * len(lines) + 2 * pad
    d.rounded_rectangle((12, 12, 12 + w, 12 + h), 8, fill=(11, 11, 11, 150))
    for i, s in enumerate(lines):
        d.text((12 + pad, 12 + pad + i * lh), s, font=f,
               fill=(255, 255, 255, 235))
    if push is not None:
        fmag = float(np.linalg.norm(push))
        s = f"PUSH  {fmag:.0f} N"
        fw = int(d.textlength(s, font=f))
        cx = frame.shape[1] // 2
        d.rounded_rectangle((cx - fw // 2 - pad, 14, cx + fw // 2 + pad, 48),
                            8, fill=(227, 73, 72, 220))
        d.text((cx - fw // 2, 20), s, font=f, fill=(255, 255, 255, 255))
    if badge:
        f2 = _font(17)
        bw = int(d.textlength(badge, font=f2))
        W = frame.shape[1]
        d.rounded_rectangle((W - bw - 2 * pad - 12, 12, W - 12, 46), 8,
                            fill=(11, 11, 11, 150))
        d.text((W - bw - pad - 12, 18), badge, font=f2,
               fill=(255, 255, 255, 235))
    return np.asarray(img.convert("RGB"))


def compose(frame3d: np.ndarray, panel: np.ndarray) -> np.ndarray:
    assert frame3d.shape[0] == panel.shape[0], "heights must match"
    return np.hstack([frame3d, panel])
