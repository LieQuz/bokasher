#!/usr/bin/env python3
"""Generate DMG background images and app/volume icons for macOS packaging."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "desktop" / "build" / "dmg"
ICONSET = OUT / "icon.iconset"

W, H = 600, 420
APP_X, APP_Y = 168, 228
APPS_X, APPS_Y = 432, 228


def load_latin_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSText.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def load_japanese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = "/System/Library/Fonts/Hiragino Sans GB.ttc"
    if Path(path).exists():
        try:
            return ImageFont.truetype(path, size=size, index=2)
        except OSError:
            pass
    return load_latin_font(size)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def draw_vertical_gradient(img: Image.Image) -> None:
    px = img.load()
    top, mid, bottom = (8, 8, 10), (14, 14, 16), (6, 6, 8)
    for y in range(H):
        t = y / max(H - 1, 1)
        if t < 0.45:
            k = t / 0.45
            src, dst = top, mid
        else:
            k = (t - 0.45) / 0.55
            src, dst = mid, bottom
        color = (
            int(lerp(src[0], dst[0], k)),
            int(lerp(src[1], dst[1], k)),
            int(lerp(src[2], dst[2], k)),
        )
        for x in range(W):
            px[x, y] = color


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    color: tuple[int, ...],
) -> None:
    tw = draw.textlength(text, font=font)
    draw.text(((W - tw) / 2, y), text, font=font, fill=color)


def render_background() -> Image.Image:
    base = Image.new("RGB", (W, H))
    draw_vertical_gradient(base)
    img = base.convert("RGBA")

    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((W // 2 - 170, -70, W // 2 + 170, 150), fill=(10, 132, 255, 36))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(radius=28)))

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw_centered_text(draw, 36, "bokasher", load_latin_font(28, bold=True), (245, 245, 247, 255))
    draw_centered_text(draw, 74, "ぼかっしゃー", load_japanese_font(13), (152, 152, 157, 255))

    y = APP_Y
    x0, x1 = APP_X + 58, APPS_X - 58
    draw.line((x0, y, x1 - 8, y), fill=(255, 255, 255, 48), width=2)
    draw.polygon([(x1 - 10, y - 6), (x1, y), (x1 - 10, y + 6)], fill=(255, 255, 255, 90))

    img.alpha_composite(layer)
    return img.convert("RGB")


def rounded_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, ...],
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def render_app_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pad = size // 8
    rounded_rect(draw, (pad, pad, size - pad, size - pad), size // 5, fill=(28, 28, 30, 255))

    cx, cy = size // 2, size // 2 - size // 18
    r = size // 4
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(10, 132, 255, 255))
    blur = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(blur)
    for i, alpha in enumerate((90, 70, 50)):
        spread = r + 8 + i * 10
        bdraw.ellipse(
            (cx - spread, cy - spread + i * 3, cx + spread, cy + spread + i * 3),
            fill=(10, 132, 255, alpha),
        )
    img.alpha_composite(blur.filter(ImageFilter.GaussianBlur(radius=max(2, size // 32))))

    face = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(face).ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 210))
    img.alpha_composite(face.filter(ImageFilter.GaussianBlur(radius=max(3, size // 24))))
    return img


def write_icns(png_1024: Path, icns_path: Path) -> None:
    if ICONSET.exists():
        shutil.rmtree(ICONSET)
    ICONSET.mkdir(parents=True)

    specs = [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ]
    master = Image.open(png_1024)
    for name, dim in specs:
        master.resize((dim, dim), Image.Resampling.LANCZOS).save(ICONSET / name)

    subprocess.run(["iconutil", "-c", "icns", str(ICONSET), "-o", str(icns_path)], check=True)
    shutil.rmtree(ICONSET)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    bg1 = render_background()
    bg2 = bg1.resize((W * 2, H * 2), Image.Resampling.LANCZOS)
    bg1.save(OUT / "background.png", optimize=True)
    bg2.save(OUT / "background@2x.png", optimize=True)

    icon_master = render_app_icon(1024)
    icon_master.save(OUT / "icon.png")
    write_icns(OUT / "icon.png", OUT / "icon.icns")

    print(f"Wrote DMG assets to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
