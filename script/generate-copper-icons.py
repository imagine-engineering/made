#!/usr/bin/env python3
"""Export the supplied Copper artwork. Requires Pillow 12.3; no runtime dependency."""

import argparse
import base64
from collections import deque
import hashlib
import io
import json
from pathlib import Path
import struct

import PIL
from PIL import Image, ImageChops, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/product_flavors/rust"
SOURCE_HASH = "e314675e6364000e9ef11961a1d7909c4618273e7617d3afcb1051b96abf7ea5"
SIZES = (16, 24, 32, 48, 64, 128, 256, 512, 1024)
APPLICATION_CROP = (24, 8, 1230, 1214)
MASCOT_CROP = (148, 126, 1106, 1084)


def components(mask):
    width, height = mask.size
    remaining = bytearray(mask.tobytes())
    regions = []
    for index, value in enumerate(remaining):
        if not value:
            continue
        remaining[index] = 0
        pending = deque([index])
        count = 0
        left, top, right, bottom = width, height, 0, 0
        while pending:
            current = pending.popleft()
            column, row = current % width, current // width
            count += 1
            left, top = min(left, column), min(top, row)
            right, bottom = max(right, column), max(bottom, row)
            for neighbor in (
                current - 1 if column else -1,
                current + 1 if column + 1 < width else -1,
                current - width if row else -1,
                current + width if row + 1 < height else -1,
            ):
                if neighbor >= 0 and remaining[neighbor]:
                    remaining[neighbor] = 0
                    pending.append(neighbor)
        regions.append({"pixels": count, "bounds": [left, top, right + 1, bottom + 1]})
    return sorted(regions, key=lambda region: region["pixels"], reverse=True)


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, "PNG", optimize=False, compress_level=9)
    return stream.getvalue()


def svg(image):
    encoded = base64.b64encode(png_bytes(image)).decode()
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" '
        'viewBox="0 0 1024 1024">\n'
        '  <title>Copper</title>\n'
        '  <image width="1024" height="1024" '
        f'href="data:image/png;base64,{encoded}"/>\n</svg>\n'
    ).encode()


def connected_region(mask, seed):
    width, height = mask.size
    values = mask.tobytes()
    selected = bytearray(len(values))
    start = seed[1] * width + seed[0]
    if not values[start]:
        raise RuntimeError("Artwork mask seed is outside the expected region")
    selected[start] = 255
    pending = deque([start])
    while pending:
        current = pending.popleft()
        column, row = current % width, current // width
        for neighbor in (
            current - 1 if column else -1,
            current + 1 if column + 1 < width else -1,
            current - width if row else -1,
            current + width if row + 1 < height else -1,
        ):
            if neighbor >= 0 and values[neighbor] and not selected[neighbor]:
                selected[neighbor] = 255
                pending.append(neighbor)
    return Image.frombytes("L", mask.size, bytes(selected))


def artwork():
    if PIL.__version__ != "12.3.0":
        raise RuntimeError("Use Pillow 12.3.0 for reproducible Copper exports")
    original = (ASSETS / "source/copper-mascot-tile-original.png").read_bytes()
    if hashlib.sha256(original).hexdigest() != SOURCE_HASH:
        raise RuntimeError("Original Copper artwork differs from the supplied image")
    source = Image.open(io.BytesIO(original)).convert("RGB")
    if source.size != (1254, 1254):
        raise RuntimeError("Unexpected original dimensions")
    copper = Image.new("L", source.size)
    copper.putdata([
        255 if max(red, green, blue) >= 45 else 0
        for red, green, blue in source.get_flattened_data()
    ])
    # Only the corner-connected exterior is removed. Enclosed eyes, reflections
    # and mascot pixels must remain opaque even though they are not copper.
    exterior = connected_region(ImageChops.invert(copper), (0, 0))
    tile_mask = ImageChops.invert(exterior)
    white = Image.new("L", source.size)
    white.putdata([
        255 if min(pixel) >= 170 and max(pixel) - min(pixel) <= 60 else 0
        for pixel in source.get_flattened_data()
    ])
    body = connected_region(white, (627, 500))
    # Filling only this body's enclosed holes retains the original eye RGB,
    # including both white reflections, rather than punching out dark pixels.
    mascot_mask = ImageChops.invert(
        connected_region(ImageChops.invert(body), (0, 0))
    )
    for name, mask in (("tile", tile_mask), ("mascot", mascot_mask)):
        if len([item for item in components(mask) if item["pixels"] >= 100]) != 1:
            raise RuntimeError(f"Expected one connected {name} silhouette")
    if ImageChops.subtract(mascot_mask, tile_mask).getbbox():
        raise RuntimeError("Mascot mask extends beyond the tile")
    for point in ((470, 710), (730, 710), (495, 665), (750, 665)):
        if mascot_mask.getpixel(point) != 255:
            raise RuntimeError("Mascot extraction lost an eye or reflection")
    primary = source.convert("RGBA")
    primary.putalpha(tile_mask)
    primary = primary.crop(APPLICATION_CROP)
    mascot = source.convert("RGBA")
    mascot.putalpha(mascot_mask)
    mascot = mascot.crop(MASCOT_CROP)
    return source, primary, mascot, tile_mask, mascot_mask


def exports():
    source, primary, mascot, tile_mask, mascot_mask = artwork()
    result = {}
    applications = {}
    for size in SIZES:
        # One alpha-aware downsample from the source avoids repeated blur and
        # preserves the supplied smooth tile and mascot curves at small sizes.
        applications[size] = primary.resize((size, size), Image.Resampling.LANCZOS)
        logo = mascot.resize((size, size), Image.Resampling.LANCZOS)
        result[f"icons/app-icon-{size}.png"] = png_bytes(applications[size])
        result[f"logos/logo-{size}.png"] = png_bytes(logo)
    result["app-icon.png"] = result["icons/app-icon-512.png"]
    result["app-icon@2x.png"] = result["icons/app-icon-1024.png"]
    result["app-icon.svg"] = svg(applications[1024])
    result["logo.png"] = result["logos/logo-1024.png"]
    result["logo.svg"] = svg(mascot.resize((1024, 1024), Image.Resampling.LANCZOS))
    icon = io.BytesIO()
    applications[256].save(
        icon, "ICO", sizes=[(size, size) for size in SIZES if size <= 256],
        append_images=[applications[size] for size in SIZES if size < 256],
    )
    result["app-icon.ico"] = icon.getvalue()
    # Explicit PNG representations avoid an ICNS writer silently resampling the
    # already size-specific exports. icp4 also supplies the native 16px slot.
    chunks = []
    for kind, size in ((b"icp4", 16), (b"icp5", 32), (b"icp6", 64),
                       (b"ic07", 128), (b"ic08", 256), (b"ic09", 512),
                       (b"ic10", 1024), (b"ic11", 32), (b"ic12", 64),
                       (b"ic13", 256), (b"ic14", 512)):
        contents = result[f"icons/app-icon-{size}.png"]
        chunks.append(kind + struct.pack(">I", len(contents) + 8) + contents)
    contents = b"".join(chunks)
    result["app-icon.icns"] = b"icns" + struct.pack(">I", len(contents) + 8) + contents
    manifest = {
        "source": "source/copper-mascot-tile-original.png", "source_sha256": SOURCE_HASH,
        "source_dimensions": list(source.size),
        "application_crop": list(APPLICATION_CROP), "mascot_crop": list(MASCOT_CROP),
        "tile_bounds": list(tile_mask.getbbox()), "mascot_bounds": list(mascot_mask.getbbox()),
        "mask_method": "corner-connected exterior only; mascot connected white body with enclosed eyes filled",
        "retained_source_rgb": "unchanged before resizing",
        "resampling": "single-pass alpha-aware Lanczos", "pillow_version": PIL.__version__,
        "small_size_processing": "sampling only; no redrawing, sharpening, or substituted eyes",
        "exports": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in sorted(result.items())},
    }
    result["generated-icons.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return result


def contact_sheet(destination, before):
    sheet = Image.new("RGB", (1800, 1650), "#edeae5")
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 16), "COPPER / supplied mascot tile / application and standalone mascot", fill="#302921")
    old = Image.open(before).convert("RGBA")
    for column, (label, image) in enumerate((
        ("Before", old), ("Application", Image.open(ASSETS / "app-icon@2x.png")),
        ("Transparent / light", Image.open(ASSETS / "logo.png")),
        ("Transparent / dark", Image.open(ASSETS / "logo.png")),
    )):
        left = 24 + column * 344
        draw.text((left, 48), label, fill="#302921")
        tile = Image.new("RGBA", (320, 320), "#101211" if column == 3 else "#ffffff")
        tile.alpha_composite(image.convert("RGBA").resize((300, 300), Image.Resampling.NEAREST), (10, 10))
        sheet.paste(tile.convert("RGB"), (left, 70))
    top = 414
    for size in SIZES:
        draw.text((24, top + 8), f"{size}px", fill="#302921")
        for column, (directory, color) in enumerate((("icons", "#ffffff"), ("icons", "#101211"), ("logos", "#ffffff"), ("logos", "#101211"))):
            filename = f"app-icon-{size}.png" if directory == "icons" else f"logo-{size}.png"
            image = Image.open(ASSETS / directory / filename).convert("RGBA")
            # Full 1024px exports remain available separately; large previews
            # fit here, and 16-128px samples are displayed at actual pixels.
            preview_size = min(size, 128)
            tile = Image.new("RGBA", (360, 132), color)
            tile.alpha_composite(image.resize((preview_size, preview_size), Image.Resampling.NEAREST), (8, 2))
            if size <= 64:
                enlarged = image.resize((min(size * 4, 128),) * 2, Image.Resampling.NEAREST)
                tile.alpha_composite(enlarged, (168, 2))
            sheet.paste(tile.convert("RGB"), (160 + column * 400, top))
        top += 132
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--before", type=Path, help="Previous app-icon PNG for the comparison sheet")
    arguments = parser.parse_args()
    generated = exports()
    for name, data in generated.items():
        destination = ASSETS / name
        if arguments.check:
            if not destination.is_file() or destination.read_bytes() != data:
                raise RuntimeError(f"Stale Copper export: {destination}")
        elif not destination.is_file() or destination.read_bytes() != data:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
    if arguments.contact_sheet:
        if arguments.before is None:
            parser.error("--contact-sheet requires --before")
        contact_sheet(arguments.contact_sheet, arguments.before)
    print(f"{'Verified' if arguments.check else 'Generated'} {len(generated)} Copper exports; original SHA-256 {SOURCE_HASH}")


if __name__ == "__main__":
    main()
