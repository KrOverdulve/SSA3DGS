"""
Offline preprocessing: split COLMAP scenes into a 7/8 watermarked train set
and a 1/8 clean test set, and rewrite the COLMAP calibration so that image
names match the new files.

Iterates over (artefact, scene) pairs and writes one dataset per combination:

    <output_root>/<artefact>/<scene>/
        images/        corr_<basename>.jpg     (training, watermarked, re-encoded)
                       clean_<original_name>   (test, byte-for-byte copy)
        sparse/0/      cameras / images / points3D with image names updated
        watermark.png  recovered RGBA preview of the artefact

Test images do not receive the watermark: the system must remove the artefact,
so the held-out evaluation must be clean.
"""

import argparse
import os
import shutil
import struct
import sys
from pathlib import Path

import albumentations as A
import cv2
import imageio.v3 as iio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transforms.Dashboard import Dashboard
from transforms.DrawButton import DrawButton
from transforms.Watermark import Watermark


def read_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num_reg_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<dddd", f.read(32))
            tvec = struct.unpack("<ddd", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name_bytes = bytearray()
            while True:
                b = f.read(1)
                if b == b"\x00" or b == b"":
                    break
                name_bytes.extend(b)
            name = name_bytes.decode("utf-8")
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            points2D_blob = f.read(num_points2D * 24)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "num_points2D": num_points2D,
                "points2D_blob": points2D_blob,
            }
    return images


def write_images_bin(images, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, img in images.items():
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<dddd", *img["qvec"]))
            f.write(struct.pack("<ddd", *img["tvec"]))
            f.write(struct.pack("<I", img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", img["num_points2D"]))
            f.write(img["points2D_blob"])


def read_images_txt(path):
    images = {}
    with open(path, "r") as f:
        raw = [l.rstrip("\n") for l in f.readlines()]
    payload = [l for l in raw if l.strip() and not l.lstrip().startswith("#")]
    if len(payload) % 2 != 0:
        raise ValueError(f"{path}: expected an even number of non-comment lines")
    for i in range(0, len(payload), 2):
        header = payload[i].split()
        image_id = int(header[0])
        qvec = tuple(float(x) for x in header[1:5])
        tvec = tuple(float(x) for x in header[5:8])
        camera_id = int(header[8])
        name = " ".join(header[9:])
        points_line = payload[i + 1]
        images[image_id] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
            "points_line": points_line,
        }
    return images


def write_images_txt(images, path):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for image_id, img in images.items():
            q = " ".join(repr(x) for x in img["qvec"])
            t = " ".join(repr(x) for x in img["tvec"])
            f.write(f"{image_id} {q} {t} {img['camera_id']} {img['name']}\n")
            f.write(f"{img['points_line']}\n")


def build_transform(artifact, watermark_path, alpha):
    if artifact == "deadpix":
        return A.SaltAndPepper(amount=(0.01, 0.01), salt_vs_pepper=(0.2, 0.8), p=1.0)
    if artifact == "rain":
        return A.Spatter(mode="rain", mean=(0.65, 0.65), std=(0.8, 0.8),
                         gauss_sigma=(4, 4), intensity=(0.6, 0.6), p=1.0)
    if artifact == "mud":
        return A.Spatter(mode="mud", mean=(0.65, 0.65), std=(0.8, 0.8),
                         gauss_sigma=(4, 4), intensity=(0.6, 0.6), p=1.0)
    if artifact == "noise":
        return A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)
    if artifact == "watermark":
        return Watermark(watermark_path=watermark_path, alpha=alpha, p=1.0)
    if artifact == "ui":
        return DrawButton(p=1.0)
    if artifact == "dashboard":
        return Dashboard()
    if artifact == "clean":
        # No-artifact identity pass: the corr_ training images become the clean
        # images re-encoded exactly like every other category, so the ONLY
        # difference vs. the corrupted runs is the absence of an artifact. This
        # is the Clean (Ref) upper bound, kept on 360_v2_prep for reproducibility.
        return A.NoOp(p=1.0)
    raise ValueError(f"Unknown artifact: {artifact}")


def derive_watermark_rgba(replay_params, shape):
    """Recover the artefact as an RGBA layer by replaying it on a black and a white reference.

    For a linear blend `result = (1-α)·image + α·c` the system
        r0 = α·c              (image = 0)
        r1 = 255 − α·(255−c)  (image = 255)
    is solvable per-channel: α = (r0 + 255 − r1) / 255, c = r0 / α.

    For non-linear transforms (noise/spatter) the recovered alpha is an approximation of
    visibility, which is what you want for a "semitransparent watermark" preview anyway.
    """
    black = np.zeros(shape, dtype=np.uint8)
    white = np.full(shape, 255, dtype=np.uint8)
    r0 = A.ReplayCompose.replay(replay_params, image=black)["image"]
    r1 = A.ReplayCompose.replay(replay_params, image=white)["image"]
    if r0.dtype != np.uint8:
        r0 = np.clip(r0, 0, 255).astype(np.uint8)
    if r1.dtype != np.uint8:
        r1 = np.clip(r1, 0, 255).astype(np.uint8)

    r0f = r0.astype(np.float32)
    r1f = r1.astype(np.float32)
    alpha_per_ch = np.clip((r0f + 255.0 - r1f) / 255.0, 0.0, 1.0)  # (H, W, C)

    # One alpha channel for the PNG: take the max disturbance across colour channels.
    alpha = alpha_per_ch.max(axis=-1)  # (H, W) in [0, 1]

    # Solve c = r0 / α per channel; safe-divide where α is ~0 (no watermark contribution).
    safe_alpha = np.maximum(alpha_per_ch, 1e-3)
    rgb = np.clip(r0f / safe_alpha, 0, 255)
    rgb[alpha_per_ch < 1e-3] = 0  # undefined where alpha vanishes; black is harmless

    rgba = np.empty((shape[0], shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb.astype(np.uint8)
    rgba[..., 3] = (alpha * 255.0).astype(np.uint8)
    return rgba


def prefixed_name(orig_name, prefix, force_jpg):
    """Return new relative path: prefix applied to basename, optional .jpg."""
    parent = os.path.dirname(orig_name)
    base = os.path.basename(orig_name)
    stem, ext = os.path.splitext(base)
    new_base = f"{prefix}_{stem}.jpg" if force_jpg else f"{prefix}_{base}"
    return os.path.join(parent, new_base) if parent else new_base


# Scenes and supported artefact types — match ssa3dgs.sh and the paper benchmark.
DEFAULT_SCENES = ["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]
DEFAULT_ARTIFACTS = ["watermark", "ui", "dashboard", "rain", "mud"]

# Per-scene downsample factor from ssa3dgs.sh. Falls back to 4 for unknown scenes
# (the else branch in ssa3dgs.sh).
SCENE_FACTORS = {
    "bonsai": 2, "counter": 2, "kitchen": 2, "room": 2,
    "garden": 4, "bicycle": 4, "stump": 4,
}


def _list_rel_paths(root):
    """All file paths inside root, expressed relative to root, sorted."""
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            out.append(os.path.relpath(os.path.join(dp, fn), root))
    out.sort()
    return out


def process_scene(input_dir, output_dir, artifact, watermark_path, alpha,
                  test_every, jpg_quality, seed, watermark_out, factor):
    """Build a watermarked/clean COLMAP dataset under output_dir for one (scene, artefact).

    Reads images from <input_dir>/images_<factor>/ (or <input_dir>/images/ for factor=1)
    and maps COLMAP names → actual downsampled filenames using the same sorted-zip
    convention as datasets/colmap.py:250-254, so file-extension changes during
    downscaling (e.g. .JPG → .png) are handled.
    """

    # Locate input colmap dir, mirror the layout on output
    in_sparse_0 = os.path.join(input_dir, "sparse", "0")
    in_sparse_flat = os.path.join(input_dir, "sparse")
    if os.path.exists(in_sparse_0):
        in_colmap = in_sparse_0
        out_colmap = os.path.join(output_dir, "sparse", "0")
    elif os.path.exists(in_sparse_flat):
        in_colmap = in_sparse_flat
        out_colmap = os.path.join(output_dir, "sparse")
    else:
        raise FileNotFoundError(f"No sparse/ folder under {input_dir}")

    colmap_image_dir = os.path.join(input_dir, "images")  # canonical names listed in COLMAP (optional)
    suffix = "" if factor == 1 else f"_{factor}"
    in_images = os.path.join(input_dir, "images" + suffix)
    out_images = os.path.join(output_dir, "images")
    if not os.path.isdir(in_images):
        raise FileNotFoundError(f"Missing downsampled image folder: {in_images}")

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_colmap, exist_ok=True)

    have_bin = os.path.exists(os.path.join(in_colmap, "images.bin"))
    have_txt = os.path.exists(os.path.join(in_colmap, "images.txt"))
    if not (have_bin or have_txt):
        raise FileNotFoundError(f"No images.bin or images.txt under {in_colmap}")

    if have_bin:
        images_meta = read_images_bin(os.path.join(in_colmap, "images.bin"))
    else:
        images_meta = read_images_txt(os.path.join(in_colmap, "images.txt"))

    # Build COLMAP-name → on-disk-name mapping.
    # If <input_dir>/images/ exists we replicate the sorted-zip from datasets/colmap.py:250-254.
    # Otherwise (the canonical full-res folder isn't shipped) we match each COLMAP entry to its
    # downsampled counterpart by stem (e.g. "IMG_001.JPG" ↔ "IMG_001.png").
    image_files = _list_rel_paths(in_images)
    if os.path.isdir(colmap_image_dir) and os.path.abspath(colmap_image_dir) != os.path.abspath(in_images):
        colmap_files = _list_rel_paths(colmap_image_dir)
        if len(colmap_files) != len(image_files):
            raise ValueError(
                f"Image count mismatch: {len(colmap_files)} in {colmap_image_dir} "
                f"vs {len(image_files)} in {in_images}"
            )
        colmap_to_image = dict(zip(colmap_files, image_files))
    else:
        by_stem = {}
        for rel in image_files:
            stem = os.path.splitext(rel)[0]
            if stem in by_stem:
                raise ValueError(
                    f"Cannot stem-match: {in_images} has multiple files with stem '{stem}' "
                    f"({by_stem[stem]!r} and {rel!r}). Restore {colmap_image_dir} as a reference."
                )
            by_stem[stem] = rel
        colmap_to_image = {}
        for meta in images_meta.values():
            cname = meta["name"]
            stem = os.path.splitext(cname)[0]
            if stem not in by_stem:
                raise FileNotFoundError(
                    f"COLMAP entry '{cname}' has no file with stem '{stem}' under {in_images}"
                )
            colmap_to_image[cname] = by_stem[stem]

    sorted_ids = sorted(images_meta.keys(), key=lambda k: images_meta[k]["name"])
    test_ids = {img_id for i, img_id in enumerate(sorted_ids) if i % test_every == 0}

    print(f"  [split] {len(sorted_ids)} images (factor={factor}, src={in_images}) "
          f"-> {len(sorted_ids) - len(test_ids)} train (corr_), {len(test_ids)} test (clean_)")

    np.random.seed(seed)
    transform = A.ReplayCompose([build_transform(artifact, watermark_path, alpha)])
    replay_params = None

    rename_map = {}
    first_train_shape = None
    for idx, image_id in enumerate(sorted_ids):
        colmap_name = images_meta[image_id]["name"]
        if colmap_name not in colmap_to_image:
            raise FileNotFoundError(
                f"COLMAP image '{colmap_name}' has no counterpart in {in_images} "
                f"(sorted-zip mapping mismatch — does {colmap_image_dir} contain the same files?)"
            )
        actual_relpath = colmap_to_image[colmap_name]
        src = os.path.join(in_images, actual_relpath)
        if not os.path.exists(src):
            raise FileNotFoundError(f"Image not found on disk: {src}")

        is_test = image_id in test_ids
        if is_test:
            new_name = prefixed_name(actual_relpath, "clean", force_jpg=False)
            dst = os.path.join(out_images, new_name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        else:
            new_name = prefixed_name(actual_relpath, "corr", force_jpg=True)
            dst = os.path.join(out_images, new_name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            img = iio.imread(src)[..., :3]
            if first_train_shape is None:
                first_train_shape = img.shape
            if replay_params is None:
                out = transform(image=img)
                replay_params = out["replay"]
                img_out = out["image"]
            else:
                img_out = A.ReplayCompose.replay(replay_params, image=img)["image"]
            if img_out.dtype != np.uint8:
                img_out = np.clip(img_out, 0, 255).astype(np.uint8)
            cv2.imwrite(dst, cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])

        rename_map[image_id] = new_name
        if (idx + 1) % 50 == 0 or idx + 1 == len(sorted_ids):
            print(f"    [{idx + 1}/{len(sorted_ids)}] {colmap_name} ({actual_relpath}) -> {new_name}")

    for image_id, new_name in rename_map.items():
        images_meta[image_id]["name"] = new_name

    for fname in ("cameras.bin", "cameras.txt", "points3D.bin", "points3D.txt"):
        src = os.path.join(in_colmap, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_colmap, fname))

    if have_bin:
        write_images_bin(images_meta, os.path.join(out_colmap, "images.bin"))
    if have_txt:
        if have_bin:
            txt_meta = read_images_txt(os.path.join(in_colmap, "images.txt"))
            for image_id, new_name in rename_map.items():
                if image_id in txt_meta:
                    txt_meta[image_id]["name"] = new_name
            write_images_txt(txt_meta, os.path.join(out_colmap, "images.txt"))
        else:
            write_images_txt(images_meta, os.path.join(out_colmap, "images.txt"))

    extconf = os.path.join(input_dir, "ext_metadata.json")
    if os.path.exists(extconf):
        shutil.copy2(extconf, os.path.join(output_dir, "ext_metadata.json"))

    if artifact != "clean" and watermark_out != "" and replay_params is not None and first_train_shape is not None:
        wm_path = watermark_out or os.path.join(output_dir, "watermark.png")
        os.makedirs(os.path.dirname(wm_path) or ".", exist_ok=True)
        rgba = derive_watermark_rgba(replay_params, first_train_shape)
        iio.imwrite(wm_path, rgba, extension=".png")
        print(f"  [watermark] {wm_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes_root", required=True,
                    help="Root folder containing one subdirectory per scene "
                         "(e.g. <root>/garden/, <root>/bicycle/, ...).")
    ap.add_argument("--output_root", required=True,
                    help="Bulk output root. Datasets are written under "
                         "<output_root>/<artifact>/<scene>/.")
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES,
                    help=f"Scene names (subdirs of --scenes_root). Default: {' '.join(DEFAULT_SCENES)}")
    ap.add_argument("--artifacts", nargs="+", default=DEFAULT_ARTIFACTS,
                    choices=DEFAULT_ARTIFACTS + ["clean"],
                    help=f"Artefact types to apply. Default: {' '.join(DEFAULT_ARTIFACTS)}. "
                         "Use 'clean' for the no-artifact Clean (Ref) folder.")
    ap.add_argument("--watermark_path", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png"),
                    help="Path to watermark logo (only used when 'watermark' is in --artifacts)")
    ap.add_argument("--alpha", type=float, default=0.4, help="Watermark alpha (--artifacts=watermark only)")
    ap.add_argument("--test_every", type=int, default=8, help="Hold out every Nth sorted image as test")
    ap.add_argument("--jpg_quality", type=int, default=95, help="JPEG quality for corrupted train images")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for the artefact transform")
    ap.add_argument("--data_factor", type=int, default=None,
                    help="Override per-scene downsample factor. By default the value from "
                         "ssa3dgs.sh is used (2 for bonsai/counter/kitchen/room, 4 for the rest).")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip a (scene, artefact) pair when its output dir already exists.")
    args = ap.parse_args()

    failures = []
    pairs = [(a, s) for a in args.artifacts for s in args.scenes]
    for n, (artifact, scene) in enumerate(pairs, 1):
        input_dir = os.path.join(args.scenes_root, scene)
        output_dir = os.path.join(args.output_root, artifact, scene)
        factor = args.data_factor if args.data_factor is not None else SCENE_FACTORS.get(scene, 4)
        header = f"[{n}/{len(pairs)}] artifact={artifact} scene={scene} factor={factor}"
        if not os.path.isdir(input_dir):
            print(f"{header} -- SKIP (missing input {input_dir})")
            failures.append((artifact, scene, "missing input"))
            continue
        if args.skip_existing and os.path.isdir(output_dir):
            print(f"{header} -- SKIP (output exists {output_dir})")
            continue
        print(f"{header} -> {output_dir}")
        try:
            process_scene(
                input_dir=input_dir,
                output_dir=output_dir,
                artifact=artifact,
                watermark_path=args.watermark_path,
                alpha=args.alpha,
                test_every=args.test_every,
                jpg_quality=args.jpg_quality,
                seed=args.seed,
                watermark_out=None,  # falls back to <output_dir>/watermark.png
                factor=factor,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            failures.append((artifact, scene, str(e)))

    print()
    print(f"[bulk] {len(pairs) - len(failures)}/{len(pairs)} succeeded")
    if failures:
        print("[bulk] failures:")
        for artifact, scene, msg in failures:
            print(f"  - {artifact}/{scene}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
