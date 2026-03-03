#!/usr/bin/env python3
"""
One-command retarget + visualize for ACCAD → G1.

Usage:
    # List all available clips matching a keyword
    python HST/retargeting/run_retarget.py --list walk_turn

    # Retarget and visualize (picks first match by default)
    python HST/retargeting/run_retarget.py walk_turn_left_90

    # Retarget only (skip visualization)
    python HST/retargeting/run_retarget.py walk_turn_left_90 --no-vis

    # Visualize only (reuse previous retarget output)
    python HST/retargeting/run_retarget.py walk_turn_left_90 --vis-only

    # Use a specific .npz path directly
    python HST/retargeting/run_retarget.py --clip path/to/clip.npz
"""

import argparse
import glob
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
ACCAD_DIR = os.path.join(PROJECT_ROOT, "HST", "legged_gym", "data", "ACCAD")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "HST", "legged_gym", "data", "retarget_out")


def find_clips(keyword: str = "") -> list[tuple[str, str]]:
    """Return (short_name, full_path) for all ACCAD clips matching keyword."""
    pattern = os.path.join(ACCAD_DIR, "**", "*_stageii.npz")
    all_clips = sorted(glob.glob(pattern, recursive=True))

    results = []
    for path in all_clips:
        rel = os.path.relpath(path, ACCAD_DIR)
        basename = os.path.splitext(os.path.basename(path))[0].replace("_stageii", "")
        # Normalize to a clean short name: lowercase, collapse separators
        short = re.sub(r"[_\-\s]+", "_", basename).strip("_").lower()
        if not keyword or keyword.lower() in rel.lower():
            results.append((short, path))
    return results


def list_clips(keyword: str):
    clips = find_clips(keyword)
    if not clips:
        print(f"No clips found matching '{keyword}'")
        return
    print(f"Found {len(clips)} clip(s):")
    for short, path in clips:
        print(f"  {short:45s}  {os.path.relpath(path, PROJECT_ROOT)}")


def resolve_clip(query: str) -> str:
    """Resolve a query string to a clip path. Tries exact path first, then fuzzy match."""
    if os.path.isfile(query):
        return os.path.abspath(query)

    clips = find_clips(query)
    if not clips:
        sys.exit(f"Error: no clip found matching '{query}'. Use --list to browse.")
    if len(clips) > 1:
        print(f"Multiple matches for '{query}':")
        for i, (short, path) in enumerate(clips):
            print(f"  [{i}] {short:45s}  {os.path.relpath(path, PROJECT_ROOT)}")
        try:
            idx = int(input("Pick one (number): "))
            return clips[idx][1]
        except (ValueError, IndexError):
            sys.exit("Invalid selection.")
    return clips[0][1]


def output_paths(clip_path: str) -> tuple[str, str]:
    """Derive output .npy paths from the clip path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    basename = os.path.splitext(os.path.basename(clip_path))[0].replace("_stageii", "")
    short = re.sub(r"[_\-\s]+", "_", basename).strip("_").lower()
    joints = os.path.join(OUTPUT_DIR, f"{short}_g1.npy")
    root = os.path.join(OUTPUT_DIR, f"{short}_g1_root.npy")
    return joints, root


def main():
    parser = argparse.ArgumentParser(
        description="Retarget ACCAD SMPL-X clip to G1 and visualize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Clip search keyword (e.g. 'walk_turn_left_90') or path to .npz",
    )
    parser.add_argument("--clip", type=str, default=None, help="Explicit path to .npz clip")
    parser.add_argument("--list", dest="list_kw", nargs="?", const="", default=None,
                        help="List available clips (optionally filter by keyword)")
    parser.add_argument("--fps", type=int, default=10, help="Target fps (default: 10)")
    parser.add_argument("--no-vis", action="store_true", help="Retarget only, skip visualization")
    parser.add_argument("--vis-only", action="store_true", help="Visualize only (reuse existing .npy)")
    parser.add_argument("--spheres", action="store_true", help="Use sphere geometry in viewer")
    parser.add_argument("--urdf", type=str, default=os.path.join(PROJECT_ROOT, "g1.urdf"))
    parser.add_argument("--no-loop", action="store_true", help="Don't loop the visualization")
    args = parser.parse_args()

    if args.list_kw is not None:
        list_clips(args.list_kw)
        return

    clip_spec = args.clip or args.query
    if clip_spec is None:
        parser.print_help()
        sys.exit(1)

    clip_path = resolve_clip(clip_spec)
    joints_path, root_path = output_paths(clip_path)

    # --- Retarget ---
    if not args.vis_only:
        print(f"\n{'='*60}")
        print(f"RETARGET: {os.path.relpath(clip_path, PROJECT_ROOT)}")
        print(f"{'='*60}")
        from HST.retargeting.retarget_clip import retarget_clip
        retarget_clip(clip_path, joints_path, target_fps=args.fps)
    else:
        if not os.path.isfile(joints_path):
            sys.exit(f"Error: --vis-only but {joints_path} not found. Run retarget first.")
        print(f"Skipping retarget, reusing: {joints_path}")

    # --- Visualize ---
    if not args.no_vis:
        print(f"\n{'='*60}")
        print("VISUALIZE")
        print(f"{'='*60}")
        from HST.retargeting.visualize_g1 import visualize
        visualize(
            joints_path=joints_path,
            root_path=root_path,
            urdf_path=args.urdf,
            fps=float(args.fps),
            loop=not args.no_loop,
            use_spheres=args.spheres,
        )


if __name__ == "__main__":
    main()
