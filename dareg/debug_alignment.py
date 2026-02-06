#!/usr/bin/env python3
"""Debug script to compare dynamic frame and seg frame grid orientations"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path

def compare_nifti_files(frame_path, seg_path):
    """Compare frame and segmentation NIfTI files"""

    print("=" * 70)
    print("COMPARING FRAME AND SEGMENTATION ORIENTATIONS")
    print("=" * 70)

    # Load files
    frame_nii = nib.load(str(frame_path))
    seg_nii = nib.load(str(seg_path))

    frame_data = frame_nii.get_fdata()
    seg_data = seg_nii.get_fdata()

    print(f"\n1. FILE PATHS:")
    print(f"   Frame: {frame_path.name}")
    print(f"   Seg:   {seg_path.name}")

    print(f"\n2. DATA SHAPES:")
    print(f"   Frame shape: {frame_data.shape}")
    print(f"   Seg shape:   {seg_data.shape}")

    print(f"\n3. AFFINE MATRICES:")
    print(f"   Frame affine:\n{frame_nii.affine}")
    print(f"   Seg affine:\n{seg_nii.affine}")

    # Check if affines are identical
    affine_diff = np.abs(frame_nii.affine - seg_nii.affine).max()
    print(f"\n   Max affine difference: {affine_diff:.6f}")

    print(f"\n4. DATA STATISTICS:")
    print(f"   Frame: min={frame_data.min():.3f}, max={frame_data.max():.3f}, mean={frame_data.mean():.3f}")
    print(f"   Seg:   min={seg_data.min():.0f}, max={seg_data.max():.0f}, unique labels={np.unique(seg_data).astype(int).tolist()}")

    print(f"\n5. NON-ZERO REGION ANALYSIS:")
    # Find bounding box of non-zero regions
    frame_nz = np.where(frame_data > frame_data.mean())
    seg_nz = np.where(seg_data > 0)

    if len(frame_nz[0]) > 0:
        frame_bounds = [(arr.min(), arr.max()) for arr in frame_nz]
        print(f"   Frame non-zero bounds (X,Y,Z): {frame_bounds}")
    else:
        print(f"   Frame: No significant non-zero region")

    if len(seg_nz[0]) > 0:
        seg_bounds = [(arr.min(), arr.max()) for arr in seg_nz]
        print(f"   Seg non-zero bounds (X,Y,Z):   {seg_bounds}")
    else:
        print(f"   Seg: No non-zero region")

    print(f"\n6. CENTER OF MASS COMPARISON:")
    # Compute center of mass
    if len(frame_nz[0]) > 0:
        frame_com = [arr.mean() for arr in frame_nz]
        print(f"   Frame center of mass (voxels): [{frame_com[0]:.1f}, {frame_com[1]:.1f}, {frame_com[2]:.1f}]")
    if len(seg_nz[0]) > 0:
        seg_com = [arr.mean() for arr in seg_nz]
        print(f"   Seg center of mass (voxels):   [{seg_com[0]:.1f}, {seg_com[1]:.1f}, {seg_com[2]:.1f}]")

    print(f"\n7. SLICE-BY-SLICE CHECK (middle slices):")
    # Check middle slices
    mid_x = frame_data.shape[0] // 2
    mid_y = frame_data.shape[1] // 2
    mid_z = frame_data.shape[2] // 2

    print(f"   Checking slice at Z={mid_z}:")
    frame_slice = frame_data[:, :, mid_z]
    seg_slice = seg_data[:, :, mid_z]
    print(f"     Frame slice non-zero pixels: {(frame_slice > frame_slice.mean()).sum()}")
    print(f"     Seg slice non-zero pixels:   {(seg_slice > 0).sum()}")

    # Check if there's any overlap
    frame_mask = frame_slice > frame_slice.mean()
    seg_mask = seg_slice > 0
    overlap = (frame_mask & seg_mask).sum()
    print(f"     Overlap (frame>mean AND seg>0): {overlap} pixels")

    print(f"\n8. ORIENTATION TESTS:")
    # Test different axis permutations
    print("   Testing if seg data needs axis swap to match frame:")
    for perm_name, perm in [
        ("original", (0, 1, 2)),
        ("swap XY", (1, 0, 2)),
        ("swap XZ", (2, 1, 0)),
        ("swap YZ", (0, 2, 1)),
        ("XYZ->YXZ", (1, 0, 2)),
        ("XYZ->ZYX", (2, 1, 0)),
    ]:
        seg_perm = np.transpose(seg_data, perm)
        if seg_perm.shape == frame_data.shape:
            # Check center of mass alignment
            seg_perm_nz = np.where(seg_perm > 0)
            if len(seg_perm_nz[0]) > 0:
                seg_perm_com = [arr.mean() for arr in seg_perm_nz]
                if len(frame_nz[0]) > 0:
                    frame_com = [arr.mean() for arr in frame_nz]
                    dist = np.sqrt(sum((a-b)**2 for a, b in zip(frame_com, seg_perm_com)))
                    print(f"     {perm_name}: COM distance = {dist:.1f} voxels")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Default paths - adjust as needed
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("motion_outputs")

    frames_dir = output_dir / "frames"
    seg_dir = output_dir / "segmentations_frame_grid"

    # Find matching pairs
    # New naming: dynamic_extracted_f###_t###.nii.gz
    frame_files = sorted(frames_dir.glob("dynamic_extracted_f*_t*.nii.gz"))

    if not frame_files:
        print(f"No frame files found in {frames_dir}")
        sys.exit(1)

    import re
    for frame_file in frame_files[:3]:  # Check first 3 frames
        # Parse frame indices from filename like "dynamic_extracted_f003_t001.nii.gz"
        # f### = absolute (1-indexed), t### = relative (1-indexed)
        match = re.search(r'f(\d+)_t(\d+)', frame_file.stem)
        if not match:
            print(f"Could not parse frame indices from {frame_file.name}")
            continue
        abs_idx = int(match.group(1))
        rel_idx = int(match.group(2))
        # Segmentation file uses same naming pattern
        seg_file = seg_dir / f"seg_f{abs_idx:03d}_t{rel_idx:03d}_frame_grid.nii.gz"

        if seg_file.exists():
            compare_nifti_files(frame_file, seg_file)
        else:
            print(f"Missing seg file: {seg_file}")
