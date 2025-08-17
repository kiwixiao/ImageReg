#!/usr/bin/env python3
"""
Check segmentation continuity and holes
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path

def check_segmentation_continuity(seg_path):
    """Check if segmentation has holes or discontinuities"""
    print(f"\n🔍 CHECKING SEGMENTATION: {seg_path}")
    print("="*60)
    
    # Load segmentation
    seg_sitk = sitk.ReadImage(str(seg_path))
    seg_array = sitk.GetArrayFromImage(seg_sitk)
    
    print(f"📊 Basic Info:")
    print(f"   Size: {seg_sitk.GetSize()}")
    print(f"   Spacing: {seg_sitk.GetSpacing()}")
    print(f"   Array shape: {seg_array.shape}")
    
    # Check label values
    unique_labels = np.unique(seg_array)
    print(f"   Unique labels: {unique_labels}")
    
    # Count non-zero voxels (foreground)
    foreground_voxels = np.sum(seg_array > 0)
    total_voxels = seg_array.size
    foreground_percentage = (foreground_voxels / total_voxels) * 100
    
    print(f"\n📈 Label Statistics:")
    print(f"   Total voxels: {total_voxels}")
    print(f"   Foreground voxels: {foreground_voxels}")
    print(f"   Foreground percentage: {foreground_percentage:.2f}%")
    
    # Check for each label
    for label in unique_labels:
        if label == 0:
            continue
        label_mask = (seg_array == label)
        label_count = np.sum(label_mask)
        label_percentage = (label_count / total_voxels) * 100
        print(f"   Label {label}: {label_count} voxels ({label_percentage:.2f}%)")
    
    # Check connectivity (simple method)
    if len(unique_labels) > 1:  # Has foreground
        # Get largest connected component for main label
        main_label = unique_labels[unique_labels > 0][0]  # First non-zero label
        binary_mask = (seg_array == main_label).astype(np.uint8)
        
        # Use SimpleITK connected components
        binary_sitk = sitk.GetImageFromArray(binary_mask)
        binary_sitk.CopyInformation(seg_sitk)
        
        connected_components = sitk.ConnectedComponent(binary_sitk)
        cc_array = sitk.GetArrayFromImage(connected_components)
        
        num_components = np.max(cc_array)
        print(f"\n🔗 Connectivity Analysis (Label {main_label}):")
        print(f"   Connected components: {num_components}")
        
        if num_components > 1:
            # Find sizes of each component
            component_sizes = []
            for i in range(1, num_components + 1):
                size = np.sum(cc_array == i)
                component_sizes.append(size)
            
            component_sizes.sort(reverse=True)
            largest_component = component_sizes[0]
            largest_percentage = (largest_component / np.sum(binary_mask)) * 100
            
            print(f"   Largest component: {largest_component} voxels ({largest_percentage:.1f}% of label)")
            print(f"   Component sizes: {component_sizes[:5]}...")  # Show top 5
            
            if num_components > 5:
                print(f"   ⚠️  WARNING: {num_components} separate components detected!")
                print(f"      This suggests fragmentation or holes in segmentation")
            else:
                print(f"   ✅ Reasonable number of components: {num_components}")
        else:
            print(f"   ✅ Single connected component - no fragmentation")
    
    # Check for interpolation artifacts (fractional values)
    non_integer_values = seg_array[~np.equal(seg_array, seg_array.astype(int))]
    if len(non_integer_values) > 0:
        print(f"\n⚠️  INTERPOLATION ARTIFACTS DETECTED:")
        print(f"   Non-integer values found: {len(non_integer_values)} voxels")
        print(f"   Sample values: {non_integer_values[:10]}")
        print(f"   This indicates linear interpolation was used instead of nearest neighbor!")
    else:
        print(f"\n✅ No interpolation artifacts - all values are integers")
    
    return {
        'total_voxels': total_voxels,
        'foreground_voxels': foreground_voxels,
        'unique_labels': unique_labels.tolist(),
        'num_components': num_components if len(unique_labels) > 1 else 0,
        'has_interpolation_artifacts': len(non_integer_values) > 0
    }

def main():
    """Check segmentations in outputs directory"""
    
    output_dir = Path("outputs_OSAMRI016")
    
    # Check original segmentation
    original_seg = Path("inputs_OSAMRI016/OSAMRI016_2501_airway_seg.nii.gz")
    if original_seg.exists():
        check_segmentation_continuity(original_seg)
    
    # Check moved segmentation
    moved_seg = output_dir / "static_seg_moved_to_frame0.nii.gz"
    if moved_seg.exists():
        check_segmentation_continuity(moved_seg)
    else:
        print(f"\n❌ Moved segmentation not found: {moved_seg}")
        print("   Run registration first to generate moved segmentation")

if __name__ == "__main__":
    main()