#!/usr/bin/env python3
"""
Main Sequential Registration Pipeline - Clean Modular Implementation
STEP-BY-STEP: Rigid → Affine → SVFFD with comprehensive debugging

This script follows the first-principles plan:
1. Load images and setup common coordinate space
2. Rigid registration (6 DOF) 
3. Affine registration (12 DOF) - chains from rigid output
4. SVFFD registration (diffeomorphic) - chains from affine output
5. Comprehensive debugging and visualization at each stage
"""

import sys
sys.path.append('/Users/xiaz9n/Dropbox/CCHMCProjects/PythonProjects/ImageReg/deepali/src')

import torch
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
import sys
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our clean modular components
from utils.image_loader import load_image_pair, create_side_by_side_comparison
from utils.pdf_report import create_comprehensive_pdf_report
from utils.segmentation_utils import save_transformed_segmentations, create_segmentation_overlay_visualization
from modules.rigid_registration import RigidRegistration
from modules.affine_registration import AffineRegistration


def main_sequential_registration(source_path: str, target_path: str, output_dir: str, 
                               device: str = "cpu", stage_limit: str = "rigid"):
    """
    Main sequential registration pipeline
    
    Args:
        source_path: Path to source/moving image
        target_path: Path to target/fixed image  
        output_dir: Output directory for all results
        device: Device for computation ("cpu" or "cuda")
        stage_limit: How far to proceed ("rigid", "affine", "svffd")
    """
    
    print("🚀 SEQUENTIAL REGISTRATION PIPELINE - CLEAN IMPLEMENTATION")
    print("=" * 80)
    print(f"📂 Source: {Path(source_path).name}")
    print(f"📂 Target: {Path(target_path).name}")
    print(f"📁 Output: {output_dir}")
    print(f"🖥️  Device: {device}")
    print(f"🎯 Stage limit: {stage_limit}")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # PHASE 1: IMAGE LOADING & COMMON COORDINATE SETUP
    # =========================================================================
    print("\\n📋 PHASE 1: IMAGE LOADING & COMMON COORDINATE SETUP")
    image_pair = load_image_pair(source_path, target_path, device)
    
    # Create initial side-by-side comparison
    create_side_by_side_comparison(
        image_pair.source_normalized,
        image_pair.target_normalized, 
        output_path / "debug_analysis",
        "initial_alignment"
    )
    
    # =========================================================================
    # PHASE 2: RIGID REGISTRATION (6 DOF)
    # =========================================================================
    print("\\n📋 PHASE 2: RIGID REGISTRATION")
    
    rigid_reg = RigidRegistration(device=device, iterations=200, learning_rate=1e-2)
    rigid_result = rigid_reg.register(image_pair)
    
    # Save all rigid results and debug outputs
    rigid_reg.save_intermediate_results(
        image_pair, 
        rigid_result, 
        output_path,
        create_visualizations=True
    )
    
    print(f"\\n✅ RIGID REGISTRATION COMPLETE")
    print(f"   🎯 Final loss: {rigid_result.final_loss:.6f}")
    print(f"   📐 Translation: {rigid_result.translation.tolist()}")
    print(f"   🔄 Rotation: {rigid_result.rotation.tolist()}")
    
    # Stop here if only rigid requested
    if stage_limit == "rigid":
        print(f"\\n🏁 PIPELINE COMPLETE - RIGID ONLY")
        print(f"✅ Results saved in: {output_path}")
        return
    
    # =========================================================================
    # PHASE 3: AFFINE REGISTRATION (12 DOF) - CHAINS FROM RIGID OUTPUT
    # =========================================================================
    if stage_limit in ["affine", "svffd"]:
        print(f"\\n📋 PHASE 3: AFFINE REGISTRATION")
        
        affine_reg = AffineRegistration(device=device)
        affine_result = affine_reg.register(
            rigid_result.source_after_rigid_common,  # Chain from rigid output
            image_pair.target_normalized,
            image_pair.common_grid,
            rigid_result.transform  # Pass rigid transform for initialization
        )
        
        # Save all affine results and debug outputs
        affine_reg.save_intermediate_results(
            image_pair,
            affine_result, 
            output_path,
            create_visualizations=True
        )
        
        print(f"\\n✅ AFFINE REGISTRATION COMPLETE")
        print(f"   🎯 Final loss: {affine_result.final_loss:.6f}")
        print(f"   📐 Matrix determinant: {torch.det(affine_result.matrix[:, :3, :3]).item():.6f}")
        print(f"   🔢 Condition number: {torch.linalg.cond(affine_result.matrix[:, :3, :3]).item():.1f}")
    
    # =========================================================================
    # PHASE 4: SVFFD REGISTRATION (DIFFEOMORPHIC) - COMING LAST  
    # =========================================================================
    if stage_limit == "svffd":
        print(f"\\n⏭️  SVFFD REGISTRATION - TO BE IMPLEMENTED")
        print(f"   Will chain from affine output")
    
    # =========================================================================
    # PHASE 4B: TRANSFORM SEGMENTATION MASKS (IF AVAILABLE)
    # =========================================================================
    print(f"\\n📋 PHASE 4B: SEGMENTATION TRANSFORMATION")
    print(f"   Checking segmentation availability: {image_pair.source_seg_original is not None}")
    
    if image_pair.source_seg_original is not None:
        print("\\n🎯 TRANSFORMING SEGMENTATION MASKS")
        print("=" * 50)
        # Collect transforms based on completed stages
        affine_transform_for_seg = None
        if stage_limit in ["affine", "svffd"] and 'affine_result' in locals():
            affine_transform_for_seg = affine_result.transform
        
        # Transform segmentation masks
        save_transformed_segmentations(
            image_pair, 
            rigid_result.transform, 
            affine_transform_for_seg,
            output_path, 
            stage_limit
        )
        
        # Create segmentation overlay visualization  
        create_segmentation_overlay_visualization(
            image_pair,
            rigid_result.transform,
            affine_transform_for_seg, 
            output_path,
            stage_limit
        )
    
    # =========================================================================
    # PHASE 5: CREATE COMPREHENSIVE PDF REPORT  
    # =========================================================================
    print(f"\\n📄 GENERATING COMPREHENSIVE PDF REPORT")
    try:
        pdf_path = create_comprehensive_pdf_report(output_path, stage_limit)
        print(f"✅ PDF Report: {pdf_path}")
    except Exception as e:
        print(f"⚠️  PDF generation failed: {e}")
        print(f"   Individual PNG files available in: {output_path}/debug_analysis/")
    
    # Final completion message
    print(f"\\n🏁 PIPELINE COMPLETE - {stage_limit.upper()} STAGE")
    print(f"✅ Results saved in: {output_path}")
    
    # Show key outputs
    if image_pair.source_seg_original is not None:
        print(f"🎯 Segmentation transformation completed")
        print(f"   📂 Segmentation results: {output_path}/segmentation_results/")
    print(f"📄 PDF Report: {output_path}/registration_report.pdf")
    print(f"📊 Debug analysis: {output_path}/debug_analysis/")


if __name__ == "__main__":
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Input paths - use 3D_image_test_inputs as specified
    # Look in parent directory since we're running from sequential_reg_clean/
    parent_dir = Path(__file__).parent.parent
    input_dir = parent_dir / "3D_image_test_inputs"
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    print(f"📁 Using input directory: {input_dir}")
    
    # Find source and target files
    nii_files = list(input_dir.glob("*.nii*"))
    if len(nii_files) < 2:
        raise FileNotFoundError(f"Need at least 2 .nii files in {input_dir}")
    
    # Specifically use source.nii and target.nii.gz from 3D_image_test_inputs
    source_path = str(input_dir / "source.nii") 
    target_path = str(input_dir / "target.nii.gz")
    output_dir = "outputs"
    
    print(f"🎯 STARTING WITH RIGID + AFFINE REGISTRATION")
    print(f"📂 Source: {Path(source_path).name}")
    print(f"📂 Target: {Path(target_path).name}")
    
    # Run pipeline - TEST RIGID + AFFINE CHAINING
    main_sequential_registration(
        source_path=source_path,
        target_path=target_path, 
        output_dir=output_dir,
        device=device,
        stage_limit="affine"  # Test rigid → affine chaining
    )