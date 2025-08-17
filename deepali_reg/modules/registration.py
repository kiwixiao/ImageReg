"""Registration module using Deepali's diffeomorphic SVF approach."""

from pathlib import Path
from typing import Dict, Optional, Union, List
import torch
from torch import Tensor

from deepali.core import Grid
from deepali.data import Image
from deepali.losses import NormalizedMutualInformationLoss, BendingEnergyLoss
from deepali.spatial import (
    StationaryVelocityFieldTransform,
    SequentialTransform,
    RigidTransform,
    AffineTransform,
)
from deepali.modules import TransformImage
import torch.optim as optim


def register_pairwise_svf(
    target: Image,
    source: Image,
    config: Dict,
    mask: Optional[Image] = None,
    initial_transform: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    verbose: int = 0,
) -> StationaryVelocityFieldTransform:
    """Register two images using Stationary Velocity Field (diffeomorphic).
    
    This is equivalent to MIRTK's FFD registration but guarantees
    diffeomorphic (smooth, invertible) transformations.
    
    Args:
        target: Fixed target image
        source: Moving source image  
        config: Registration configuration
        mask: Optional mask for constrained registration (e.g., nose rigid)
        initial_transform: Initial transformation or "guess"
        device: Computation device
        verbose: Verbosity level
        
    Returns:
        SVF transformation from target to source space
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move images to device
    target = target.to(device)
    source = source.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    # Get configuration parameters
    transform_config = config.get("transform", {})
    similarity_config = config.get("similarity", {})
    reg_config = config.get("regularization", {})
    optim_config = config.get("optimization", {})
    pyramid_config = config.get("pyramid", {})
    
    # Create multi-resolution pyramid
    levels = pyramid_config.get("levels", 4)
    downsample_factors = pyramid_config.get("downsample_factors", [12, 6, 3, 0])
    
    # Initialize SVF transformation
    transform_grid = target.grid()
    if transform_config.get("control_point_spacing", 1) > 1:
        # Downsample control point grid
        spacing = transform_config["control_point_spacing"]
        transform_grid = transform_grid.downsample(spacing)
    
    # Create SVF transform (diffeomorphic)
    svf = StationaryVelocityFieldTransform(
        grid=transform_grid,
        groups=1,  # Single transformation for all images
        params=True,  # Initialize as optimizable parameters
    )
    svf = svf.to(device)
    
    # Initialize with existing transformation if provided
    if initial_transform == "guess":
        # Simple center-of-mass alignment
        svf = initialize_with_center_alignment(svf, target, source)
    elif initial_transform and Path(initial_transform).exists():
        # Load existing transformation
        svf = load_initial_transform(svf, initial_transform, device)
    
    # Multi-resolution registration
    for level in range(levels):
        if verbose > 0:
            print(f"\nLevel {level + 1}/{levels}")
        
        # Downsample images for current level
        factor = downsample_factors[level] if level < len(downsample_factors) else 0
        if factor > 0:
            target_level = downsample_image(target, factor)
            source_level = downsample_image(source, factor)
            if mask is not None:
                mask_level = downsample_image(mask, factor, mode="nearest")
            else:
                mask_level = None
        else:
            target_level = target
            source_level = source
            mask_level = mask
        
        # Update transform grid for current level
        if level > 0:
            svf.grid_(target_level.grid())
        
        # Setup loss function
        loss_fn = create_registration_loss(
            target_level, source_level, svf,
            similarity_config, reg_config,
            mask=mask_level
        )
        
        # Setup optimizer
        optimizer = create_optimizer(svf, optim_config)
        
        # Run optimization
        max_steps = optim_config.get("max_steps", 100)
        min_delta = optim_config.get("min_delta", 1e-5)
        
        prev_loss = float('inf')
        for step in range(max_steps):
            optimizer.zero_grad()
            
            # Forward pass
            loss = loss_fn()
            
            # Backward pass
            loss.backward()
            
            # Gradient smoothing (if configured)
            if reg_config.get("smooth_grad", 0) > 0:
                smooth_gradients(svf, reg_config["smooth_grad"])
            
            # Optimization step
            optimizer.step()
            
            # Check convergence
            loss_val = loss.item()
            if verbose > 1 and step % 10 == 0:
                print(f"  Step {step:3d}: Loss = {loss_val:.6f}")
            
            if abs(prev_loss - loss_val) < min_delta:
                if verbose > 0:
                    print(f"  Converged at step {step}")
                break
            prev_loss = loss_val
    
    return svf


def create_registration_loss(
    target: Image,
    source: Image,
    transform: StationaryVelocityFieldTransform,
    similarity_config: Dict,
    reg_config: Dict,
    mask: Optional[Image] = None,
):
    """Create composite loss function for registration."""
    
    # Image warping module
    warp = TransformImage(
        target=target.grid(),
        source=source.grid(),
        sampling="linear",
        padding="border",
    )
    
    # Similarity loss (NMI)
    similarity_loss = NormalizedMutualInformationLoss(
        bins=similarity_config.get("bins", 64),
        mask=mask.tensor() if mask else None,
    )
    
    # Regularization losses
    bending_weight = reg_config.get("bending_energy", 0.001)
    
    def compute_loss():
        # Warp source image
        warped_source = warp(transform.tensor(), source.tensor())
        
        # Compute similarity
        sim_loss = similarity_loss(warped_source, target.tensor())
        
        # Compute regularization
        reg_loss = 0
        if bending_weight > 0:
            # Bending energy of the velocity field
            velocity = transform.velocity()
            reg_loss += bending_weight * compute_bending_energy(velocity)
        
        return sim_loss + reg_loss
    
    return compute_loss


def compute_bending_energy(velocity: Tensor) -> Tensor:
    """Compute bending energy of velocity field."""
    # Simplified bending energy using finite differences
    # In practice, you'd compute second derivatives
    dx = velocity[..., 1:, :, :] - velocity[..., :-1, :, :]
    dy = velocity[..., :, 1:, :] - velocity[..., :, :-1, :]
    dz = velocity[..., :, :, 1:] - velocity[..., :, :, :-1]
    
    energy = (dx**2).mean() + (dy**2).mean() + (dz**2).mean()
    return energy


def create_optimizer(transform, optim_config: Dict):
    """Create optimizer for registration."""
    optimizer_type = optim_config.get("optimizer", "LBFGS")
    lr = optim_config.get("lr", 1.0)
    
    if optimizer_type == "LBFGS":
        return optim.LBFGS(
            transform.parameters(),
            lr=lr,
            line_search_fn=optim_config.get("line_search", "strong_wolfe"),
        )
    elif optimizer_type == "Adam":
        return optim.Adam(transform.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def smooth_gradients(transform, sigma: float):
    """Apply Gaussian smoothing to gradients."""
    import torch.nn.functional as F
    
    for param in transform.parameters():
        if param.grad is not None:
            # Simple Gaussian smoothing (you'd implement proper 3D smoothing)
            grad = param.grad.data
            # This is a placeholder - implement actual Gaussian smoothing
            param.grad.data = grad


def downsample_image(image: Image, factor: int, mode: str = "linear") -> Image:
    """Downsample image by given factor."""
    import torch.nn.functional as F
    
    data = image.tensor()
    
    # Simple downsampling using interpolation
    # Add batch dimension if needed
    if data.ndim == 4:  # (C, X, Y, Z)
        data = data.unsqueeze(0)  # (1, C, X, Y, Z)
    
    # Calculate new size
    new_size = tuple(s // factor for s in data.shape[2:])
    
    # Downsample
    if mode == "nearest":
        data_down = F.interpolate(data, size=new_size, mode="nearest")
    else:
        data_down = F.interpolate(data, size=new_size, mode="trilinear", align_corners=True)
    
    # Remove batch dimension
    data_down = data_down.squeeze(0)
    
    # Create new grid
    old_grid = image.grid()
    new_grid = Grid(
        size=new_size,
        spacing=tuple(s * factor for s in old_grid.spacing()),
        origin=old_grid.origin(),
        direction=old_grid.direction(),
        device=old_grid.device(),
    )
    
    return Image(data_down, new_grid)


def initialize_with_center_alignment(
    transform: StationaryVelocityFieldTransform,
    target: Image,
    source: Image,
) -> StationaryVelocityFieldTransform:
    """Initialize transform with center-of-mass alignment."""
    # This is a simplified version - implement actual COM alignment
    return transform


def load_initial_transform(
    transform: StationaryVelocityFieldTransform,
    path: Union[str, Path],
    device: torch.device,
) -> StationaryVelocityFieldTransform:
    """Load initial transformation from file."""
    # Load transformation parameters
    if Path(path).suffix == ".pt":
        state = torch.load(path, map_location=device)
        transform.load_state_dict(state)
    return transform


def compose_transformations(
    transforms: List[StationaryVelocityFieldTransform],
) -> StationaryVelocityFieldTransform:
    """Compose multiple SVF transformations.
    
    For SVFs, composition is done by adding velocity fields.
    """
    if not transforms:
        raise ValueError("No transformations provided")
    
    # Get the grid from first transform
    grid = transforms[0].grid()
    
    # Create new composed transform
    composed = StationaryVelocityFieldTransform(grid=grid)
    
    # Sum velocity fields
    total_velocity = torch.zeros_like(transforms[0].velocity())
    for t in transforms:
        total_velocity += t.velocity()
    
    # Set the composed velocity
    composed.velocity_(total_velocity)
    
    return composed