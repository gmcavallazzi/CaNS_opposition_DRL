"""
Unified wrapper for testing policy smoothness - auto-detects architecture type.

This script automatically detects whether the checkpoint is from:
- Patch-based system (64 agents, 8×8 patches) → uses test_policy_smoothness.py
- Point-based system (4096 agents, scalar actions) → uses test_policy_smoothness_point.py

Detection is based on:
1. Config file name (contains "point")
2. Network architecture (actor MLP vs CNN)
3. Checkpoint metadata

Usage:
    cd run0
    python test_policy_smoothness_auto.py \
      --checkpoint ../checkpoints/model.pt \
      --config config_consistency_with_memory_fix1.yaml \
      --num_episodes 5

    python test_policy_smoothness_auto.py \
      --checkpoint ../checkpoints_point/model.pt \
      --config config_point.yaml \
      --num_episodes 5
"""

import argparse
import sys
import os

# Import utility to load config
from utils import load_config


def detect_architecture(config_path, checkpoint_path=None):
    """
    Detect whether this is patch-based or point-based architecture.

    Returns:
        str: 'patch' or 'point'
    """
    # Method 1: Check config filename
    if 'point' in os.path.basename(config_path).lower():
        print("Detected POINT-BASED architecture (from config filename)")
        return 'point'

    # Method 2: Load config and check architecture parameters
    config = load_config(config_path)

    # Point-based has actor_hidden_dim (MLP), patch-based has actor_channels (CNN)
    if 'actor_hidden_dim' in config.get('net_arch', {}):
        print("Detected POINT-BASED architecture (from net_arch.actor_hidden_dim)")
        return 'point'
    elif 'actor_channels' in config.get('net_arch', {}):
        print("Detected PATCH-BASED architecture (from net_arch.actor_channels)")
        return 'patch'

    # Default to patch-based (original system)
    print("Could not detect architecture, defaulting to PATCH-BASED")
    return 'patch'


def main():
    parser = argparse.ArgumentParser(
        description='Unified policy smoothness test - auto-detects architecture'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Number of episodes to run (default: 5)')
    parser.add_argument('--point_i', type=int, default=32,
                       help='i-coordinate of point to track (default: 32)')
    parser.add_argument('--point_j', type=int, default=32,
                       help='j-coordinate of point to track (default: 32)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results')
    parser.add_argument('--force_arch', type=str, choices=['patch', 'point'],
                       help='Force specific architecture (skip auto-detection)')

    args = parser.parse_args()

    # Detect architecture
    if args.force_arch:
        arch_type = args.force_arch
        print(f"Forced architecture: {arch_type.upper()}")
    else:
        arch_type = detect_architecture(args.config, args.checkpoint)

    print("\n" + "="*80)
    print(f"ARCHITECTURE: {arch_type.upper()}-BASED")
    print("="*80 + "\n")

    # Import and run appropriate script
    if arch_type == 'point':
        print("Using point-based smoothness test...")
        from test_policy_smoothness_point import analyze_policy_smoothness
    else:  # patch
        print("Using patch-based smoothness test...")
        from test_policy_smoothness import analyze_policy_smoothness

    # Run analysis
    analyze_policy_smoothness(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.num_episodes,
        save_results=not args.no_save,
        point_idx=(args.point_i, args.point_j)
    )


if __name__ == "__main__":
    main()
