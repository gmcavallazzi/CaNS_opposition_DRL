#!/usr/bin/env python3
"""
Plot training trajectories to identify concerning trends.

Usage:
    python plot_training_trends.py <log_directory> [output_dir]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tensorboard.backend.event_processing import event_accumulator
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_training_trends.py <log_directory> [output_dir]")
        sys.exit(1)

    log_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "training_analysis"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading TensorBoard logs from: {log_dir}")
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Create comprehensive plots
    fig, axes = plt.subplots(4, 3, figsize=(20, 20))
    axes = axes.flatten()

    metrics = [
        ('Episode/reward', 'Reward', 'Episode', 'Reward'),
        ('Episode/dpdx_mean', 'Drag (dpdx)', 'Episode', 'dpdx'),
        ('Episode/actor_loss', 'Actor Loss', 'Episode', 'Loss'),
        ('Episode/critic_loss', 'Critic Loss', 'Episode', 'Loss'),
        ('Episode/consistency_loss', 'Consistency Loss ⚠️', 'Episode', 'Loss'),
        ('Episode/spatial_loss', 'Spatial Loss', 'Episode', 'Loss'),
        ('Episode/global_mean_loss', 'Global Mean Loss', 'Episode', 'Loss'),
        ('Episode/actor_grad_norm', 'Actor Grad Norm ⚠️', 'Episode', 'Norm'),
        ('Episode/critic_grad_norm', 'Critic Grad Norm', 'Episode', 'Norm'),
        ('Episode/action_mean', 'Action Mean (should be ~0)', 'Episode', 'Mean'),
        ('Episode/action_std', 'Action Std Dev', 'Episode', 'Std'),
        ('Episode/noise_scale', 'Exploration Noise Scale', 'Episode', 'Noise'),
    ]

    for idx, (tag, title, xlabel, ylabel) in enumerate(metrics):
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            axes[idx].plot(steps, values, linewidth=1.5, alpha=0.8)
            axes[idx].set_xlabel(xlabel, fontsize=10)
            axes[idx].set_ylabel(ylabel, fontsize=10)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

            # Add trend indicators
            if len(values) > 20:
                # Fit linear trend
                z = np.polyfit(steps, values, 1)
                p = np.poly1d(z)
                axes[idx].plot(steps, p(steps), "r--", alpha=0.5, linewidth=2,
                              label=f'Trend: {z[0]:.2e}/ep')
                axes[idx].legend(fontsize=8)

                # Identify concerning trends
                trend_slope = z[0]
                if 'consistency_loss' in tag and trend_slope > 0.001:
                    axes[idx].text(0.5, 0.95, '⚠️ INCREASING!',
                                  transform=axes[idx].transAxes,
                                  fontsize=12, color='red', weight='bold',
                                  ha='center', va='top',
                                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                elif 'grad_norm' in tag and abs(values[-1] - max(values)) < 0.001:
                    axes[idx].text(0.5, 0.95, '⚠️ SATURATED!',
                                  transform=axes[idx].transAxes,
                                  fontsize=12, color='red', weight='bold',
                                  ha='center', va='top',
                                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

            # Add reference lines for specific metrics
            if 'dpdx' in tag:
                axes[idx].axhline(y=-0.002, color='g', linestyle='--',
                                 alpha=0.7, label='Target')
                axes[idx].axhline(y=-0.0042, color='orange', linestyle='--',
                                 alpha=0.7, label='Uncontrolled')
                axes[idx].legend(fontsize=8)
            elif 'action_mean' in tag:
                axes[idx].axhline(y=0, color='black', linestyle='--',
                                 alpha=0.5, linewidth=2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_trajectories.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # Detailed trend analysis
    print("\n" + "="*80)
    print("TREND ANALYSIS")
    print("="*80)

    concerning_trends = []
    good_trends = []

    # Analyze consistency loss
    if 'Episode/consistency_loss' in ea.Tags()['scalars']:
        events = ea.Scalars('Episode/consistency_loss')
        if len(events) > 20:
            steps = [e.step for e in events]
            values = [e.value for e in events]

            # Activation point
            activation_idx = next((i for i, v in enumerate(values) if v > 0.01), None)

            if activation_idx is not None:
                # Analyze trend after activation
                post_activation_steps = steps[activation_idx:]
                post_activation_values = values[activation_idx:]

                z = np.polyfit(post_activation_steps, post_activation_values, 1)
                slope = z[0]

                print(f"\n📊 Consistency Loss Trend:")
                print(f"  Activated at episode: {steps[activation_idx]}")
                print(f"  Value at activation: {values[activation_idx]:.6f}")
                print(f"  Latest value: {values[-1]:.6f}")
                print(f"  Trend slope: {slope:.6e} per episode")

                if slope > 0.001:
                    print(f"  ⚠️  CONCERNING: Consistency loss INCREASING")
                    print(f"     This suggests:")
                    print(f"     1. Policy is learning features that are LESS consistent")
                    print(f"     2. Consistency weight may be too low to constrain learning")
                    print(f"     3. Other losses (actor, critic) are overwhelming consistency")
                    concerning_trends.append("Consistency loss increasing")
                elif slope < -0.001:
                    print(f"  ✅ GOOD: Consistency loss decreasing")
                    good_trends.append("Consistency loss improving")
                else:
                    print(f"  → STABLE: Consistency loss plateaued")

                # Check if it's plateauing vs growing
                last_20_mean = np.mean(post_activation_values[-20:])
                first_20_mean = np.mean(post_activation_values[:20])
                relative_change = (last_20_mean - first_20_mean) / first_20_mean * 100

                print(f"  First 20 ep avg: {first_20_mean:.6f}")
                print(f"  Last 20 ep avg: {last_20_mean:.6f}")
                print(f"  Relative change: {relative_change:+.1f}%")

    # Analyze spatial loss
    if 'Episode/spatial_loss' in ea.Tags()['scalars']:
        events = ea.Scalars('Episode/spatial_loss')
        if len(events) > 20:
            values = [e.value for e in events]
            first_10 = np.mean(values[:10])
            last_10 = np.mean(values[-10:])
            change = (last_10 - first_10) / first_10 * 100

            print(f"\n📊 Spatial Loss Trend:")
            print(f"  First 10 avg: {first_10:.6f}")
            print(f"  Last 10 avg: {last_10:.6f}")
            print(f"  Change: {change:+.1f}%")

            if change > 100:
                print(f"  ⚠️  CONCERNING: Spatial loss increasing significantly")
                print(f"     → Actions becoming spatially rougher")
                concerning_trends.append("Spatial loss increasing")

    # Analyze actor loss
    if 'Episode/actor_loss' in ea.Tags()['scalars']:
        events = ea.Scalars('Episode/actor_loss')
        if len(events) > 20:
            values = [e.value for e in events]
            steps = [e.step for e in events]

            # Actor loss should become more negative (increasing Q-values)
            z = np.polyfit(steps, values, 1)
            slope = z[0]

            print(f"\n📊 Actor Loss Trend:")
            print(f"  Latest value: {values[-1]:.2f}")
            print(f"  Trend slope: {slope:.4f} per episode")

            if slope > 0:
                print(f"  ⚠️  CONCERNING: Actor loss becoming less negative")
                print(f"     → Q-values decreasing (policy getting worse?)")
                concerning_trends.append("Actor loss degrading")
            else:
                print(f"  ✅ GOOD: Actor loss becoming more negative (Q-values increasing)")
                good_trends.append("Actor learning progressing")

    # Analyze gradient norms
    if 'Episode/actor_grad_norm' in ea.Tags()['scalars']:
        events = ea.Scalars('Episode/actor_grad_norm')
        if len(events) > 20:
            values = [e.value for e in events]
            steps = [e.step for e in events]

            # Check saturation
            max_val = max(values)
            last_20 = values[-20:]
            saturation_count = sum(1 for v in last_20 if abs(v - max_val) < 0.001)
            saturation_pct = saturation_count / len(last_20) * 100

            print(f"\n📊 Actor Gradient Norm:")
            print(f"  Max value: {max_val:.4f}")
            print(f"  Latest value: {values[-1]:.4f}")
            print(f"  Saturation (last 20): {saturation_pct:.1f}%")

            if saturation_pct > 80:
                print(f"  ❌ CRITICAL: Gradients saturated {saturation_pct:.0f}% of time")
                print(f"     → Increase gradient_clip from {max_val:.1f} to {max_val*2:.1f}")
                concerning_trends.append("Gradient saturation")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF TRENDS")
    print("="*80)

    if good_trends:
        print(f"\n✅ Positive Trends ({len(good_trends)}):")
        for t in good_trends:
            print(f"  - {t}")

    if concerning_trends:
        print(f"\n⚠️  Concerning Trends ({len(concerning_trends)}):")
        for t in concerning_trends:
            print(f"  - {t}")

        print(f"\n💡 Recommendations:")
        if "Consistency loss increasing" in concerning_trends:
            print(f"  1. Increase consistency loss weight:")
            print(f"     lambda: 0.1 → 0.2 or 0.3")
            print(f"  2. Check if actor/critic losses are too dominant")
            print(f"  3. Consider adding temporal smoothness penalty")

        if "Gradient saturation" in concerning_trends:
            print(f"  1. CRITICAL: Increase gradient_clip immediately")
            print(f"  2. Resume training from current checkpoint")

        if "Spatial loss increasing" in concerning_trends:
            print(f"  1. Consider increasing lambda_spatial")
            print(f"  2. Check if consistency loss is conflicting with spatial smoothness")
    else:
        print("\n✅ No concerning trends detected!")

    print(f"\n📁 Visualization saved to: {output_dir}/training_trajectories.png")


if __name__ == "__main__":
    main()
