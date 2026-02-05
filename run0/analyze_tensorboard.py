#!/usr/bin/env python3
"""
Extract and analyze TensorBoard training logs.

Usage:
    python analyze_tensorboard.py <log_directory>
    python analyze_tensorboard.py smoothness_check1/logs/maddpg_consistency_agents64_20260204_150218
"""

import sys
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_tensorboard.py <log_directory>")
        print("Example: python analyze_tensorboard.py smoothness_check1/logs/maddpg_consistency_agents64_20260204_150218")
        sys.exit(1)

    log_dir = sys.argv[1]

    print(f"Loading TensorBoard logs from: {log_dir}")
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # Extract full trajectories
    metrics_to_plot = {
        'Episode/reward': 'Reward',
        'Episode/dpdx_mean': 'Drag (dpdx)',
        'Episode/actor_loss': 'Actor Loss',
        'Episode/critic_loss': 'Critic Loss',
        'Episode/consistency_loss': 'Consistency Loss',
        'Episode/spatial_loss': 'Spatial Loss',
        'Episode/global_mean_loss': 'Global Mean Loss',
        'Episode/actor_grad_norm': 'Actor Grad Norm',
        'Episode/action_std': 'Action Std Dev',
        'Episode/noise_scale': 'Noise Scale',
    }

    print("="*80)
    print(f"TRAINING ANALYSIS - Episode {len(ea.Scalars('Episode/reward'))} (Latest)")
    print("="*80)

    for metric, name in metrics_to_plot.items():
        if metric in ea.Tags()['scalars']:
            events = ea.Scalars(metric)
            if len(events) >= 2:
                # Get last 20 episodes
                recent = events[-20:] if len(events) >= 20 else events
                values = [e.value for e in recent]

                print(f"\n{name}:")
                print(f"  Total episodes: {len(events)}")
                print(f"  Latest value: {events[-1].value:.6f}")
                print(f"  Last {len(recent)} mean: {np.mean(values):.6f}")
                print(f"  Last {len(recent)} std: {np.std(values):.6f}")

                # Check for trends
                if len(events) >= 10:
                    first_10_mean = np.mean([e.value for e in events[:10]])
                    last_10_mean = np.mean([e.value for e in events[-10:]])
                    change = ((last_10_mean - first_10_mean) / abs(first_10_mean) * 100) if first_10_mean != 0 else 0
                    trend = "↑" if change > 5 else "↓" if change < -5 else "→"
                    print(f"  Trend (first 10 → last 10): {trend} {change:+.1f}%")

    # Check specific concerns
    print("\n" + "="*80)
    print("DIAGNOSTIC FLAGS")
    print("="*80)

    # 1. Gradient clipping
    if 'Episode/actor_grad_norm' in ea.Tags()['scalars']:
        actor_grad = ea.Scalars('Episode/actor_grad_norm')
        if actor_grad:
            last_10_grads = [e.value for e in actor_grad[-10:]]
            # Try to infer clip value from data
            max_grad = max([e.value for e in actor_grad])
            clip_value = max_grad if max_grad > 0 else 0.5

            clipping_freq = sum(1 for g in last_10_grads if abs(g - clip_value) < 0.001) / len(last_10_grads)
            print(f"\n⚠️  Gradient Clipping:")
            print(f"  Inferred clip value: {clip_value:.3f}")
            print(f"  Clipping frequency (last 10 ep): {clipping_freq*100:.1f}%")
            if clipping_freq > 0.5:
                print(f"  ❌ ISSUE: Gradients being clipped frequently (>{50}%)")
                print(f"     → May prevent proper learning")
                print(f"     → Recommendation: Increase gradient_clip to {clip_value*2:.1f}")
            else:
                print(f"  ✅ OK: Clipping is occasional")

    # 2. Consistency loss activation
    if 'Episode/consistency_loss' in ea.Tags()['scalars']:
        consistency = ea.Scalars('Episode/consistency_loss')
        if consistency:
            non_zero = [e for e in consistency if e.value > 0.01]
            activation_ep = non_zero[0].step if non_zero else None
            print(f"\n📊 Consistency Loss:")
            print(f"  Activated at episode: {activation_ep if activation_ep else 'Not yet'}")
            print(f"  Latest value: {consistency[-1].value:.6f}")
            if activation_ep and activation_ep > 10:
                print(f"  ℹ️  Consistency activated after {activation_ep} episodes")

    # 3. Zero-mean constraint
    if 'Episode/global_mean_loss' in ea.Tags()['scalars'] and 'Episode/action_mean' in ea.Tags()['scalars']:
        global_mean_loss = ea.Scalars('Episode/global_mean_loss')
        action_mean = ea.Scalars('Episode/action_mean')
        if global_mean_loss and action_mean:
            latest_gml = global_mean_loss[-1].value
            latest_am = abs(action_mean[-1].value)
            print(f"\n✅ Zero-Mean Constraint:")
            print(f"  Global mean loss: {latest_gml:.6f}")
            print(f"  Action mean: {latest_am:.6f}")
            status = 'PERFECT' if latest_am < 1e-5 else 'OK' if latest_am < 0.05 else 'NEEDS WORK'
            print(f"  Status: {status}")

    # 4. Drag reduction progress
    if 'Episode/dpdx_mean' in ea.Tags()['scalars']:
        dpdx = ea.Scalars('Episode/dpdx_mean')
        if dpdx:
            target = -0.002
            uncontrolled = -0.0042
            latest_dpdx = dpdx[-1].value
            improvement = (uncontrolled - latest_dpdx) / (uncontrolled - target) * 100

            print(f"\n🎯 Drag Reduction:")
            print(f"  Current dpdx: {latest_dpdx:.6f}")
            print(f"  Target: {target:.6f}")
            print(f"  Uncontrolled: {uncontrolled:.6f}")
            print(f"  Progress to target: {improvement:.1f}%")

            # Show trajectory
            if len(dpdx) >= 10:
                first_10 = np.mean([e.value for e in dpdx[:10]])
                last_10 = np.mean([e.value for e in dpdx[-10:]])
                print(f"  First 10 episodes avg: {first_10:.6f}")
                print(f"  Last 10 episodes avg: {last_10:.6f}")
                improvement_pct = (first_10 - last_10)/(uncontrolled - target)*100
                print(f"  Improvement: {improvement_pct:.1f}% of possible gain")

    # 5. Training steps
    if 'Training/training_step' in ea.Tags()['scalars']:
        training_step = ea.Scalars('Training/training_step')
        if training_step:
            total_episodes = len(ea.Scalars('Episode/reward'))
            print(f"\n📈 Training Progress:")
            print(f"  Total training updates: {training_step[-1].value:.0f}")
            print(f"  (~{training_step[-1].value / total_episodes:.0f} updates per episode)")

    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    # Overall assessment
    issues = []
    successes = []

    # Check gradient clipping
    if 'Episode/actor_grad_norm' in ea.Tags()['scalars']:
        actor_grad = ea.Scalars('Episode/actor_grad_norm')
        last_10_grads = [e.value for e in actor_grad[-10:]]
        max_grad = max([e.value for e in actor_grad])
        clipping_freq = sum(1 for g in last_10_grads if abs(g - max_grad) < 0.001) / len(last_10_grads)

        if clipping_freq > 0.5:
            issues.append("High gradient clipping frequency")
        else:
            successes.append("Stable gradients")

    # Check zero-mean
    if 'Episode/action_mean' in ea.Tags()['scalars']:
        action_mean = ea.Scalars('Episode/action_mean')
        latest_am = abs(action_mean[-1].value)
        if latest_am < 1e-5:
            successes.append("Perfect zero-mean constraint")

    # Check drag reduction
    if 'Episode/dpdx_mean' in ea.Tags()['scalars']:
        dpdx = ea.Scalars('Episode/dpdx_mean')
        target = -0.002
        uncontrolled = -0.0042
        latest_dpdx = dpdx[-1].value
        improvement = (uncontrolled - latest_dpdx) / (uncontrolled - target) * 100

        if improvement > 50:
            successes.append("Good drag reduction progress")
        elif improvement < 20:
            issues.append("Limited drag reduction")

    # Check consistency loss
    if 'Episode/consistency_loss' in ea.Tags()['scalars']:
        consistency = ea.Scalars('Episode/consistency_loss')
        if consistency[-1].value < 0.1:
            issues.append("Consistency loss very low (may not be effective)")
        elif consistency[-1].value > 2.0:
            issues.append("Consistency loss very high (may be overwhelming)")

    print(f"\n✅ Successes ({len(successes)}):")
    for s in successes:
        print(f"  - {s}")

    if issues:
        print(f"\n❌ Issues ({len(issues)}):")
        for i in issues:
            print(f"  - {i}")
    else:
        print(f"\n❌ Issues: None detected")

    # Final recommendation
    print(f"\n💡 Recommendation:")
    if 'Episode/dpdx_mean' in ea.Tags()['scalars']:
        dpdx = ea.Scalars('Episode/dpdx_mean')
        improvement = (uncontrolled - dpdx[-1].value) / (uncontrolled - target) * 100

        if len(issues) > len(successes):
            print("  Training shows concerning patterns. Consider:")
            if any("gradient clipping" in i.lower() for i in issues):
                print("  1. Increase gradient_clip (e.g., from 0.5 to 1.0 or 2.0)")
            print("  2. Check if action memory channel is being used")
            print("  3. Increase consistency loss weight if too low")
        elif improvement < 50:
            print("  Training is progressing but not converged yet.")
            print("  Recommend continuing for more episodes (target: 200-300)")
        else:
            print("  Training looks healthy overall!")
            if any("gradient clipping" in i.lower() for i in issues):
                print("  Main issue is gradient clipping - increase the clip threshold")
                print("  to allow policy refinement and smoother control.")
            else:
                print("  The high temporal gradient in test may be a transient phenomenon.")


if __name__ == "__main__":
    main()
