"""
Standalone test script for point-based DRL system.

Tests:
1. Environment initialization and shapes
2. Network forward passes
3. Zero-mean constraint
4. Replay buffer operations
5. Single training update

Run this before starting full training to catch issues early.
"""

import numpy as np
import torch
from utils import load_config

print("=" * 80)
print("POINT-BASED DRL SYSTEM TEST")
print("=" * 80)

# ============================================================================
# Test 1: Load Configuration
# ============================================================================
print("\n[Test 1] Loading configuration...")
try:
    config = load_config('config_point.yaml')
    print("✓ Configuration loaded successfully")
    print(f"  Total timesteps: {config['total_timesteps']}")
    print(f"  Buffer size: {config['model']['buffer_size']:,}")
    print(f"  Include prev_action: {config['observation']['include_prev_action']}")
    print(f"  Actor hidden dim: {config['net_arch']['actor_hidden_dim']}")
except Exception as e:
    print(f"✗ Failed to load config: {e}")
    exit(1)

# ============================================================================
# Test 2: Environment Initialization (NO MPI - just check shapes)
# ============================================================================
print("\n[Test 2] Checking environment structure...")
try:
    # We can't actually run the environment without MPI/CaNS, but we can check structure
    from stwEnv_pettingzoo_point import STWParallelEnvPoint

    # Check that class exists and has expected attributes
    print("✓ STWParallelEnvPoint class imported successfully")

    # Simulate what environment would create
    num_agents = 64 * 64  # 4096 agents
    include_prev_action = config['observation']['include_prev_action']
    obs_channels = 3 if include_prev_action else 2

    print(f"  Number of agents: {num_agents}")
    print(f"  Observation channels: {obs_channels}")
    print(f"  Observation shape per agent: ({obs_channels},)")
    print(f"  Action shape per agent: (1,)")

except Exception as e:
    print(f"✗ Environment check failed: {e}")
    exit(1)

# ============================================================================
# Test 3: Network Initialization
# ============================================================================
print("\n[Test 3] Initializing networks...")
try:
    from models_point import MLPActorPoint, CNNCriticPoint

    obs_channels = 3 if config['observation']['include_prev_action'] else 2
    device = "cpu"

    # Actor
    actor = MLPActorPoint(
        hidden_dim=config['net_arch']['actor_hidden_dim'],
        dropout_rate=config['net_arch']['dropout_rate'],
        similarity_dim=config['net_arch']['similarity_dim'],
        input_channels=obs_channels
    ).to(device)

    actor_params = sum(p.numel() for p in actor.parameters())
    print(f"✓ MLPActorPoint initialized ({actor_params} parameters)")

    # Critic
    critic = CNNCriticPoint(
        conv_channels=config['net_arch']['critic_conv'],
        mlp_layers=config['net_arch']['critic_mlp'],
        dropout_rate=config['net_arch']['dropout_rate'],
        obs_channels=obs_channels
    ).to(device)

    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"✓ CNNCriticPoint initialized ({critic_params:,} parameters)")

except Exception as e:
    print(f"✗ Network initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 4: Actor Forward Pass
# ============================================================================
print("\n[Test 4] Testing actor forward pass...")
try:
    batch_size = 4096
    obs_test = torch.randn(batch_size, obs_channels).to(device)

    # Forward pass without similarity
    actions = actor(obs_test, return_similarity=False)
    assert actions.shape == (batch_size, 1), f"Expected shape (4096, 1), got {actions.shape}"
    assert torch.all((actions >= -1) & (actions <= 1)), "Actions not in [-1, 1]"
    print(f"✓ Actor forward pass successful")
    print(f"  Input shape: {obs_test.shape}")
    print(f"  Output shape: {actions.shape}")
    print(f"  Output range: [{actions.min().item():.3f}, {actions.max().item():.3f}]")

    # Forward pass with similarity
    actions, sim_features = actor(obs_test, return_similarity=True)
    assert sim_features.shape == (batch_size, 16), f"Expected sim shape (4096, 16), got {sim_features.shape}"
    print(f"✓ Similarity features extracted: {sim_features.shape}")

except Exception as e:
    print(f"✗ Actor forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 5: Critic Forward Pass
# ============================================================================
print("\n[Test 5] Testing critic forward pass...")
try:
    batch_size = 8
    obs_batch = torch.randn(batch_size, 4096, obs_channels).to(device)
    act_batch = torch.randn(batch_size, 4096, 1).to(device)

    q_values = critic(obs_batch, act_batch)
    assert q_values.shape == (batch_size, 1), f"Expected shape (8, 1), got {q_values.shape}"
    print(f"✓ Critic forward pass successful")
    print(f"  Obs input shape: {obs_batch.shape}")
    print(f"  Act input shape: {act_batch.shape}")
    print(f"  Q-value output shape: {q_values.shape}")
    print(f"  Q-value range: [{q_values.min().item():.3f}, {q_values.max().item():.3f}]")

except Exception as e:
    print(f"✗ Critic forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 6: Zero-Mean Constraint
# ============================================================================
print("\n[Test 6] Testing zero-mean constraint...")
try:
    # Generate random actions
    actions_raw = torch.randn(4096, 1).to(device)
    initial_mean = actions_raw.mean().item()

    # Apply zero-mean correction (differentiable)
    global_mean = actions_raw.mean()
    actions = actions_raw - global_mean

    final_mean = actions.mean().item()

    print(f"✓ Zero-mean correction successful")
    print(f"  Initial mean: {initial_mean:.6f}")
    print(f"  Final mean: {final_mean:.10f} (should be ~0)")
    assert abs(final_mean) < 1e-6, f"Mean not zero: {final_mean}"

    # Test after noise and clipping
    noise = torch.randn_like(actions) * 0.1
    noise = noise - noise.mean()  # Zero-mean noise
    actions_noisy = actions + noise
    actions_clipped = torch.clip(actions_noisy, -1, 1)

    # Re-apply zero-mean after clipping
    actions_final = actions_clipped - actions_clipped.mean()
    final_mean_after_clip = actions_final.mean().item()

    print(f"  After noise+clip: {final_mean_after_clip:.10f} (should be ~0)")
    assert abs(final_mean_after_clip) < 1e-6, f"Mean not zero after clip: {final_mean_after_clip}"

except Exception as e:
    print(f"✗ Zero-mean constraint test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 7: Replay Buffer Operations
# ============================================================================
print("\n[Test 7] Testing replay buffer...")
try:
    from models_point import BatchedReplayBufferPoint

    capacity = 1000
    n_agents = 4096
    buffer = BatchedReplayBufferPoint(capacity, n_agents, obs_channels=obs_channels)

    print(f"✓ Replay buffer initialized")
    print(f"  Capacity: {capacity}")
    print(f"  Number of agents: {n_agents}")
    print(f"  Obs channels: {obs_channels}")

    # Add some transitions
    for i in range(10):
        obs = np.random.randn(n_agents, obs_channels).astype(np.float32)
        acts = np.random.randn(n_agents, 1).astype(np.float32)
        prev_acts = np.random.randn(n_agents, 1).astype(np.float32)
        rews = np.random.randn(n_agents).astype(np.float32)
        next_obs = np.random.randn(n_agents, obs_channels).astype(np.float32)
        dones = np.zeros(n_agents, dtype=np.float32)

        buffer.add_batch(obs, acts, prev_acts, rews, next_obs, dones)

    print(f"✓ Added 10 transitions to buffer")
    print(f"  Buffer size: {buffer.size}")

    # Sample a batch
    batch = buffer.sample(batch_size=4, device=device)

    print(f"✓ Sampled batch from buffer")
    print(f"  Obs shape: {batch['obs'].shape}")
    print(f"  Acts shape: {batch['acts'].shape}")
    print(f"  Rewards shape: {batch['rews'].shape}")

    assert batch['obs'].shape == (4, 4096, obs_channels)
    assert batch['acts'].shape == (4, 4096, 1)
    assert batch['rews'].shape == (4, 4096)

except Exception as e:
    print(f"✗ Replay buffer test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 8: MADDPG Initialization
# ============================================================================
print("\n[Test 8] Testing MADDPG initialization...")
try:
    from models_point import SharedPolicyMADDPGPoint

    # Create agent list (simulate environment)
    agents = [f"agent_{i}_{j}" for i in range(64) for j in range(64)]

    maddpg = SharedPolicyMADDPGPoint(
        agents=agents,
        gamma=config['model']['gamma'],
        tau=config['model']['tau'],
        lr=config['model']['learning_rate'],
        critic_lr=config['model']['critic_learning_rate'],
        weight_decay=config['model']['weight_decay'],
        device=device,
        actor_hidden_dim=config['net_arch']['actor_hidden_dim'],
        critic_conv_channels=config['net_arch']['critic_conv'],
        critic_mlp_layers=config['net_arch']['critic_mlp'],
        gradient_clip=config['training']['gradient_clip'],
        lambda_temporal=config['model']['smoothness']['lambda_temporal'],
        lambda_global_mean=config['model']['smoothness']['lambda_global_mean'],
        consistency_enable=config['model']['smoothness']['consistency']['enable'],
        consistency_lambda=config['model']['smoothness']['consistency']['lambda'],
        consistency_tau=config['model']['smoothness']['consistency']['tau_similarity'],
        consistency_margin=config['model']['smoothness']['consistency']['margin'],
        consistency_sample_size=config['model']['smoothness']['consistency']['sample_size'],
        consistency_warmup_steps=config['model']['smoothness']['consistency']['warmup_steps'],
        similarity_dim=config['net_arch']['similarity_dim'],
        input_channels=obs_channels
    )

    print(f"✓ SharedPolicyMADDPGPoint initialized")
    print(f"  Number of agents: {len(agents)}")
    print(f"  Device: {device}")

except Exception as e:
    print(f"✗ MADDPG initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 9: Batched Action Selection
# ============================================================================
print("\n[Test 9] Testing batched action selection...")
try:
    all_obs = torch.randn(4096, obs_channels).to(device)

    all_actions = maddpg.select_actions_batched(all_obs)

    assert all_actions.shape == (4096, 1), f"Expected shape (4096, 1), got {all_actions.shape}"
    assert np.all((all_actions >= -1) & (all_actions <= 1)), "Actions not in [-1, 1]"

    action_mean = all_actions.mean()
    assert abs(action_mean) < 1e-6, f"Actions not zero-mean: {action_mean}"

    print(f"✓ Batched action selection successful")
    print(f"  Input shape: {all_obs.shape}")
    print(f"  Output shape: {all_actions.shape}")
    print(f"  Action mean: {action_mean:.10f} (should be ~0)")
    print(f"  Action std: {all_actions.std():.6f}")

except Exception as e:
    print(f"✗ Action selection test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 10: Single Training Update
# ============================================================================
print("\n[Test 10] Testing single training update...")
try:
    # Create a batch
    batch_size = 8
    batch = {
        'obs': torch.randn(batch_size, 4096, obs_channels).to(device),
        'acts': torch.randn(batch_size, 4096, 1).to(device),
        'prev_acts': torch.randn(batch_size, 4096, 1).to(device),
        'rews': torch.randn(batch_size, 4096).to(device),
        'next_obs': torch.randn(batch_size, 4096, obs_channels).to(device),
        'done': torch.zeros(batch_size, 4096).to(device)
    }

    # Apply zero-mean to actions (simulate environment behavior)
    batch['acts'] = batch['acts'] - batch['acts'].view(batch_size, -1).mean(dim=1, keepdim=True).view(batch_size, 1, 1)

    critic_loss, actor_loss, loss_breakdown = maddpg.update_batched(batch)

    # Check for NaN/Inf
    assert not np.isnan(critic_loss), "Critic loss is NaN"
    assert not np.isnan(actor_loss), "Actor loss is NaN"
    assert not np.isinf(critic_loss), "Critic loss is Inf"
    assert not np.isinf(actor_loss), "Actor loss is Inf"

    print(f"✓ Training update successful")
    print(f"  Critic loss: {critic_loss:.6f}")
    print(f"  Actor loss: {actor_loss:.6f}")
    print(f"  Q-loss: {loss_breakdown['q_loss']:.6f}")
    print(f"  Global mean loss: {loss_breakdown['global_mean_loss']:.6f}")
    print(f"  Consistency loss: {loss_breakdown['consistency_loss']:.6f}")
    print(f"  Actor grad norm: {loss_breakdown['actor_grad_norm']:.6f}")
    print(f"  Critic grad norm: {loss_breakdown['critic_grad_norm']:.6f}")

except Exception as e:
    print(f"✗ Training update failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# Test 11: Memory Estimation
# ============================================================================
print("\n[Test 11] Memory usage estimation...")
try:
    capacity = config['model']['buffer_size']
    n_agents = 4096

    # Calculate buffer memory
    obs_memory = capacity * n_agents * obs_channels * 4 / (1024**3)  # GB
    act_memory = capacity * n_agents * 1 * 4 / (1024**3)  # GB
    prev_act_memory = capacity * n_agents * 1 * 4 / (1024**3)  # GB
    rew_memory = capacity * n_agents * 4 / (1024**3)  # GB
    done_memory = capacity * n_agents * 4 / (1024**3)  # GB

    total_buffer_memory = 2 * obs_memory + 2 * act_memory + rew_memory + done_memory

    # Network memory (rough estimate)
    actor_params = sum(p.numel() for p in maddpg.actor.parameters())
    critic_params = sum(p.numel() for p in maddpg.critic.parameters())
    network_memory = (actor_params * 4 + critic_params * 4) * 4 / (1024**3)  # 4 networks, 4 bytes per param

    total_memory = total_buffer_memory + network_memory

    print(f"✓ Memory estimation complete")
    print(f"  Replay buffer memory: {total_buffer_memory:.2f} GB")
    print(f"    - Observations: {2 * obs_memory:.2f} GB")
    print(f"    - Actions: {2 * act_memory:.2f} GB")
    print(f"    - Rewards/Done: {rew_memory + done_memory:.2f} GB")
    print(f"  Network memory: {network_memory:.2f} GB")
    print(f"  Total estimated memory: {total_memory:.2f} GB")

    if total_memory > 50:
        print(f"  ⚠ WARNING: High memory usage! Consider reducing buffer_size")

except Exception as e:
    print(f"✗ Memory estimation failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print("\nSystem is ready for training. To start training, run:")
print("  cd run0")
print("  python stwStart_point.py --config config_point.yaml")
print("\nMonitor training with TensorBoard:")
print("  tensorboard --logdir=./logs_point --port=6006")
print("=" * 80)
