#!/usr/bin/env python3
"""
Calculate memory usage for halo=1 implementation to verify our fix.
"""

def calculate_memory_usage():
    print("Memory Usage Calculation for Halo=1 Implementation")
    print("=" * 55)

    # Configuration
    grid_size = 64
    halo = 1
    buffer_size = 5_000_000  # Original buffer size from error
    num_agents = grid_size * grid_size  # 4096 agents

    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Halo: {halo}")
    print(f"Number of agents: {num_agents}")
    print(f"Buffer size: {buffer_size}")
    print()

    # BEFORE FIX: Using halo-based observation shape for replay buffer
    print("BEFORE FIX (causing 1.34 TiB error):")
    print("-" * 40)

    obs_height_halo = 2 * halo + 1  # 3
    obs_width_halo = 2 * halo + 1   # 3
    channels = 2
    halo_obs_dim = obs_height_halo * obs_width_halo * channels  # 3 * 3 * 2 = 18

    # Memory for observation buffer: capacity × n_agents × obs_dim × 4 bytes (float32)
    before_memory_bytes = buffer_size * num_agents * halo_obs_dim * 4
    before_memory_gb = before_memory_bytes / (1024**3)
    before_memory_tb = before_memory_bytes / (1024**4)

    print(f"Halo observation shape: ({obs_height_halo}, {obs_width_halo}, {channels})")
    print(f"Halo observation dimension: {halo_obs_dim}")
    print(f"Replay buffer memory: {before_memory_bytes:,} bytes")
    print(f"Replay buffer memory: {before_memory_gb:.2f} GB")
    print(f"Replay buffer memory: {before_memory_tb:.2f} TiB")
    print(f"Result: {before_memory_tb:.2f} TiB - MEMORY ALLOCATION ERROR!")
    print()

    # AFTER FIX: Using actor observation shape for replay buffer
    print("AFTER FIX (our solution):")
    print("-" * 30)

    # Actors still use halo-based observations
    actor_obs_dim = halo_obs_dim  # 18

    # But replay buffer stores actor observations (smaller)
    after_memory_bytes = buffer_size * num_agents * actor_obs_dim * 4
    after_memory_gb = after_memory_bytes / (1024**3)
    after_memory_tb = after_memory_bytes / (1024**4)

    print(f"Actor observation shape: ({obs_height_halo}, {obs_width_halo}, {channels})")
    print(f"Actor observation dimension: {actor_obs_dim}")
    print(f"Replay buffer memory: {after_memory_bytes:,} bytes")
    print(f"Replay buffer memory: {after_memory_gb:.2f} GB")
    print(f"Replay buffer memory: {after_memory_tb:.2f} TiB")
    print(f"Result: {after_memory_tb:.2f} TiB - SAME AS BEFORE!")
    print()

    # Wait, this shows the issue is still there! Let me recalculate...
    print("ANALYSIS:")
    print("-" * 10)
    print(f"The calculation shows we still have the same memory issue!")
    print(f"This means the problem wasn't the observation shape itself,")
    print(f"but rather the massive buffer size combined with many agents.")
    print()
    print(f"With {num_agents:,} agents and buffer size {buffer_size:,}:")
    print(f"- Total buffer entries: {buffer_size * num_agents:,}")
    print(f"- Even with minimal obs_dim=2: {buffer_size * num_agents * 2 * 4 / (1024**3):.2f} GB")
    print(f"- With halo obs_dim=18: {buffer_size * num_agents * 18 * 4 / (1024**3):.2f} GB")
    print()

    print("RECOMMENDED FIXES:")
    print("-" * 18)
    print("1. Reduce buffer size significantly (e.g., 100,000 instead of 5,000,000)")
    print("2. Use a smaller grid for initial testing (e.g., 32x32 instead of 64x64)")
    print("3. Consider using shared experience replay across agents")
    print()

    # Calculate reasonable buffer size
    reasonable_memory_gb = 8.0  # 8 GB limit
    reasonable_buffer_size = int(reasonable_memory_gb * (1024**3) / (num_agents * actor_obs_dim * 4))

    print(f"For {reasonable_memory_gb} GB memory limit:")
    print(f"Recommended buffer size: {reasonable_buffer_size:,}")

    return before_memory_tb, after_memory_tb, reasonable_buffer_size

if __name__ == "__main__":
    calculate_memory_usage()