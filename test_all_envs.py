"""
Test all NeuroGym environments from the MPN paper figure.

This script verifies each environment works with our training setup
by running a short training session.
"""

import subprocess
import sys
from pathlib import Path

# Map plot labels to NeuroGym environment names
PAPER_ENVS = {
    "ContextDecisionMaking-v0": "Context DM",
    "DelayComparison-v0": "Delay Comparison",
    "DelayMatchSample-v0": "Delay Match Sample",
    "DelayMatchSampleDistractor1D-v0": "Delay Match Sample Dist",
    "DelayPairedAssociation-v0": "Delay Paired Association",
    "DualDelayMatchSample-v0": "Dual Delay Match Sample",
    "GoNogo-v0": "Go No-go",
    "IntervalDiscrimination-v0": "Interval Discrimination",
    "MotorTiming-v0": "Motor Timing",
    "MultiSensoryIntegration-v0": "Multi-Sensory Integration",
    "OneTwoThreeGo-v0": "One Two Three Go",
    "PerceptualDecisionMaking-v0": "Perceptual DM",
    "PerceptualDecisionMakingDelayResponse-v0": "Perc. DM Delay Response",
    "ProbabilisticReasoning-v0": "Probabilistic Reasoning",
    "ReadySetGo-v0": "Ready-Set-Go",
}


def test_env_basic(env_name: str) -> dict:
    """Test if environment can be created and stepped."""
    import neurogym as ngym
    import numpy as np

    try:
        env = ngym.make(env_name)
        obs, info = env.reset()

        # Get env info
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Run a few steps
        total_reward = 0
        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, info = env.reset()

        env.close()

        return {
            "status": "OK",
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "sample_reward": total_reward,
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
        }


def test_env_torchrl(env_name: str) -> dict:
    """Test if environment works with TorchRL wrapper."""
    import neurogym  # Register neurogym environments with gymnasium
    import torch
    from torchrl.envs import Compose, InitTracker, StepCounter, TransformedEnv
    from torchrl.envs.libs.gym import GymEnv

    try:
        env = TransformedEnv(
            GymEnv(env_name, device="cpu"),
            Compose(
                StepCounter(max_steps=100),
                InitTracker(),
            ),
        )

        # Test reset and step
        td = env.reset()
        obs_shape = td["observation"].shape

        # Run a few random steps
        for _ in range(10):
            action = env.action_spec.rand()
            td["action"] = action
            td = env.step(td)
            if td["next", "done"].item():
                td = env.reset()
            else:
                td = env.step_mdp(td)

        env.close()

        return {
            "status": "OK",
            "obs_shape": list(obs_shape),
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error": str(e),
        }


def test_env_seeding(env_name: str) -> dict:
    """Test that reseeding produces reproducible trials on both code paths.

    Checks:
      1. Direct neurogym path (oracle_agents.py style): env.unwrapped.rng
      2. TorchRL path (_reseed_torchrl_env walker): via GymWrapper
    """
    import neurogym as ngym
    import numpy as np

    # --- Path 1: direct neurogym ---
    try:
        env = ngym.make(env_name)

        def reseed_direct(e, seed):
            e.unwrapped.rng = np.random.RandomState(seed)

        reseed_direct(env, 42)
        env.reset()
        t1 = dict(env.unwrapped.trial)
        reseed_direct(env, 42)
        env.reset()
        t2 = dict(env.unwrapped.trial)
        reseed_direct(env, 99)
        env.reset()
        t3 = dict(env.unwrapped.trial)
        env.close()

        def trials_equal(a, b):
            if a.keys() != b.keys():
                return False
            for k in a:
                va, vb = a[k], b[k]
                if isinstance(va, np.ndarray):
                    if not np.array_equal(va, vb):
                        return False
                elif va != vb:
                    return False
            return True

        direct_reproducible = trials_equal(t1, t2)
    except Exception as e:
        return {"status": "FAILED", "path": "direct", "error": str(e)}

    # --- Path 2: TorchRL walker (_reseed_torchrl_env) ---
    try:
        import os
        import sys

        import neurogym  # noqa: F401 – registers envs with gymnasium
        from torchrl.envs.libs.gym import GymWrapper

        sys.path.insert(0, os.path.dirname(__file__))
        from main import _reseed_torchrl_env

        gym_env = ngym.make(env_name)
        tenv = GymWrapper(gym_env)

        _reseed_torchrl_env(tenv, 42)
        tenv.reset()
        r1 = dict(gym_env.unwrapped.trial)
        _reseed_torchrl_env(tenv, 42)
        tenv.reset()
        r2 = dict(gym_env.unwrapped.trial)
        tenv.close()

        torchrl_reproducible = trials_equal(r1, r2)
    except Exception as e:
        return {
            "status": "PARTIAL" if direct_reproducible else "FAILED",
            "direct_reproducible": direct_reproducible,
            "torchrl_error": str(e),
        }

    status = "OK" if (direct_reproducible and torchrl_reproducible) else "FAILED"
    return {
        "status": status,
        "direct_reproducible": direct_reproducible,
        "torchrl_reproducible": torchrl_reproducible,
        "trial_keys": list(t1.keys()),
    }


def test_env_training(
    env_name: str, model_type: str = "lstm", frames: int = 1000
) -> dict:
    """Run a quick training test on the environment."""
    import shutil
    import tempfile

    exp_name = f"test-{env_name.replace('-v0', '').lower()}-{model_type}"

    try:
        # Run training
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "train-neurogym",
                "--env-name",
                env_name,
                "--model-type",
                model_type,
                "--experiment-name",
                exp_name,
                "--total-frames",
                str(frames),
                "--hidden-dim",
                "32",
                "--print-freq",
                "500",
                "--checkpoint-freq",
                "10000",  # Don't checkpoint
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        success = result.returncode == 0

        # Clean up experiment dir
        exp_dir = Path("experiments") / exp_name
        if exp_dir.exists():
            shutil.rmtree(exp_dir)

        if success:
            return {
                "status": "OK",
                "output": (
                    result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                ),
            }
        else:
            return {
                "status": "FAILED",
                "error": (
                    result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                ),
            }

    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test NeuroGym environments")
    parser.add_argument(
        "--test-type",
        choices=["basic", "torchrl", "training", "seeding", "all"],
        default="basic",
        help="Type of test to run",
    )
    parser.add_argument(
        "--env", type=str, default=None, help="Test specific environment"
    )
    parser.add_argument(
        "--model-type", type=str, default="lstm", help="Model type for training test"
    )
    parser.add_argument(
        "--frames", type=int, default=1000, help="Frames for training test"
    )
    args = parser.parse_args()

    envs_to_test = (
        {args.env: PAPER_ENVS.get(args.env, args.env)} if args.env else PAPER_ENVS
    )

    print("=" * 70)
    print("Testing NeuroGym Environments")
    print("=" * 70)

    results = {}

    for env_name, paper_name in envs_to_test.items():
        print(f"\n{env_name} ({paper_name})")
        print("-" * 50)

        results[env_name] = {}

        if args.test_type in ["basic", "all"]:
            print("  Basic test...", end=" ", flush=True)
            res = test_env_basic(env_name)
            results[env_name]["basic"] = res
            if res["status"] == "OK":
                print(f"OK (obs={res['obs_dim']}, act={res['action_dim']})")
            else:
                print(f"FAILED: {res.get('error', 'Unknown')}")

        if args.test_type in ["torchrl", "all"]:
            print("  TorchRL test...", end=" ", flush=True)
            res = test_env_torchrl(env_name)
            results[env_name]["torchrl"] = res
            if res["status"] == "OK":
                print(f"OK (shape={res['obs_shape']})")
            else:
                print(f"FAILED: {res.get('error', 'Unknown')}")

        if args.test_type in ["seeding", "all"]:
            print("  Seeding test...", end=" ", flush=True)
            res = test_env_seeding(env_name)
            results[env_name]["seeding"] = res
            if res["status"] == "OK":
                keys = res.get("trial_keys", [])
                print(f"OK (trial keys: {keys})")
            elif res["status"] == "PARTIAL":
                print(
                    f"PARTIAL (direct=OK, torchrl error: {res.get('torchrl_error', '')[:80]})"
                )
            else:
                print(
                    f"FAILED: {res.get('error', res.get('torchrl_error', 'Unknown'))[:80]}"
                )

        if args.test_type in ["training", "all"]:
            print(
                f"  Training test ({args.model_type}, {args.frames} frames)...",
                end=" ",
                flush=True,
            )
            res = test_env_training(env_name, args.model_type, args.frames)
            results[env_name]["training"] = res
            print(res["status"])
            if res["status"] == "FAILED":
                print(f"    Error: {res.get('error', 'Unknown')[:200]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_type in ["basic", "torchrl", "seeding", "training"]:
        if any(test_type in r for r in results.values()):
            passed = sum(
                1
                for r in results.values()
                if r.get(test_type, {}).get("status") == "OK"
            )
            total = sum(1 for r in results.values() if test_type in r)
            print(f"  {test_type:12s}: {passed}/{total} passed")

    print()

    # Show any failures
    failures = [
        (env, test, res)
        for env, tests in results.items()
        for test, res in tests.items()
        if res.get("status") != "OK"
    ]

    if failures:
        print("FAILURES:")
        for env, test, res in failures:
            print(f"  {env} ({test}): {res.get('error', res.get('status'))[:100]}")


if __name__ == "__main__":
    main()
