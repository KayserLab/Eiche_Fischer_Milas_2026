#!/usr/bin/env python3
"""Smoke test for scripts/distill_policy.py.

Checks, for each teacher checkpoint:
  1. the stacked observation order produced by the training-equivalent env,
  2. that P(ON) extracted from the categorical actor reproduces
     model.predict(..., deterministic=True) on rollout and random observations,
  3. the tie-to-OFF hard-decision rule, rule/controller consistency and
     trajectory-grouped splitting without leakage.

Run from the mini-repo root: python scripts/test_distill_policy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distill_policy as dp  # noqa: E402

TEACHERS = {
    "paper_candidate": dp.DEFAULT_MODEL,
    "experimental_agent": dp.MINIREPO.parent
    / "Agent_used_for_experiment/2025_10_01_lv_dont_treat_1_load_best_reward",
}


def check_observation_order() -> None:
    env = dp.make_env(dp.DEFAULT_CONFIG)
    try:
        obs = env.reset()[0]
        # Zero-padded history; newest frame = [initial normalised burden, no prior action].
        assert np.allclose(obs[:6], 0.0) and np.isclose(obs[6], 1.0) and obs[7] == 0, obs
        actions = [1, 0, 1, 1, 0, 0]
        for k, a in enumerate(actions):
            new = env.step(np.array([a]))[0][0]
            assert np.allclose(new[:6], obs[2:]), (k, obs, new)  # frames shift oldest-out
            assert new[7] == a, (k, new)  # newest previous_action = action just executed
            obs = new
        assert list(obs[1::2].astype(int)) == actions[-4:], obs
        burdens = obs[0::2]
        state = env.venv.envs[0].unwrapped.state
        assert np.isclose(burdens[-1], state[0] + state[1], rtol=1e-6), (burdens, state)
    finally:
        env.close()
    print("observation order OK: [B_t-3, a_t-3, B_t-2, a_t-2, B_t-1, a_t-1, B_t, a_t]")


def check_teacher(name: str, path: Path) -> None:
    model = dp.load_teacher(path)
    df = dp.generate_dataset(model, dp.DEFAULT_CONFIG, episodes=4, seed=0, max_steps=40,
                             dataset="primary")
    rng = np.random.default_rng(0)
    random_obs = np.zeros((50, 8), dtype=np.float32)
    random_obs[:, 0::2] = rng.uniform(0, 3, size=(50, 4))
    random_obs[:, 1::2] = rng.integers(0, 2, size=(50, 4))
    obs = np.vstack([df[dp.FEATURES].to_numpy(dtype=np.float32), random_obs])
    p_on = dp.teacher_probabilities(model, obs)
    predicted, _ = model.predict(obs, deterministic=True)
    assert np.array_equal(dp.hard_action(p_on), predicted), name
    assert np.allclose(dp.teacher_probabilities(model, obs[:5]), p_on[:5])
    # Recorded labels equal a fresh query of the same observation.
    assert np.allclose(df["p_on"].to_numpy(), p_on[: len(df)], atol=1e-6)
    print(f"{name}: P(ON) argmax == predict(deterministic=True) on {len(obs)} observations "
          f"(policy.pth sha256 {dp.policy_weights_sha256(path)[:12]})")


def check_helpers() -> None:
    assert list(dp.hard_action(np.array([0.5, 0.5000001, 0.4999]))) == [0, 1, 0]
    model = dp.load_teacher(dp.DEFAULT_MODEL)
    df = dp.generate_dataset(model, dp.DEFAULT_CONFIG, episodes=16, seed=1, max_steps=30,
                             dataset="primary")
    greedy = df[df["epsilon"] == 0]
    assert greedy["trajectory_id"].nunique() == 1, "deterministic greedy episodes should coincide"
    parts = dp.split_groups(df, seed=0)
    assert sum(len(p) for p in parts.values()) == len(df)
    rule = dp.hysteresis_rule(df, start=1.5, stay=2.2)
    ctrl = dp.hysteresis_controller(start=1.5, stay=2.2)
    assert np.array_equal(rule, [ctrl(x) for x in df[dp.FEATURES].to_numpy(dtype=np.float32)])
    print("helpers OK: tie->OFF, duplicate greedy trajectories grouped, no split leakage")


def main() -> None:
    check_observation_order()
    for name, path in TEACHERS.items():
        check_teacher(name, path)
    check_helpers()
    print("All distillation smoke checks passed.")


if __name__ == "__main__":
    main()
