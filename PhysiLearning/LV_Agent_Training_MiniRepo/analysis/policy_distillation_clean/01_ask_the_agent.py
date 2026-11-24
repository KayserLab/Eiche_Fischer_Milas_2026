"""Step 1: load the agent and ask it what it would do in a few situations.

Run from anywhere, in the `distill` environment:
    python analysis/policy_distillation_clean/01_ask_the_agent.py
"""
import numpy as np
import torch

from common import load_agent, p_treat

agent = load_agent()


def ask(label, situation):
    """Print one situation (8 numbers) and the agent's P(treat) for it."""
    burdens = situation[0::2]   # positions 0, 2, 4, 6
    running = situation[1::2]   # positions 1, 3, 5, 7
    p = p_treat(agent, situation)[0]
    print(label)
    print(f"  burdens, oldest -> newest:  {burdens}")
    print(f"  action that was running:    {running}")
    print(f"  P(treat) = {p:.3f}  ->  {'TREAT' if p > 0.5 else 'no treatment'}\n")


# 1. What is inside the black box? -------------------------------------------
print("The agent's decision network:")
print(agent.policy.mlp_extractor.policy_net)  # 8 numbers -> 64 -> 64
print(agent.policy.action_net, "\n")          # 64 -> 2 scores (OFF, ON)

# 2. A few questions -----------------------------------------------------------
ask("First decision of a run (no history yet, so zeros):",
    [0, 0, 0, 0, 0, 0, 1.0, 0])
ask("Burden grew to 1.6, no treatment running:",
    [1.3, 0, 1.4, 0, 1.5, 0, 1.6, 0])
ask("Same burdens, but treatment ran during the last interval:",
    [1.3, 0, 1.4, 0, 1.5, 0, 1.6, 1])
ask("Burden grew to 2.4, treatment ran during the last interval:",
    [2.1, 0, 2.2, 0, 2.3, 0, 2.4, 1])

# 3. P(treat) vs. the action -------------------------------------------------
# P(treat) is plain arithmetic: the same situation always gives the same number.
borderline = [1.3, 0, 1.4, 0, 1.45, 0, 1.5, 0]
print("A borderline situation, asked 3 times:", p_treat(agent, [borderline] * 3), "\n")

# agent.predict turns P(treat) into an action (0 or 1), in one of two ways:
#   deterministic=True  (used in the DPD runs): treat if P(treat) > 0.5
#   deterministic=False (used during training): flip a coin that says "treat"
#                        with probability P(treat)
torch.manual_seed(0)  # makes the coin flips repeatable
x = np.array(borderline, dtype=np.float32)
for deterministic in (True, False):
    actions = [int(agent.predict(x, deterministic=deterministic)[0]) for _ in range(1000)]
    print(f"agent.predict(deterministic={deterministic}): treated in {np.mean(actions):.0%} of 1000 tries")
