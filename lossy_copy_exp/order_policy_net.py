"""
Order Policy Network (Agent) for LO-ARMs

A lightweight MLP that predicts which position to generate next based on
hidden states from the AOGPT model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrderPolicyNet(nn.Module):
    """
    Agent that learns to select optimal generation order.

    The policy network takes hidden states from the model and outputs
    a probability distribution over available (unfilled) positions.

    Architecture:
        - Input: Hidden states [B, L+1, d_model] (includes [None] token)
        - Process: MLP projection to logits
        - Output: Softmax distribution [B, L] over positions

    Args:
        d_model: Hidden state dimension from AOGPT
        policy_dim: Hidden dimension for policy network
        num_positions: Number of positions (typically 2 for [x, y])
    """

    def __init__(self, d_model=768, policy_dim=128, num_positions=2):
        super().__init__()
        self.d_model = d_model
        self.policy_dim = policy_dim
        self.num_positions = num_positions

        # Policy network: hidden states -> position logits
        # We use hidden states from the [None] token as context
        self.policy_net = nn.Sequential(
            nn.Linear(d_model, policy_dim),
            nn.ReLU(),
            nn.Linear(policy_dim, num_positions)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with small values for stable training."""
        for module in self.policy_net:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states, filled_mask):
        """
        Compute action probabilities over unfilled positions.

        Args:
            hidden_states: [B, L+1, d_model] from AOGPT (includes [None] token)
            filled_mask: [B, L] binary mask (1=filled/unavailable, 0=available)

        Returns:
            action_probs: [B, L] softmax distribution over positions
                          (filled positions have probability 0)
        """
        B, L_plus_1, d_model = hidden_states.shape
        L = L_plus_1 - 1  # Remove [None] token

        # Use [None] token hidden state as context
        # The [None] token (position 0) aggregates info about the sequence
        context = hidden_states[:, 0, :]  # [B, d_model]

        # Compute logits for each position
        logits = self.policy_net(context)  # [B, L]

        # Mask filled positions (set to -inf so softmax gives 0 probability)
        # Create large negative value for masking
        mask_value = -1e9
        masked_logits = logits.masked_fill(filled_mask.bool(), mask_value)

        # Softmax to get probabilities
        action_probs = F.softmax(masked_logits, dim=-1)  # [B, L]

        return action_probs

    def sample_action(self, hidden_states, filled_mask, return_logits=False):
        """
        Sample an action from the policy distribution.

        Args:
            hidden_states: [B, L+1, d_model]
            filled_mask: [B, L]
            return_logits: If True, also return raw logits for BC loss

        Returns:
            actions: [B] sampled position indices
            log_probs: [B] log probabilities of sampled actions
            logits (optional): [B, L] raw logits before masking (for BC loss)
        """
        B, L_plus_1, d_model = hidden_states.shape
        L = L_plus_1 - 1

        # Get context from [None] token
        context = hidden_states[:, 0, :]  # [B, d_model]

        # Get raw logits
        logits = self.policy_net(context)  # [B, L]

        # Mask and get probabilities
        mask_value = -1e9
        masked_logits = logits.masked_fill(filled_mask.bool(), mask_value)
        action_probs = F.softmax(masked_logits, dim=-1)  # [B, L]

        # Sample from categorical distribution
        dist = torch.distributions.Categorical(action_probs)
        actions = dist.sample()  # [B]
        log_probs = dist.log_prob(actions)  # [B]

        if return_logits:
            return actions, log_probs, logits
        return actions, log_probs

    def get_action_logprob(self, hidden_states, filled_mask, actions):
        """
        Get log probability of specific actions.

        Useful for computing policy gradients.

        Args:
            hidden_states: [B, L+1, d_model]
            filled_mask: [B, L]
            actions: [B] action indices

        Returns:
            log_probs: [B] log probabilities
        """
        action_probs = self.forward(hidden_states, filled_mask)  # [B, L]
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        return log_probs


def test_order_policy_net():
    """Test the OrderPolicyNet implementation."""
    print("Testing OrderPolicyNet...")

    # Create policy network
    d_model = 128
    policy_dim = 64
    num_positions = 2
    agent = OrderPolicyNet(d_model=d_model, policy_dim=policy_dim, num_positions=num_positions)
    print(f"✓ Agent initialized: {sum(p.numel() for p in agent.parameters())} parameters")

    # Test with dummy hidden states
    B = 4  # batch size
    L = 2  # sequence length
    hidden_states = torch.randn(B, L+1, d_model)

    # Test 1: No positions filled
    filled_mask = torch.zeros(B, L)
    action_probs = agent(hidden_states, filled_mask)
    print(f"✓ Forward pass (no mask): action_probs shape {action_probs.shape}")
    assert action_probs.shape == (B, L), f"Expected shape {(B, L)}, got {action_probs.shape}"

    # Check probabilities sum to 1
    prob_sums = action_probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones(B), atol=1e-5), "Probabilities don't sum to 1"
    print(f"  Probability sums: {prob_sums.tolist()}")

    # Check all probabilities are positive
    assert (action_probs >= 0).all(), "Some probabilities are negative"
    print(f"  Action probs sample: {action_probs[0].tolist()}")

    # Test 2: First position filled
    filled_mask = torch.zeros(B, L)
    filled_mask[:, 0] = 1  # Fill position 0
    action_probs = agent(hidden_states, filled_mask)
    print(f"✓ Forward pass (position 0 filled)")

    # Check that filled position has ~0 probability
    assert (action_probs[:, 0] < 1e-6).all(), "Filled position should have ~0 probability"
    print(f"  Position 0 probs (should be ~0): {action_probs[:, 0].tolist()}")
    print(f"  Position 1 probs (should be ~1): {action_probs[:, 1].tolist()}")

    # Test 3: Sample actions
    filled_mask = torch.zeros(B, L)
    actions, log_probs = agent.sample_action(hidden_states, filled_mask)
    print(f"✓ Sample actions: {actions.tolist()}")
    print(f"  Log probs: {log_probs.tolist()}")
    assert actions.shape == (B,), f"Expected actions shape {(B,)}, got {actions.shape}"
    assert log_probs.shape == (B,), f"Expected log_probs shape {(B,)}, got {log_probs.shape}"
    assert (actions >= 0).all() and (actions < L).all(), "Actions out of range"

    # Test 4: Get log prob of specific actions
    test_actions = torch.tensor([0, 1, 0, 1])
    log_probs_test = agent.get_action_logprob(hidden_states, filled_mask, test_actions)
    print(f"✓ Get log prob: {log_probs_test.tolist()}")
    assert log_probs_test.shape == (B,), f"Expected shape {(B,)}, got {log_probs_test.shape}"

    # Test 5: Gradient flow
    hidden_states_grad = torch.randn(B, L+1, d_model, requires_grad=True)
    filled_mask_grad = torch.zeros(B, L)
    actions_grad, log_probs_grad = agent.sample_action(hidden_states_grad, filled_mask_grad)
    loss = -log_probs_grad.mean()  # REINFORCE loss
    loss.backward()
    print(f"✓ Gradient flow: loss {loss.item():.4f}")
    assert hidden_states_grad.grad is not None, "No gradient on hidden states"

    print("\n✅ All agent tests passed!")


if __name__ == '__main__':
    test_order_policy_net()
