from mpn_rl.commands.train import LSTMConfig, MPNConfig, MPNFrozenConfig, RNNConfig
from mpn_rl.models.actor_critic import ActorCriticNet


def test_actor_critic_net_builds_from_each_model_config() -> None:
    for config in (LSTMConfig(), RNNConfig(), MPNConfig(), MPNFrozenConfig()):
        net = ActorCriticNet(
            input_dim=4, action_dim=2, hidden_dim=8, **config.model_dump()
        )
        assert net.core.model_type == config.model_type
