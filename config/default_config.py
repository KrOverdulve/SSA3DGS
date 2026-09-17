from dataclasses import dataclass, field

from gsplat import DefaultStrategy

from config.config import Config


@dataclass
class DefaultConfig(Config):
    strategy: DefaultStrategy = field(
        default_factory=lambda: DefaultStrategy(
            verbose=True
        )
    )