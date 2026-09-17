from dataclasses import dataclass, field

from gsplat import MCMCStrategy

from config.config import Config


@dataclass
class MCMCConfig(Config):
    init_opa: float = 0.5
    init_scale: float = 0.1
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    strategy: MCMCStrategy = field(
        default_factory=lambda: MCMCStrategy(
            verbose=False
        )
    )
