

from time import strftime


def setup():
    """Setup the notebook environment"""
    # from easypip import easyinstall

    # Useful when using a timestamp for a directory name
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver(
        "current_time", lambda: strftime("%Y%m%d-%H%M%S"), replace=True
    )