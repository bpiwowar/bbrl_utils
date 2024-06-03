from time import strftime


def setup(maze_mdp=False):
    """Setup the notebook environment

    :param maze_mdp: install mazeMDP environment
    """
    from easypip import easyinstall

    if maze_mdp:
        easyinstall("mazemdp")

    # Useful when using a timestamp for a directory name
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver(
        "current_time", lambda: strftime("%Y%m%d-%H%M%S"), replace=True
    )
