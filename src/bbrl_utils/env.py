from time import strftime


def setup(maze_mdp=False, box2d=True):
    """Setup the notebook environment

    :param maze_mdp: install mazeMDP environment
    :param box2d: installs gymnasium[box2d]
    """
    from easypip import easyinstall

    if maze_mdp:
        easyinstall("mazemdp")

    if box2d:
        easyinstall("swig")
        easyinstall("gymnasium[box2d]")

    # Useful when using a timestamp for a directory name
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver(
        "current_time", lambda: strftime("%Y%m%d-%H%M%S"), replace=True
    )
