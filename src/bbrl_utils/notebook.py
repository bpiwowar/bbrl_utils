import os
import logging
from enum import Enum
from pathlib import Path
from functools import lru_cache


class IPython(Enum):
    NONE = 0
    OTHER = 1
    IPYTHON = 2
    JUPYTER = 3
    GOOGLE_COLAB = 4

    def is_notebook(self):
        """Returns True if running into a notebook"""
        return self not in [IPython.NONE, IPython.OTHER, IPython.IPYTHON]

    @staticmethod
    def get():
        return get_ipython()  # noqa: F821

    @staticmethod
    def mode():
        """Returns the IPython mode"""
        try:
            shell = IPython.get().__class__.__module__  # noqa: F821
            if shell is None:
                return IPython.NONE

            return {
                "IPython.terminal.interactiveshell": IPython.IPYTHON,
                "ipykernel.zmqshell": IPython.JUPYTER,
                "google.colab._shell": IPython.GOOGLE_COLAB,
            }.get(shell, IPython.OTHER)

        except NameError:
            return IPython.NONE


MODE = IPython.mode()

if MODE == IPython.NONE:
    from tqdm.auto import tqdm  # noqa: F401
else:
    from tqdm.autonotebook import tqdm  # noqa: F401


if MODE.is_notebook():
    # Use video display
    from moviepy.editor import ipython_display as video_display  # noqa: F401

    # Ensures that we load matplotlib properly in notebooks
    IPython.get().run_line_magic("matplotlib", "inline")
else:
    logging.debug("Not displaying video (hidden since not in a notebook)")

    def video_display(*args, **kwargs):  # noqa: F811
        pass

    def display(*args, **kwargs):
        print(*args, **kwargs)


def setup_tensorboard(path):
    """Utility function for launching tensorboard

    For Colab - otherwise, it is easier and better to launch tensorboard from
    the terminal

    :param path: _description_
    """
    path = Path(path)
    answer = ""
    if MODE.is_notebook():
        if MODE == IPython.GOOGLE_COLAB:
            answer = "y"
        while answer not in ["y", "n"]:
            answer = input(
                "Do you want to launch tensorboard in this notebook [y/n] "
            ).lower()

    if answer == "y":
        MODE.get().run_line_magic("load_ext", "tensorboard")
        MODE.get().run_line_magic(
            "tensorboard", f"""--logdir '{outputs_dir().absolute()}'"""
        )
    else:
        import os.path as osp
        import sys

        print(
            f"Launch tensorboard from the shell: \n{osp.dirname(sys.executable)}"
            f"/tensorboard --logdir '{outputs_dir().absolute()}'"
        )


@lru_cache()
def testing_mode():
    """Return whether the notebook is in testing mode"""
    return os.environ.get("TESTING_MODE", "").lower() in ["on", "1"]


@lru_cache()
def outputs_dir():
    """Returns the working directory"""
    if testing_mode():
        return Path("outputs-testing")
    return Path("outputs")
