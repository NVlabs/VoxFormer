# Copyright (c) 2021. All rights reserved.
from models.MSNet2D import MSNet2D
from models.MSNet3D import MSNet3D
from models.submodule import model_loss

__models__ = {
    "MSNet2D": MSNet2D,
    "MSNet3D": MSNet3D
}
