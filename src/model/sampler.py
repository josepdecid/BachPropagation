import torch
import numpy as np

from constants import INPUT_FEATURES, SEQUENCE_LEN
from model.gan.GANGenerator import GANGenerator
from model.gan.GANModel import GANModel
from utils.typings import NDArray


def generate_sample(model: GANModel, length: int) -> NDArray:
    """
    Generate synthetic song from the Generator.
    :param model:
    :param length: Length of the song.
    :return: NDArray with song data
    """
    model.eval_mode()
    with torch.no_grad():
        input_seq = GANGenerator.noise((1, SEQUENCE_LEN))
        sample_output = model.generator(input_seq, pretraining=True).view(INPUT_FEATURES).cpu().numpy()

        full_sample = [sample_output.tolist()]
        for note_index in range(length - 1):
            input_seq.narrow(1, 1, SEQUENCE_LEN - 1)
            input_seq = torch.cat(tensors=(input_seq, torch.tensor([[sample_output]])), dim=1)

            sample_output = model.generator(input_seq, pretraining=True).view(INPUT_FEATURES).cpu().numpy()
            full_sample.append(sample_output.tolist())

    return np.array(full_sample)
