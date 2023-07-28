from lightning.pytorch.callbacks import ProgressBar
from tqdm import tqdm
import sys

class LitProgressBar(ProgressBar):

    # def init_train_tqdm(self) -> tqdm:
    #     """ Override this to customize the tqdm bar for training. """
    #     bar = tqdm(
    #         desc='Training',
    #         initial=self.train_batch_idx,
    #         position=(2 * self.process_position),
    #         disable=False,
    #         leave=True,
    #         #dynamic_ncols=False,  # This two lines are only for pycharm
    #         ncols=100,
    #         file=sys.stdout,
    #         smoothing=0,
    #     )
    #     return bar

    def init_validation_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for validation. """
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        bar = tqdm(
            #desc='Validating',
            #position=(2 * self.process_position + has_main_bar),
            disable=True,
            # leave=False,
            # dynamic_ncols=False,
            # ncols=100,
            # file=sys.stdout
        )
        return bar

    # def init_test_tqdm(self) -> tqdm:
    #     """ Override this to customize the tqdm bar for testing. """
    #     bar = tqdm(
    #         desc="Testing",
    #         position=(2 * self.process_position),
    #         disable=self.is_disabled,
    #         leave=True,
    #         dynamic_ncols=False,
    #         ncols=100,
    #         file=sys.stdout
    #     )
    #     return bar