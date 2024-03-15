import pytorch_lightning as pl
import src.constants as cst


def callback_save_model(path, metric, top_k=3):
    check_point_callback = pl.callbacks.ModelCheckpoint(
        monitor=metric,
        verbose=True,
        save_top_k=top_k,
        mode='max',
        dirpath=path,
        filename='{epoch}-{' + metric + ':.2f}'
    )
    return check_point_callback


# TODO avoid early stopping
