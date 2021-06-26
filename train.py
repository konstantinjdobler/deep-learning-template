from helpers.klib import CustomWandbLogger, process_click_args
import click
import pytorch_lightning as pl
from src.data_loading import BasicDataModule
from src.model import BinaryClassifier

WANDB_PROJECT = "<adjust me>"
WANDB_ENTITY = "<adjust me>"


@click.command()
@click.pass_context
@click.option('--workers', '-w', help="Number of workers", default=4, type=int)
@click.option('--epochs', '-e', help="Number of epochs to train", default=10, type=int)
@click.option('--batch-size', '-b', help="Batch size per GPU", default=8, type=int)
@click.option('--image-resizing', '-i', help="Image training size", default=64, type=int)
@click.option('--offline', help="Disbale wandb online syncing", is_flag=True)
@click.option('--seed', help="Specify seed", type=int, default=None)
@click.option('--gpus', '-g', help="Specify in one string all the GPU indices like \"0,1,3,5\". Default is to use the CPU.")
@click.option('--datasets', '-d', help="Datasets to train on", required=True, multiple=True, type=click.Path(exists=True, writable=True, file_okay=False))
def main(ctx, **cmd_args):
    cmd_args = process_click_args(ctx, cmd_args)
    manual_seed_specified = cmd_args.seed is not None
    cmd_args.seed  = pl.seed_everything(workers=True, seed=cmd_args.seed)
    print(cmd_args)

    dm = BasicDataModule(
        data_dirs=cmd_args.datasets, workers=cmd_args.workers, batch_size=cmd_args.batch_size)
    model = BinaryClassifier()
    wandb_logger = CustomWandbLogger(
        project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type='train')

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=cmd_args.epochs,
                         progress_bar_refresh_rate=1,
                         gpus=cmd_args.gpus,
                         logger=wandb_logger,
                         benchmark=not manual_seed_specified,
                         deterministic=manual_seed_specified)

    trainer.fit(model, dm)
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    # Needed against multiprocessing error in Colab
    __spec__ = None
    main()