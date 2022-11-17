import argparse

import os
import numpy as np
import json
import logging

from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torch.profiler

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from pytorch_lightning import Trainer, LightningDataModule, LightningModule, Callback

from pytorch_lightning.loggers import TensorBoardLogger

from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

NUM_CLASSES = 10

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

class TorchTensorboardProfilerCallback(Callback):
    """Quick-and-dirty Callback for invoking TensorboardProfiler during training.
    
    For greater robustness, extend the pl.profiler.profilers.BaseProfiler. See
    https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html"""

    def __init__(self, profiler):
        super().__init__()
        self.profiler = profiler 

    def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
        self.profiler.step()
        # pl_module.log_dict(outputs)  # also logging the loss, while we're here
        
        
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR10DataModule(LightningDataModule):

    def __init__(
        self,
        val_test_split: Tuple[int, int] = (5_000, 5_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        
        # data transformations
        self.train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        CIFAR10(root=os.getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"), train=True, download=True)
        CIFAR10(root=os.getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"), train=False, download=True)


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = CIFAR10(root=os.getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"), train=True, transform=self.train_transform)
            testset = CIFAR10(root=os.getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"), train=False, transform=self.test_transform)
            self.data_val, self.data_test = random_split(
                dataset=testset,
                lengths=self.hparams.val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
    
class CIFAR10LitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module
        ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()
        
        self.example_input_array = torch.rand((1,3,32,32))
        
    def forward(self, x: torch.Tensor):
        return self.net(x)
    
    def add_param_histogram(self):
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)
            
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        # self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # add all parameters to a histogram
        self.add_param_histogram()
        self.logger.experiment.add_scalar("Accuracy/Train", self.train_acc.compute(), self.current_epoch)
        
        # `outputs` is a list of dicts returned from `training_step()`
        self.train_acc.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        # self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.logger.experiment.add_scalar("Accuracy/Val", acc, self.current_epoch)
        self.val_acc_best.update(acc)
        # self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        # self.log("test/loss", loss, on_step=False, on_epoch=True)
        # self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self.logger.experiment.add_scalar("Accuracy/Test", self.test_acc.compute(), self.current_epoch)
        self.test_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        return {
            "optimizer": torch.optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9),
        }


def train(args):

    net = Net()
    logger = TensorBoardLogger(save_dir=os.environ["SM_OUTPUT_DATA_DIR"], name="tensorboard", log_graph=True)

    
    datamodule = CIFAR10DataModule(batch_size=args.batch_size,
                                num_workers=args.num_workers)
    model = CIFAR10LitModule(net=net)
            
    # to print/log environment variables
    LOGGER.info(os.environ)
    
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(os.environ["SM_OUTPUT_DATA_DIR"], "tb_profiler")),
        record_shapes=True,
        profile_memory=True,
        with_stack=False)
    
    

    with profiler:
        
        profiler_callback = TorchTensorboardProfilerCallback(profiler)
        
        trainer = Trainer(
            accelerator=DEVICE,
            max_epochs=args.max_epochs,
            devices=args.num_gpus, 
            logger=logger,
            num_sanity_val_steps=0,
            callbacks=[profiler_callback],
            )
        
        LOGGER.info("Training model...")
        trainer.fit(model, datamodule)

    
    LOGGER.info("Saving model...")
    torch.save(net.state_dict(), os.path.join(os.getenv("SM_MODEL_DIR"), "model.pth"))
    


def model_fn(model_dir):
    LOGGER.info("Insider model loader")

    model = Net()
    if torch.cuda.device_count() > 1:
        print("Gpu count: {}".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    try:
        with open(os.path.join(model_dir, "model.pth"), "rb") as f:
            LOGGER.info(
                f"Trying to open model in {os.path.join(model_dir, 'model.pth')}"
            )
            model.load_state_dict(torch.load(f))
            return model.to(DEVICE)
    except Exception as e:
        LOGGER.exception(e)
        return None


def _load_from_bytearray(request_body):
    npimg = np.frombuffer(request_body, np.float32).reshape((1, 3, 32, 32))
    return torch.Tensor(npimg)


def transform_fn(model, request_body, content_type, accept_type):

    LOGGER.info("Running inference inside container")

    try:
        np_image = _load_from_bytearray(request_body)
        LOGGER.info("Deserialization completed")
    except Exception as e:
        LOGGER.exception(e)

    LOGGER.info("trying to run inference")
    try:
        outputs = model(np_image)
        _, predicted = torch.max(outputs, 1)
        LOGGER.info(f"Predictions: {predicted}")
    except Exception as e:
        LOGGER.exception(e)

    return json.dumps(predicted.numpy().tolist())

def parse_args():
    # SageMaker passes hyperparameters  as command-line arguments to the script
    # Parsing them below...
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hosts", type=list, default=os.environ["SM_HOSTS"])
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--num-gpus", type=int, default=int(os.environ["SM_NUM_GPUS"]))
    parser.add_argument("--num_nodes", type=int, default = len(os.environ["SM_HOSTS"]))
    
    # parser.add_argument("--world-size", type=int, default = os.environ["WORLD_SIZE"])
    # parser.add_argument("--rank", type=int, default = os.environ["RANK"])
    # parser.add_argument("--local-rank", type=int, default = os.environ["LOCAL_RANK"])
    args, _ = parser.parse_known_args()
    
    return args


if __name__ == "__main__":

    args = parse_args()

    train(args)
