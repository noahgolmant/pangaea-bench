import hashlib
import os as os
import pathlib
import pprint
import time

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pangaea.datasets.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder
from pangaea.engine.evaluator import Evaluator
from pangaea.engine.trainer import Trainer
from pangaea.utils.collate_fn import get_collate_fn
from pangaea.utils.logger import init_logger
from pangaea.utils.subset_sampler import get_subset_indices
from pangaea.utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)


def get_exp_info(hydra_config: HydraConf) -> dict[str, str]:
    """Create a unique experiment name based on the choices made in the config.

    Args:
        hydra_config (HydraConf): hydra config.

    Returns:
        str: experiment information.
    """
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    cfg_hash = hashlib.sha1(
        OmegaConf.to_yaml(hydra_config).encode(), usedforsecurity=False
    ).hexdigest()[:6]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fm = choices["encoder"]
    decoder = choices["decoder"]
    ds = choices["dataset"]
    task = choices["task"]
    exp_info = {
        "timestamp": timestamp,
        "fm": fm,
        "decoder": decoder,
        "ds": ds,
        "task": task,
        "exp_name": f"{timestamp}_{cfg_hash}_{fm}_{decoder}_{ds}",
    }
    return exp_info


def get_device_and_distributed_setup():
    """Get device and distributed training setup based on environment.
    
    Returns:
        tuple: (device, rank, local_rank, world_size, is_distributed)
    """
    # Check if we're in a distributed training environment
    is_distributed = "RANK" in os.environ and "LOCAL_RANK" in os.environ
    
    if is_distributed:
        # Distributed training setup
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        # Determine the appropriate backend and device
        if torch.backends.mps.is_available():
            # On Mac with MPS, we can't use distributed training effectively
            # Fall back to single device mode
            print("Warning: MPS detected with distributed environment variables. "
                  "Falling back to single-device MPS training.")
            is_distributed = False
            rank = 0
            local_rank = 0
            world_size = 1
            device = torch.device("mps")
        elif torch.cuda.is_available():
            # Use CUDA with NCCL backend
            try:
                torch.distributed.init_process_group(backend="nccl")
                device = torch.device("cuda", local_rank)
                torch.cuda.set_device(device)
            except RuntimeError as e:
                if "NCCL" in str(e):
                    # NCCL not available, try gloo backend
                    print("Warning: NCCL not available, falling back to gloo backend")
                    torch.distributed.init_process_group(backend="gloo")
                    device = torch.device("cuda", local_rank)
                    torch.cuda.set_device(device)
                else:
                    raise e
        else:
            # CPU-only distributed training with gloo
            torch.distributed.init_process_group(backend="gloo")
            device = torch.device("cpu")
    else:
        # Single device setup
        rank = 0
        local_rank = 0 
        world_size = 1
        
        # Device selection priority: MPS > CUDA > CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    return device, rank, local_rank, world_size, is_distributed


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    
    # Get device and distributed setup
    device, rank, local_rank, world_size, is_distributed = get_device_and_distributed_setup()

    # true if training else false
    train_run = cfg.train
    if train_run:
        exp_info = get_exp_info(HydraConfig.get())
        exp_name = exp_info["exp_name"]
        task_name = exp_info["task"]
        exp_dir = pathlib.Path(cfg.work_dir) / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger_path = exp_dir / "train.log"
        config_log_dir = exp_dir / "configs"
        config_log_dir.mkdir(exist_ok=True)
        # init wandb
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
            )
            cfg["wandb_run_id"] = wandb.run.id
        OmegaConf.save(cfg, config_log_dir / "config.yaml")

    else:
        exp_dir = pathlib.Path(cfg.ckpt_dir)
        exp_name = exp_dir.name
        logger_path = exp_dir / "test.log"
        # load training config
        cfg_path = exp_dir / "configs" / "config.yaml"
        cfg = OmegaConf.load(cfg_path)
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
                id=cfg.get("wandb_run_id"),
                resume="allow",
            )

    logger = init_logger(logger_path, rank=rank)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")
    logger.info(f"Distributed training: {is_distributed}")
    if is_distributed:
        logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")

    encoder: Encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)
    logger.info("Built {}.".format(encoder.model_name))

    # prepare the decoder (segmentation/regression)
    decoder: Decoder = instantiate(
        cfg.decoder,
        encoder=encoder,
    )
    decoder.to(device)
    
    # Apply data parallel based on setup
    if is_distributed:
        decoder = torch.nn.parallel.DistributedDataParallel(
            decoder,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=cfg.finetune,
        )
    elif torch.cuda.device_count() > 1 and device.type == "cuda":
        # Multi-GPU single node training
        decoder = torch.nn.DataParallel(decoder)
    # For MPS or single GPU, no parallelization needed
    
    logger.info(
        "Built {} for with {} encoder.".format(
            decoder.module.model_name if hasattr(decoder, 'module') else decoder.model_name, 
            type(encoder).__name__
        )
    )

    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)

    # training
    if train_run or cfg.task.trainer.model_name == "knn_probe":
        # get preprocessor
        train_preprocessor = instantiate(
            cfg.preprocessing.train,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )
        val_preprocessor = instantiate(
            cfg.preprocessing.val,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )

        # get datasets
        raw_train_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="train")
        raw_val_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="val")

        if 0 < cfg.limited_label_train < 1:
            indices = get_subset_indices(
                raw_train_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_train,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_train_dataset = GeoFMSubset(raw_train_dataset, indices)

        if 0 < cfg.limited_label_val < 1:
            indices = get_subset_indices(
                raw_val_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_val,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_val_dataset = GeoFMSubset(raw_val_dataset, indices)

        train_dataset = GeoFMDataset(
            raw_train_dataset, train_preprocessor, cfg.data_replicate
        )
        val_dataset = GeoFMDataset(
            raw_val_dataset, val_preprocessor, cfg.data_replicate
        )

        logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

        logger.info(
            f"Total number of train patches: {len(train_dataset)}\n"
            f"Total number of validation patches: {len(val_dataset)}\n"
        )

        # get train val data loaders
        if is_distributed:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset)
        else:
            train_sampler = None
            val_sampler = None
        
        # Pin memory is not supported on MPS
        use_pin_memory = device.type != "mps"
        # Multiprocessing can cause issues on MPS, so disable workers
        num_workers = 0 if device.type == "mps" else cfg.num_workers
        test_num_workers = 0 if device.type == "mps" else cfg.test_num_workers
            
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=cfg.batch_size,
            shuffle=(train_sampler is None),  # Only shuffle if no sampler
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            # persistent_workers=True causes memory leak
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=get_generator(cfg.seed),
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=cfg.test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            # generator=g,
            drop_last=False,
            collate_fn=collate_fn,
        )

        criterion = instantiate(cfg.criterion)
        optimizer = instantiate(cfg.optimizer, params=decoder.parameters())
        lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            total_iters=len(train_loader) * cfg.task.trainer.n_epochs,
        )

        val_evaluator: Evaluator = instantiate(
            cfg.task.evaluator, val_loader=val_loader, exp_dir=exp_dir, device=device, rank=rank, is_distributed=is_distributed
        )
        trainer: Trainer = instantiate(
            cfg.task.trainer,
            model=decoder,
            train_loader=train_loader,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=val_evaluator,
            exp_dir=exp_dir,
            device=device,
            rank=rank,
            is_distributed=is_distributed,
        )
        # resume training if model_checkpoint is provided
        if cfg.ckpt_dir is not None:
            trainer.load_model(cfg.ckpt_dir)

        trainer.train()

    
    # Evaluation
    test_preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )

    # get datasets
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

    if is_distributed:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = None

    # Pin memory is not supported on MPS
    use_pin_memory = device.type != "mps"
    # Multiprocessing can cause issues on MPS, so disable workers
    test_num_workers = 0 if device.type == "mps" else cfg.test_num_workers

    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=test_num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_evaluator: Evaluator = instantiate(
        cfg.task.evaluator, val_loader=test_loader, exp_dir=exp_dir, device=device, rank=rank, is_distributed=is_distributed
    )

    if cfg.use_final_ckpt:
        model_ckpt_path = get_final_model_ckpt_path(exp_dir)
    else:
        model_ckpt_path = get_best_model_ckpt_path(exp_dir)
        
    if model_ckpt_path is None and not cfg.task.trainer.model_name == "knn_probe":
        raise ValueError(f"No model checkpoint found in {exp_dir}")
    
    test_evaluator.evaluate(decoder, "test_model", model_ckpt_path)

    if cfg.use_wandb and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
