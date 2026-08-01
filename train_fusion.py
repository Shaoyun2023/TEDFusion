import os
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import clip
from data.prompt_dataset import PromptDataSet
from data.simple_dataset import SimpleDataSet

from model.TEDFusion_model import Text_IF as create_model
from scripts.utils import read_data, train_one_epoch, evaluate, create_lr_scheduler
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import transforms as T
import random
import numpy as np


TRAIN_SEED = 122345
UPSAMPLE_MODULE_NAMES = ("up4_3", "up3_2", "up2_1", "up2_1_2")


def set_seed(seed):
    """Configure every random source used by this training program."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    """Seed Python and NumPy inside each DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_rng_state(train_generator, val_generator):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "train_generator": train_generator.get_state(),
        "val_generator": val_generator.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state, train_generator, val_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    train_generator.set_state(state["train_generator"])
    val_generator.set_state(state["val_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def freeze_pretrained_upsamplers(model):
    """Keep the pretrained PixelShuffle filters phase-balanced during fine-tuning."""
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    frozen_parameters = 0
    for module_name in UPSAMPLE_MODULE_NAMES:
        module = getattr(base_model, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = False
            frozen_parameters += parameter.numel()
    return frozen_parameters


@torch.no_grad()
def initialize_pixelshuffle_phases(model):
    """Give all four sub-pixel phases the same from-scratch initialization."""
    initialized_parameters = 0
    for module_name in UPSAMPLE_MODULE_NAMES:
        weight = getattr(model, module_name).body[0].weight
        phase_weight = weight.view(weight.shape[0] // 4, 4, *weight.shape[1:])
        reference_phase = phase_weight[:, :1].clone()
        phase_weight.copy_(reference_phase.expand_as(phase_weight))
        initialized_parameters += weight.numel()
    return initialized_parameters

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    set_seed(TRAIN_SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using deterministic training seed: {}".format(TRAIN_SEED))

    if args.dataset_path is not None:
        args.low_light_path = args.dataset_path
        args.over_exposure_path = args.dataset_path
        args.ir_low_contrast_path = args.dataset_path
        args.ir_noise_path = args.dataset_path

    if os.path.exists("./experiments") is False:
        os.makedirs("./experiments")

    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filefold_path = "./experiments/TEDFusion_train_{}".format(file_name)
    os.makedirs(filefold_path)
    file_img_path = os.path.join(filefold_path, "img")
    os.makedirs(file_img_path)
    file_weights_path = os.path.join(filefold_path, "weights")
    os.makedirs(file_weights_path)
    file_log_path = os.path.join(filefold_path, "log")
    os.makedirs(file_log_path)

    tb_writer = SummaryWriter(log_dir=file_log_path)

    best_val_loss = 1e5
    start_epoch = 0

    print("Loading IVF Fusion and Low-Light Task!")
    if args.low_light_path is not None:
        train_low_light_path_list, val_low_light_path_list = read_data(args.low_light_path)
    else:
        train_low_light_path_list = val_low_light_path_list = None

    print("Loading IVF Fusion and Over-Exposure Task!")
    if args.over_exposure_path is not None:
        train_over_exposure_path_list, val_over_exposure_path_list = read_data(args.over_exposure_path)
    else:
        train_over_exposure_path_list = val_over_exposure_path_list = None

    print("Loading IVF Fusion and ir_low_contrast Task!")
    if args.ir_low_contrast_path is not None:
        train_ir_low_contrast_path_list, val_ir_low_contrast_path_list = read_data(args.ir_low_contrast_path)
    else:
        train_ir_low_contrast_path_list = val_ir_low_contrast_path_list = None

    print("Loading IVF Fusion and ir_noise_path Task!")
    if args.ir_noise_path is not None:
        train_ir_noise_path_list, val_ir_noise_path_list = read_data(args.ir_noise_path)
    else:
        train_ir_noise_path_list = val_ir_noise_path_list = None

    data_transform = {
        "train": T.Compose([T.RandomCrop(96),
                            T.RandomHorizontalFlip(0.5),
                            T.RandomVerticalFlip(0.5),
                            T.ToTensor()]),

        "val": T.Compose([T.CenterCrop(96),
                          T.Resize_16(),
                          T.ToTensor()])}

    train_dataset = PromptDataSet(train_low_light_path_list=train_low_light_path_list,
                                  val_low_light_path_list=val_low_light_path_list,
                                  train_over_exposure_path_list=train_over_exposure_path_list,
                                  val_over_exposure_path_list=val_over_exposure_path_list,
                                  train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
                                  val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
                                  train_ir_noise_path_list=train_ir_noise_path_list,
                                  val_ir_noise_path_list=val_ir_noise_path_list,
                                  phase="train",
                                  transform=data_transform["train"],
                                  samples_per_epoch=args.samples_per_epoch,
                                  max_val_samples=args.max_val_samples)

    val_dataset = PromptDataSet(train_low_light_path_list=train_low_light_path_list,
                                  val_low_light_path_list=val_low_light_path_list,
                                  train_over_exposure_path_list=train_over_exposure_path_list,
                                  val_over_exposure_path_list=val_over_exposure_path_list,
                                  train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
                                  val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
                                  train_ir_noise_path_list=train_ir_noise_path_list,
                                  val_ir_noise_path_list=val_ir_noise_path_list,
                                  phase="val",
                                  transform=data_transform["val"],
                                  samples_per_epoch=args.samples_per_epoch,
                                  max_val_samples=args.max_val_samples)

    print("Training draws {} samples per epoch from {} unique image pairs ({} unique dataset slot(s)).".format(
        len(train_dataset), train_dataset.num_unique_pairs, len(train_dataset.task_keys)
    ))
    print("Validation uses {} deterministic image pairs.".format(len(val_dataset)))

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_generator = torch.Generator()
    train_generator.manual_seed(TRAIN_SEED)
    val_generator = torch.Generator()
    val_generator.manual_seed(TRAIN_SEED + 1)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn,
                                               worker_init_fn=seed_worker,
                                               generator=train_generator)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn,
                                             worker_init_fn=seed_worker,
                                             generator=val_generator)

    model_clip, _ = clip.load("ViT-B/32", device=device)
    # model = create_model(model_clip, curvature=1.5).to(device)
    model = create_model(model_clip).to(device)

    for param in model.model_clip.parameters():
        param.requires_grad = False

    checkpoint = None
    checkpoint_args = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint_args = checkpoint.get("args")

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
    elif args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))
    else:
        initialized_parameters = initialize_pixelshuffle_phases(model)
        print("Checkerboard guard: phase-aligned {:,} PixelShuffle parameters.".format(
            initialized_parameters
        ))

    resumed_from_pretrained = bool(getattr(checkpoint_args, "weights", ""))
    is_finetuning = args.weights != "" or resumed_from_pretrained
    checkpoint_trained_upsamplers = bool(getattr(checkpoint_args, "train_upsamplers", False))
    protect_upsamplers = is_finetuning and not (args.train_upsamplers or checkpoint_trained_upsamplers)
    if protect_upsamplers:
        frozen_parameters = freeze_pretrained_upsamplers(model)
        print("Fine-tuning guard: froze {:,} pretrained PixelShuffle-convolution parameters.".format(
            frozen_parameters
        ))

    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    pg = [p for p in model.parameters() if p.requires_grad]
    learning_rate = args.lr
    if learning_rate is None:
        learning_rate = 2e-5 if is_finetuning else 1e-4
    weight_decay = args.weight_decay
    if weight_decay is None:
        weight_decay = 1e-4 if is_finetuning else 5e-2
    print("Optimizer: AdamW(lr={}, weight_decay={})".format(learning_rate, weight_decay))
    optimizer = optim.AdamW(pg, lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if checkpoint is not None:
        checkpoint_seed = checkpoint.get("seed", getattr(checkpoint.get("args"), "seed", None))
        if checkpoint_seed is not None and checkpoint_seed != TRAIN_SEED:
            raise ValueError(
                "The checkpoint seed ({}) does not match TRAIN_SEED ({})."
                .format(checkpoint_seed, TRAIN_SEED)
            )
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        if "rng_state" in checkpoint:
            set_rng_state(checkpoint["rng_state"], train_generator, val_generator)
        else:
            print("WARNING: checkpoint has no RNG state; resumed training cannot be exactly reproduced.")

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_ssim_loss, train_max_loss, train_color_loss, train_text_loss, train_semantic_loss,train_hyperbolic_loss, lr = train_one_epoch(model=model,
                                              model_clip=model_clip,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                lr_scheduler=lr_scheduler,
                                                device=device,
                                                epoch=epoch,
                                                grad_clip_norm=args.grad_clip_norm)

        tb_writer.add_scalar("train_total_loss", train_loss, epoch)
        tb_writer.add_scalar("train_ssim_loss", train_ssim_loss, epoch)
        tb_writer.add_scalar("train_max_loss", train_max_loss, epoch)
        tb_writer.add_scalar("train_color_loss", train_color_loss, epoch)
        tb_writer.add_scalar("train_text_loss", train_text_loss, epoch)
        tb_writer.add_scalar("train_semantic_loss", train_semantic_loss, epoch)
        tb_writer.add_scalar("train_hyperbolic_loss", train_semantic_loss, epoch)

        if epoch % args.val_every_epcho == 0 and epoch != 0:
            val_loss, val_ssim_loss, val_max_loss, val_color_loss, val_text_loss, val_semantic_loss, val_hyperbolic_loss = evaluate(model=model,
                                         data_loader=val_loader,
                                         device=device,
                                         epoch=epoch, lr=lr, filefold_path=file_img_path)

            tb_writer.add_scalar("val_total_loss", val_loss, epoch)
            tb_writer.add_scalar("val_ssim_loss", val_ssim_loss, epoch)
            tb_writer.add_scalar("val_max_loss", val_max_loss, epoch)
            tb_writer.add_scalar("val_color_loss", val_color_loss, epoch)
            tb_writer.add_scalar("val_text_loss", val_text_loss, epoch)
            tb_writer.add_scalar("val_semantic_loss", val_semantic_loss, epoch)
            tb_writer.add_scalar("val_hyperbolic_loss", val_semantic_loss, epoch)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            model_state = model.module.state_dict() if args.use_dp == True else model.state_dict()
            save_file = {
                "model": model_state,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "seed": TRAIN_SEED,
                "best_val_loss": best_val_loss,
                "rng_state": get_rng_state(train_generator, val_generator),
            }

            if is_best:
                torch.save(save_file, file_weights_path + "/" + "checkpoint.pth")

            torch.save(save_file, file_weights_path + "/" + "checkpoint_lastest.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--epochs', type=int, default=15)

    # set the appropriate batch-size value for your device
    # parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (default: 2e-5 for fine-tuning, 1e-4 from scratch)')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='AdamW weight decay (default: 1e-4 for fine-tuning, 5e-2 from scratch)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='clip gradient norm to stabilize fine-tuning; <=0 disables it')
    parser.add_argument('--samples_per_epoch', type=int, default=800,
                        help='random training pairs drawn per epoch; 800 matches the original small-MSRS run')
    parser.add_argument('--max_val_samples', type=int, default=80,
                        help='maximum number of deterministic validation pairs')

    parser.add_argument('--dataset_path', type=str, default=None,
                        help='set one dataset root for all four task slots')
    parser.add_argument('--low_light_path', type=str, default="./dataset/train_MSRS")
    parser.add_argument('--over_exposure_path', type=str, default="./dataset/train_MSRS")
    parser.add_argument('--ir_low_contrast_path', type=str, default="./dataset/train_MSRS")
    parser.add_argument('--ir_noise_path', type=str, default="./dataset/train_MSRS")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--train_upsamplers', action='store_true',
                        help='also update PixelShuffle convolutions when fine-tuning (not recommended for MSRS)')
    parser.add_argument('--val_every_epcho', type=int, default=2, help='val every epcho')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default = False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')

    opt = parser.parse_args()

    main(opt)
