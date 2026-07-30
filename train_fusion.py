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


TRAIN_SEED = 104


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

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    set_seed(TRAIN_SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using deterministic training seed: {}".format(TRAIN_SEED))

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
                              transform=data_transform["train"])

    val_dataset = PromptDataSet(train_low_light_path_list=train_low_light_path_list,
                                  val_low_light_path_list=val_low_light_path_list,
                                  train_over_exposure_path_list=train_over_exposure_path_list,
                                  val_over_exposure_path_list=val_over_exposure_path_list,
                                  train_ir_low_contrast_path_list=train_ir_low_contrast_path_list,
                                  val_ir_low_contrast_path_list=val_ir_low_contrast_path_list,
                                  train_ir_noise_path_list=train_ir_noise_path_list,
                                  val_ir_noise_path_list=val_ir_noise_path_list,
                                  phase="val",
                            transform=data_transform["val"])

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

    if args.use_dp == True:
        model = torch.nn.DataParallel(model).cuda()

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        print(model.load_state_dict(weights_dict, strict=False))


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint_seed = checkpoint.get("seed", getattr(checkpoint.get("args"), "seed", None))
        if checkpoint_seed is not None and checkpoint_seed != TRAIN_SEED:
            raise ValueError(
                "The checkpoint seed ({}) does not match TRAIN_SEED ({})."
                .format(checkpoint_seed, TRAIN_SEED)
            )
        model.load_state_dict(checkpoint['model'])
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
                                                epoch=epoch)

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
    parser.add_argument('--lr', type=float, default=0.0001)

    parser.add_argument('--low_light_path', type=str, default="./dataset/train_MSRS2")
    parser.add_argument('--over_exposure_path', type=str, default="./dataset/train_MSRS2")
    parser.add_argument('--ir_low_contrast_path', type=str, default="./dataset/train_MSRS2")
    parser.add_argument('--ir_noise_path', type=str, default="./dataset/train_MSRS2")

    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--val_every_epcho', type=int, default=2, help='val every epcho')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--use_dp', default = False, help='use dp-multigpus')
    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')

    opt = parser.parse_args()

    main(opt)
