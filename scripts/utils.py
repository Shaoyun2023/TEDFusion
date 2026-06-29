import os
import sys
import random
import clip

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import cv2
import colorsys

from skimage.feature import graycomatrix, graycoprops
from skimage import io

from scripts.losses import fusion_prompt_loss
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision.transforms.functional import to_pil_image
from args_fusion import args




def read_data(root: str):
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = os.path.join(root, "train")
    val_root = os.path.join(root, "eval")
    assert os.path.exists(train_root), "train root: {} does not exist.".format(train_root)
    assert os.path.exists(val_root), "val root: {} does not exist.".format(val_root)

    train_images_visible_path = []
    train_images_infrared_path = []
    train_images_visible_gt_path = []
    train_images_infrared_gt_path = []
    val_images_visible_path = []
    val_images_infrared_path = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']  # 支持的文件后缀类型

    train_visible_root = os.path.join(train_root, "Visible")
    train_infrared_root = os.path.join(train_root, "Infrared")

    train_visible_gt_root = os.path.join(train_root, "Visible_gt")
    train_infrared_gt_root = os.path.join(train_root, "Infrared_gt")

    val_visible_root = os.path.join(val_root, "Visible")
    val_infrared_root = os.path.join(val_root, "Infrared")

    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                          if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                           if os.path.splitext(i)[-1] in supported]

    train_visible_gt_path = [os.path.join(train_visible_gt_root, i) for i in os.listdir(train_visible_gt_root)
                             if os.path.splitext(i)[-1] in supported]
    train_infrared_gt_path = [os.path.join(train_infrared_gt_root, i) for i in os.listdir(train_infrared_gt_root)
                              if os.path.splitext(i)[-1] in supported]

    val_visible_path = [os.path.join(val_visible_root, i) for i in os.listdir(val_visible_root)
                        if os.path.splitext(i)[-1] in supported]
    val_infrared_path = [os.path.join(val_infrared_root, i) for i in os.listdir(val_infrared_root)
                         if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()
    train_visible_gt_path.sort()
    train_infrared_gt_path.sort()
    val_visible_path.sort()
    val_infrared_path.sort()

    assert len(train_visible_path) == len(
        train_infrared_path), ' The length of train dataset does not match. low:{}, high:{}'. \
        format(len(train_visible_path), len(train_infrared_path))
    assert len(val_visible_path) == len(
        val_infrared_path), ' The length of val dataset does not match. low:{}, high:{}'. \
        format(len(val_visible_path), len(val_infrared_path))
    print("Visible and Infrared images check finish")

    for index in range(len(train_visible_path)):
        img_visible_path = train_visible_path[index]
        img_infrared_path = train_infrared_path[index]
        train_images_visible_path.append(img_visible_path)
        train_images_infrared_path.append(img_infrared_path)

        img_visible_gt_path = train_visible_gt_path[index]
        img_infrared_gt_path = train_infrared_gt_path[index]
        train_images_visible_gt_path.append(img_visible_gt_path)
        train_images_infrared_gt_path.append(img_infrared_gt_path)

    for index in range(len(val_visible_path)):
        img_visible_path = val_visible_path[index]
        img_infrared_path = val_infrared_path[index]
        val_images_visible_path.append(img_visible_path)
        val_images_infrared_path.append(img_infrared_path)

    total_dataset_nums = len(train_visible_path) + len(train_infrared_path) + len(train_visible_gt_path) + len(
        train_infrared_gt_path) \
                         + len(val_visible_path) + len(val_infrared_path)
    print("{} images were found in the dataset.".format(total_dataset_nums))
    print("{} visible images for training.".format(len(train_visible_path)))
    print("{} infrared images for training.".format(len(train_infrared_path)))
    print("{} visible gt images for training.".format(len(train_visible_gt_path)))
    print("{} infrared gt images for training.".format(len(train_infrared_gt_path)))
    print("{} visible images for validation.".format(len(val_visible_path)))
    print("{} infrared images for validation.\n".format(len(val_infrared_path)))

    train_low_light_path_list = [train_visible_path, train_infrared_path, train_visible_gt_path, train_infrared_gt_path]
    val_low_light_path_list = [val_visible_path, val_infrared_path]
    return train_low_light_path_list, val_low_light_path_list


def train_one_epoch(model, model_clip, optimizer, lr_scheduler, data_loader, device, epoch):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_text = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_text = model_text.to(device)

    model.train()
    model_clip.eval()
    loss_function_prompt = fusion_prompt_loss()

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)

    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_semantic_loss = torch.zeros(1).to(device)
    accu_hyperbolic_loss = torch.zeros(1).to(device)

    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name = data
        text_line = []

        # 批量处理
        batch_images1 = [to_pil_image(img) for img in I_A]  # 将每个张量转换为 PIL 图像
        ###################################################################################################
        # 使用 Blip 生成文本描述
        inputs1 = processor(batch_images1, return_tensors="pt")  # 处理批量图像
        inputs1 = inputs1.to(device)
        out1 = model_text.generate(**inputs1)  # 生成文本描述
        caption1 = processor.decode(out1[0], skip_special_tokens=True)
        text1 = clip.tokenize(caption1).to(args.device)
        # print(text1)

        # 批量处理
        batch_images2 = [to_pil_image(img) for img in I_B]  # 将每个张量转换为 PIL 图像

        # 使用 Blip 生成文本描述
        inputs2 = processor(batch_images2, return_tensors="pt")  # 处理批量图像
        inputs2 = inputs2.to(device)
        out2 = model_text.generate(**inputs2)  # 生成文本描述
        caption2 = processor.decode(out2[0], skip_special_tokens=True)
        text2 = clip.tokenize(caption2).to(args.device)


        h1, s1, v1 = mergy_RGB_to_HSV(I_A)
        s_avg1 = calculate_edge(I_A)

        v_avg1 = np.mean(v1)
        gray_tensor1 = rgb_to_gray(I_A)
        gray_tensor1 = torch.sum(gray_tensor1, dim=0)
        gray_tensor1 = torch.sum(gray_tensor1, dim=1)
        gray_tensor1 = (gray_tensor1.detach().cpu().numpy() * 255).astype(np.uint8)
        glcm1 = graycomatrix(gray_tensor1, [2, 4, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                            256, symmetric=True, normed=True)
        glcm1 = np.array(glcm1, dtype=np.float32)

        energy1 = graycoprops(glcm1, 'energy')
        energy_avg1 = np.mean(energy1)
        homogeneity1 = graycoprops(glcm1, 'homogeneity')
        homogeneity_avg1 = np.mean(homogeneity1)
        glcm1 = torch.from_numpy(glcm1)
        texture_complexity1 = 0.7 * energy_avg1 + 0.3 * homogeneity_avg1
        contrast1 = calculate_contrast(I_A).detach().cpu().numpy()

        h2, s2, v2 = mergy_RGB_to_HSV(I_B)
        s_avg2 = calculate_edge(I_B)
        v_avg2 = np.mean(v2)
        gray_tensor2 = rgb_to_gray(I_B)
        gray_tensor2 = torch.sum(gray_tensor2, dim=0)
        gray_tensor2 = torch.sum(gray_tensor2, dim=1)
        gray_tensor2 = (gray_tensor2.detach().cpu().numpy() * 255).astype(np.uint8)
        glcm2 = graycomatrix(gray_tensor2, [2, 4, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                             256, symmetric=True, normed=True)
        glcm2 = np.array(glcm2, dtype=np.float32)
        energy2 = graycoprops(glcm2, 'energy')
        energy_avg2 = np.mean(energy2)
        homogeneity2 = graycoprops(glcm2, 'homogeneity')
        homogeneity_avg2 = np.mean(homogeneity2)
        glcm2 = torch.from_numpy(glcm2)
        texture_complexity2 = 0.6 * energy_avg2 + 0.4 * homogeneity_avg2
        contrast2 = calculate_contrast(I_B).detach().cpu().numpy()


        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)

        feature = np.stack([s_avg1, v_avg1, texture_complexity1, contrast1,s_avg2, v_avg2, texture_complexity2, contrast2], axis=-1)
        feature = torch.tensor(feature)
        feature = feature.to("cuda:0")
        feature = feature.float()

        # print(feature)




        # text = torch.cat([text1,text2],dim=1)

        I_fused,htext,hattr,distance_all = model(I_A, I_B, text1,text2, feature)


        loss, loss_ssim, loss_max, loss_color, loss_text, loss_semantic,loss_hyperbolic = loss_function_prompt(I_A_gt, I_B_gt, I_fused,htext,hattr,distance_all, task)

        loss.backward()

        accu_total_loss += loss.detach()
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text.detach()
        accu_semantic_loss += loss_semantic.detach()
        accu_hyperbolic_loss += loss_hyperbolic.detach()

        lr = optimizer.param_groups[0]["lr"]

        data_loader.desc = "[train epoch {}] loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f} semantic loss: {:.3f} hyperbolic loss: {:.3f} lr: {:.6f}".format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_semantic_loss.item() / (step + 1),accu_hyperbolic_loss.item() / (step + 1), lr)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (
                step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), accu_semantic_loss.item() / (step + 1),accu_hyperbolic_loss.item() / (step + 1), lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, lr, filefold_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model_text = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_text = model_text.to(device)

    loss_function_prompt = fusion_prompt_loss()


    model.eval()
    accu_total_loss = torch.zeros(1).to(device)
    accu_ssim_loss = torch.zeros(1).to(device)
    accu_max_loss = torch.zeros(1).to(device)
    accu_color_loss = torch.zeros(1).to(device)
    accu_text_loss = torch.zeros(1).to(device)
    accu_semantic_loss = torch.zeros(1).to(device)
    accu_hyperbolic_loss = torch.zeros(1).to(device)
    save_epoch = 1
    save_length = 60
    cnt = 0
    save_RGB_fuse = True

    if torch.cuda.is_available():
        loss_function_prompt = loss_function_prompt.to(device)

    if epoch % save_epoch == 0:
        evalfold_path = os.path.join(filefold_path, str(epoch))
        if os.path.exists(evalfold_path) is False:
            os.makedirs(evalfold_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        I_A, I_B, I_A_gt, I_B_gt, I_full, task, name = data
        text_line = []

        # 批量处理
        batch_images1 = [to_pil_image(img) for img in I_A]  # 将每个张量转换为 PIL 图像
        ###################################################################################################
        # 使用 Blip 生成文本描述
        inputs1 = processor(batch_images1, return_tensors="pt")  # 处理批量图像
        inputs1 = inputs1.to(device)
        out1 = model_text.generate(**inputs1)  # 生成文本描述
        caption1 = processor.decode(out1[0], skip_special_tokens=True)
        text1 = clip.tokenize(caption1).to(args.device)
        # print(text1)

        # 批量处理
        batch_images2 = [to_pil_image(img) for img in I_B]  # 将每个张量转换为 PIL 图像

        # 使用 Blip 生成文本描述
        inputs2 = processor(batch_images2, return_tensors="pt")  # 处理批量图像
        inputs2 = inputs2.to(device)
        out2 = model_text.generate(**inputs2)  # 生成文本描述
        caption2 = processor.decode(out2[0], skip_special_tokens=True)
        text2 = clip.tokenize(caption2).to(args.device)

        ####################################################################################################

        h1, s1, v1 = mergy_RGB_to_HSV(I_A)
        s_avg1 = np.mean(s1)
        v_avg1 = np.mean(v1)
        gray_tensor1 = rgb_to_gray(I_A)
        gray_tensor1 = torch.sum(gray_tensor1, dim=0)
        gray_tensor1 = torch.sum(gray_tensor1, dim=1)
        gray_tensor1 = (gray_tensor1.detach().cpu().numpy() * 255).astype(np.uint8)
        glcm1 = graycomatrix(gray_tensor1, [2, 4, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                             256, symmetric=True, normed=True)
        glcm1 = np.array(glcm1, dtype=np.float32)
        energy1 = graycoprops(glcm1, 'energy')
        energy_avg1 = np.mean(energy1)
        homogeneity1 = graycoprops(glcm1, 'homogeneity')
        homogeneity_avg1 = np.mean(homogeneity1)
        glcm1 = torch.from_numpy(glcm1)
        texture_complexity1 = 0.7 * energy_avg1 + 0.3 * homogeneity_avg1
        contrast1 = calculate_contrast(I_A).detach().cpu().numpy()

        h2, s2, v2 = mergy_RGB_to_HSV(I_B)
        s_avg2 = np.mean(s2)
        v_avg2 = np.mean(v2)
        gray_tensor2 = rgb_to_gray(I_B)
        gray_tensor2 = torch.sum(gray_tensor2, dim=0)
        gray_tensor2 = torch.sum(gray_tensor2, dim=1)
        gray_tensor2 = (gray_tensor2.detach().cpu().numpy() * 255).astype(np.uint8)
        glcm2 = graycomatrix(gray_tensor2, [2, 4, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                             256, symmetric=True, normed=True)
        glcm2 = np.array(glcm2, dtype=np.float32)
        energy2 = graycoprops(glcm2, 'energy')
        energy_avg2 = np.mean(energy2)
        homogeneity2 = graycoprops(glcm2, 'homogeneity')
        homogeneity_avg2 = np.mean(homogeneity2)
        glcm2 = torch.from_numpy(glcm2)
        texture_complexity2 = 0.6 * energy_avg2 + 0.4 * homogeneity_avg2
        contrast2 = calculate_contrast(I_B).detach().cpu().numpy()


        if torch.cuda.is_available():
            I_A = I_A.to(device)
            I_B = I_B.to(device)
            I_A_gt = I_A_gt.to(device)
            I_B_gt = I_B_gt.to(device)
            I_full = I_full.to(device)

        feature = np.stack([s_avg1, v_avg1, texture_complexity1, contrast1,s_avg2, v_avg2, texture_complexity2, contrast2], axis=-1)
        feature = torch.tensor(feature)
        feature = feature.to("cuda:0")
        feature = feature.float()



        I_fused,htext,hattr,distance_all = model(I_A, I_B, text1,text2, feature)

        if epoch % save_epoch == 0:
            if cnt <= save_length:
                fused_img_Y = tensor2numpy(I_fused)
                img_full = tensor2numpy(I_full)
                img_ir = tensor2numpy(I_B_gt)

                save_pic(fused_img_Y, evalfold_path, str(name[0]))
                if save_RGB_fuse == True:
                    save_pic(img_full, evalfold_path, str(name[0]) + "vis")
                    save_pic(img_ir, evalfold_path, str(name[0]) + "ir")
                cnt += 1

        loss, loss_ssim, loss_max, loss_color, loss_text, loss_semantic,loss_hyperbolic = loss_function_prompt(I_A_gt, I_B_gt, I_fused,htext,hattr,distance_all, task)

        accu_total_loss += loss
        accu_ssim_loss += loss_ssim.detach()
        accu_max_loss += loss_max.detach()
        accu_color_loss += loss_color.detach()
        accu_text_loss += loss_text
        accu_semantic_loss += loss_semantic
        accu_hyperbolic_loss += loss_hyperbolic

        data_loader.desc = "[val epoch {}] loss: {:.3f}  ssim loss: {:.3f}  max loss: {:.3f}  color loss: {:.3f}  text loss: {:.3f} semantic loss: {:.3f} hyperbolic loss: {:.3f} lr: {:.6f}".format(
            epoch, accu_total_loss.item() / (step + 1),
            accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (step + 1), accu_color_loss.item() / (step + 1),
            accu_text_loss.item() / (step + 1), accu_semantic_loss.item() / (step + 1), accu_hyperbolic_loss.item() / (step + 1), lr)

    return accu_total_loss.item() / (step + 1), accu_ssim_loss.item() / (step + 1), accu_max_loss.item() / (
                step + 1), accu_color_loss.item() / (step + 1), accu_text_loss.item() / (step + 1), accu_semantic_loss.item() / (step + 1), accu_hyperbolic_loss.item() / (step + 1)


def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])

    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])

    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def save_pic(outputpic, path, index: str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic = outputpic[:, :, ::-1]
    save_path = os.path.join(path, index + ".png")
    cv2.imwrite(save_path, outputpic)


def show_img(images, imagesl, B):
    for index in range(B):
        img = images[index, :]
        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(1)
        plt.imshow(img_np)
        img = imagesl[index, :]

        img_np = np.array(img.permute(1, 2, 0).detach().cpu())
        plt.figure(2)
        plt.imshow(img_np)
        plt.show(block=True)


def tensor2numpy(R_tensor):
    R = R_tensor.squeeze(0).cpu().detach().numpy()
    R = np.transpose(R, [1, 2, 0])
    return R


def tensor2numpy_single(L_tensor):
    L = L_tensor.squeeze(0)
    L_3 = torch.cat([L, L, L], dim=0)
    L_3 = L_3.cpu().detach().numpy()
    L_3 = np.transpose(L_3, [1, 2, 0])
    return L_3


def mergy_RGB_to_HSV(img):  # [4,3,96,96]

    batch_size, channels, height, width = img.shape
    img_numpy = img.cpu().numpy()
    hsv_images = []
    for i in range(batch_size):
        single_img = np.transpose(img_numpy[i], [1, 2, 0])
        img_HSV = cv2.cvtColor(single_img, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(img_HSV)

    return H, S, V


def rgb_to_gray(rgb_tensor):
    r, g, b = rgb_tensor[:, 0, :, :], rgb_tensor[:, 1, :, :], rgb_tensor[:, 2, :, :]
    gray_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray_tensor.unsqueeze(1)


def gray_cooccurrence(image, window_size=1):
    # 计算图像尺寸
    height, width = image[2], image[3]

    # 初始化灰度共生矩阵
    cooccurrence_matrix = np.zeros((256, 256), dtype=np.float32)

    # 遍历图像中的每个像素
    for y in range(height - window_size):
        for x in range(width - window_size):
            # 获取当前窗口内的像素值
            window = image[y:y + window_size, x:x + window_size]

            # 获取窗口内的灰度值组合
            unique_values = np.unique(window)

            # 更新灰度共生矩阵
            for i, u in enumerate(unique_values):
                for j, v in enumerate(unique_values):
                    cooccurrence_matrix[u, v] += np.sum((window == u) & (window == v))

    # 归一化处理
    norm_matrix = cooccurrence_matrix / (width * height - window_size ** 2 + 1)

    return norm_matrix


def calculate_contrast(tensor):
    # 计算每个通道的标准差作为对比度的近似
    mean = tensor.mean(dim=(2, 3), keepdim=True)
    var = ((tensor - mean) ** 2).mean(dim=(2, 3), keepdim=True)
    std = torch.sqrt(var)
    return std.mean()


def calculate_edge(tensor):
    # 计算水平和垂直梯度
    grad_x = np.gradient(tensor, axis=1)  # 水平方向梯度
    grad_y = np.gradient(tensor, axis=0)  # 垂直方向梯度

    # 计算梯度幅值（欧几里得范数）
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 计算边缘强度（取梯度幅值的均值）
    edge_strength = np.mean(gradient_magnitude)

    return edge_strength

