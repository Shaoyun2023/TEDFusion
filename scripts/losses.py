import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from metric.Metric import *
from PIL import Image

import matplotlib.pyplot as plt


class fusion_prompt_loss(nn.Module):
    def __init__(self):
        super(fusion_prompt_loss, self).__init__()
        self.fusion_loss = fusion_loss()

        self.has_analyzed = False  # 添加一个开关

    def forward(self, image_A, image_B, image_fused, htext, hattr, distance_all, task):
        total_loss = 0
        total_ssim_loss = 0
        total_max_loss = 0
        total_color_loss = 0
        total_grad_loss = 0
        total_semantic_loss = 0
        total_hyperbolic_loss = 0


        loss, ssim_loss, max_loss, color_loss, grad_loss, semantic_loss, hyperbolic_loss = self.fusion_loss(
            self.get_images(image_A), self.get_images(image_B),
            self.get_images(image_fused), htext, hattr, distance_all, max_ratio=15, consist_ratio=4, text_ratio=65)

        total_loss += loss
        total_ssim_loss += ssim_loss
        total_max_loss += max_loss
        total_color_loss += color_loss
        total_grad_loss += grad_loss
        total_semantic_loss += semantic_loss
        total_hyperbolic_loss += hyperbolic_loss

        return total_loss, total_ssim_loss, total_max_loss, total_color_loss, total_grad_loss, total_semantic_loss, total_hyperbolic_loss


    def get_images(self, images):
        return images


class fusion_loss(nn.Module):
    def __init__(self):
        super(fusion_loss, self).__init__()
        self.loss_func_ssim = L_SSIM(window_size=48)
        self.loss_func_Grad = GradientMaxLoss()
        self.loss_func_Max = L_Intensity_Max_RGB()
        # the consist loss is an aux loss, you can remove it as your needs
        self.loss_func_Consist = L_Intensity_Consist()
        self.loss_func_color = L_color()
        self.loss_func_sementic = L_Semantic()
        # self.loss_func_hyperbolic = L_hyperbolic()

    def forward(self, image_visible, image_infrared, image_fused, htext, hattr, distance_all, max_ratio=4,
                consist_ratio=1, ssim_ir_ratio=1, ssim_ratio=1, ir_compose=1, color_ratio=20, text_ratio=10,
                max_mode="l1", consist_mode="l1"):
        # max_ratio = 4, consist_ratio = 1, ssim_ir_ratio = 1, ssim_ratio = 1, ir_compose = 1, color_ratio = 20, text_ratio = 10, max_mode = "l1", consist_mode = "l1"
        image_visible_gray = self.rgb2gray(image_visible)
        image_infrared_gray = self.rgb2gray(image_infrared)
        image_fused_gray = self.rgb2gray(image_fused)
        loss_ssim = ssim_ratio * (self.loss_func_ssim(image_visible, image_fused) + ssim_ir_ratio * self.loss_func_ssim(
            image_infrared_gray, image_fused_gray))
        loss_max = max_ratio * self.loss_func_Max(image_visible, image_infrared, image_fused, max_mode)
        # the consist loss is an aux loss, you can remove it as your needs
        loss_consist = consist_ratio * self.loss_func_Consist(image_visible_gray, image_infrared_gray, image_fused_gray,
                                                              ir_compose, consist_mode)
        loss_color = color_ratio * self.loss_func_color(image_visible, image_fused)
        loss_text = text_ratio * self.loss_func_Grad(image_visible_gray, image_infrared_gray, image_fused_gray)
        # loss_semantic = 0.5 * self.loss_func_sementic(htext,hattr)
        loss_semantic = torch.tensor(0).to("cuda:0")
        loss_hyperbolic = 0.005 * distance_all

        total_loss = loss_ssim + loss_max + loss_consist + loss_color + loss_text + loss_semantic + loss_hyperbolic
        return total_loss, loss_ssim, loss_max, loss_color, loss_text, loss_semantic, loss_hyperbolic

    def rgb2gray(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray


class L_Semantic(nn.Module):
    def __init__(self):
        super(L_Semantic, self).__init__()

    def forward(self, htext, hattr):

        # 计算余弦相似度
        alignment_loss1 = F.mse_loss(htext, hattr)

        alignment_loss = alignment_loss1

        return alignment_loss

    def cosine_similarity(self, x1, x2):
        # 归一化
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)

        # 计算余弦相似度
        cosine_sim = torch.sum(x1_norm * x2_norm, dim=1)

        return cosine_sim


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, image_visible, image_fused):
        ycbcr_visible = self.rgb_to_ycbcr(image_visible)
        ycbcr_fused = self.rgb_to_ycbcr(image_fused)

        cb_visible = ycbcr_visible[:, 1, :, :]
        cr_visible = ycbcr_visible[:, 2, :, :]
        cb_fused = ycbcr_fused[:, 1, :, :]
        cr_fused = ycbcr_fused[:, 2, :, :]

        loss_cb = F.l1_loss(cb_visible, cb_fused)
        loss_cr = F.l1_loss(cr_visible, cr_fused)

        loss_color = loss_cb + loss_cr
        return loss_color

    def rgb_to_ycbcr(self, image):
        r = image[:, 0, :, :]
        g = image[:, 1, :, :]
        b = image[:, 2, :, :]

        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b

        ycbcr_image = torch.stack((y, cb, cr), dim=1)
        return ycbcr_image


class L_Intensity_Max_RGB(nn.Module):
    def __init__(self):
        super(L_Intensity_Max_RGB, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, max_mode="l1"):
        gray_visible = torch.mean(image_visible, dim=1, keepdim=True)
        gray_infrared = torch.mean(image_infrared, dim=1, keepdim=True)

        mask = (gray_infrared > gray_visible).float()

        fused_image = mask * image_infrared + (1 - mask) * image_visible
        if max_mode == "l1":
            Loss_intensity = F.l1_loss(fused_image, image_fused)
        else:
            Loss_intensity = F.mse_loss(fused_image, image_fused)
        return Loss_intensity


class L_Intensity_Consist(nn.Module):
    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, image_visible, image_infrared, image_fused, ir_compose, consist_mode="l1"):
        if consist_mode == "l2":
            Loss_intensity = (F.mse_loss(image_visible, image_fused) + ir_compose * F.mse_loss(image_infrared,
                                                                                               image_fused)) / 2
        else:
            Loss_intensity = (F.l1_loss(image_visible, image_fused) + ir_compose * F.l1_loss(image_infrared,
                                                                                             image_fused)) / 2
        return Loss_intensity


# use the GradientMaxLoss or L_Grad
class GradientMaxLoss(nn.Module):
    def __init__(self):
        super(GradientMaxLoss, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1],
                                                       [-2, 0, 2],
                                                       [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1],
                                                       [0, 0, 0],
                                                       [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_A, image_B, image_fuse):
        gradient_A_x, gradient_A_y = self.gradient(image_A)
        gradient_B_x, gradient_B_y = self.gradient(image_B)
        gradient_fuse_x, gradient_fuse_y = self.gradient(image_fuse)
        loss = F.l1_loss(gradient_fuse_x, torch.max(gradient_A_x, gradient_B_x)) + F.l1_loss(gradient_fuse_y,
                                                                                             torch.max(gradient_A_y,
                                                                                                       gradient_B_y))
        return loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x), torch.abs(gradient_y)


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3),
                                    requires_grad=False).cuda()
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3),
                                    requires_grad=False).cuda()
        self.padding = (1, 1, 1, 1)

    def forward(self, image_visible, image_infrared, image_fused):
        gray_visible = self.tensor_RGB2GRAY(image_visible)
        gray_infrared = self.tensor_RGB2GRAY(image_infrared)
        gray_fused = self.tensor_RGB2GRAY(image_fused)

        d1 = self.gradient(gray_visible)
        d2 = self.gradient(gray_infrared)
        df = self.gradient(gray_fused)
        edge_loss = F.l1_loss(torch.max(d1, d2), df)
        return edge_loss

    def gradient(self, image):
        image = F.pad(image, self.padding, mode='replicate')
        gradient_x = F.conv2d(image, self.sobel_x, padding=0)
        gradient_y = F.conv2d(image, self.sobel_y, padding=0)
        return torch.abs(gradient_x) + torch.abs(gradient_y)

    def tensor_RGB2GRAY(self, image):
        b, c, h, w = image.size()
        if c == 1:
            return image
        image_gray = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
        image_gray = image_gray.unsqueeze(dim=1)
        return image_gray


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    return 1 - ret


class L_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        (_, channel_2, _, _) = img2.size()

        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.cuda()
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.cuda()
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


def structure_loss(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1 = F.conv2d(img1, window, padding=padd, groups=channel) - mu1
    sigma2 = F.conv2d(img2, window, padding=padd, groups=channel) - mu2
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C2 = (0.03 * L) ** 2
    loss = (2 * sigma12 + C2) / (2 * sigma1 * sigma2 + C2)

    if size_average:
        ret = loss.mean()
    else:
        ret = loss.mean(1).mean(1).mean(1)

    if full:
        return 1 - ret
    return ret


def normalize_grad(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


def sigmoid1(x):
    return 1 / (1 + math.exp(-x))


def corr2(a, b):
    # 计算均值
    mean_a = torch.mean(a)
    mean_b = torch.mean(b)

    # 减去均值
    a_zero_mean = a - mean_a
    b_zero_mean = b - mean_b

    # 计算相关系数
    r_num = torch.sum(a_zero_mean * b_zero_mean)
    r_den = torch.sqrt(torch.sum(a_zero_mean * a_zero_mean) * torch.sum(b_zero_mean * b_zero_mean))
    r = r_num / r_den

    return r


def SCD_function(A, B, F):
    # 计算差异
    F_minus_B = F - B
    F_minus_A = F - A

    # 计算相关系数
    r = corr2(F_minus_B, A) + corr2(F_minus_A, B)

    return r


def AG_function_torch(image):
    width = image.shape[2]
    width = width - 1
    height = image.shape[3]
    height = height - 1
    [grady, gradx] = torch.gradient(image)
    s = torch.sqrt((torch.square(gradx) + torch.square(grady)) / 2)
    AG = torch.sum(torch.sum(s)) / (width * height)
    return AG


def SF_function_funtion(image_tensor):
    # 计算 RF
    RF = image_tensor[1:] - image_tensor[:-1]
    RF1 = torch.sqrt(torch.mean(torch.mean(RF ** 2)))
    # 计算 CF
    CF = image_tensor[:, 1:] - image_tensor[:, :-1]
    CF1 = torch.sqrt(torch.mean(torch.mean(CF ** 2)))
    # 计算 SF
    SF = torch.sqrt(RF1 ** 2 + CF1 ** 2)
    return SF.item()

