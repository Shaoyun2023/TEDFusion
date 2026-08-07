import os
import numpy as np
from PIL import Image
import cv2
import clip.clip
import torch
from torchvision.transforms import functional as F
from model.TEDFusion_model import TEDFusion1 as create_model
from model.TEDFusion_model import TEDFusion1 as create_model1


import argparse

import random
import numpy as np

from skimage.feature import graycomatrix, graycoprops

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def set_seed_thread(seed):
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    set_seed_thread(2010)

    root_path = args.dataset_path
    save_path = args.save_path
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']
    # text_line = args.input_text

    visible_root = os.path.join(root_path, "Visible")
    infrared_root = os.path.join(root_path, "Infrared")

    visible_path = [os.path.join(visible_root, i) for i in os.listdir(visible_root)
                  if os.path.splitext(i)[-1] in supported]
    infrared_path = [os.path.join(infrared_root, i) for i in os.listdir(infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    visible_path.sort()
    infrared_path.sort()

    print("Find the number of visible image: {},  the number of the infrared image: {}".format(len(visible_path), len(infrared_path)))
    assert len(visible_path) == len(infrared_path), "The number of the source images does not match!"

    print("Begin to run!")
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device=device)
        model = create_model1(model_clip).to(device)

        model_weight_path = args.weights_path
        model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
        model.eval()

    for i in range(len(visible_path)):
        ir_path = infrared_path[i]
        vi_path = visible_path[i]

        img_name = vi_path.replace("\\", "/").split("/")[-1]

        assert os.path.exists(ir_path), "file: '{}' dose not exist.".format(ir_path)
        assert os.path.exists(vi_path), "file: '{}' dose not exist.".format(vi_path)

        ir = Image.open(ir_path).convert(mode="RGB")
        vi = Image.open(vi_path).convert(mode="RGB")

        width, height = vi.size
        assert ir.size == vi.size, "The visible and infrared image sizes do not match: {} vs {}".format(
            vi.size, ir.size
        )

        # Three PixelUnshuffle stages require a multiple of 8. Reflection padding
        # preserves the original pixels and avoids nearest-neighbour resize artifacts.
        pad_right = (-width) % 8
        pad_bottom = (-height) % 8
        if pad_right or pad_bottom:
            padding = [0, 0, pad_right, pad_bottom]
            ir = F.pad(ir, padding, padding_mode="reflect")
            vi = F.pad(vi, padding, padding_mode="reflect")

        ir = F.to_tensor(ir)
        vi = F.to_tensor(vi)



        ir = ir.unsqueeze(0).cuda()
        vi = vi.unsqueeze(0).cuda()

        h1, s1, v1 = mergy_RGB_to_HSV(ir)
        s_avg1 = np.mean(s1)
        v_avg1 = np.mean(v1)
        gray_tensor1 = rgb_to_gray(ir)
        gray_tensor1 = torch.sum(gray_tensor1, dim=0)
        gray_tensor1 = torch.sum(gray_tensor1, dim=1)
        gray_tensor1 = (gray_tensor1.detach().cpu().numpy() * 255).astype(np.uint8)
        glcm1 = graycomatrix(gray_tensor1, [2, 4, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                             256, symmetric=True, normed=True)
        energy1 = graycoprops(glcm1, 'energy')
        energy_avg1 = np.mean(energy1)
        homogeneity1 = graycoprops(glcm1, 'homogeneity')
        homogeneity_avg1 = np.mean(homogeneity1)
        glcm1 = torch.from_numpy(glcm1)
        texture_complexity1 = 0.7 * energy_avg1 + 0.3 * homogeneity_avg1
        contrast1 = calculate_contrast(ir).detach().cpu().numpy()

        h2, s2, v2 = mergy_RGB_to_HSV(vi)
        s_avg2 = np.mean(s2)
        v_avg2 = np.mean(v2)
        gray_tensor2 = rgb_to_gray(vi)
        gray_tensor2 = torch.sum(gray_tensor2, dim=0)
        gray_tensor2 = torch.sum(gray_tensor2, dim=1)
        gray_tensor2 = (gray_tensor2.detach().cpu().numpy() * 255).astype(np.uint8)
        glcm2 = graycomatrix(gray_tensor2, [2, 4, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                             256, symmetric=True, normed=True)
        energy2 = graycoprops(glcm2, 'energy')
        energy_avg2 = np.mean(energy2)
        homogeneity2 = graycoprops(glcm2, 'homogeneity')
        homogeneity_avg2 = np.mean(homogeneity2)
        glcm2 = torch.from_numpy(glcm2)
        texture_complexity2 = 0.6 * energy_avg2 + 0.4 * homogeneity_avg2
        contrast2 = calculate_contrast(vi).detach().cpu().numpy()

        feature = np.stack([s_avg1, v_avg1, texture_complexity1, contrast1,s_avg2, v_avg2, texture_complexity2, contrast2], axis=-1)
        feature = torch.tensor(feature)
        feature = feature.to("cuda:0")
        feature = feature.float()

        with torch.no_grad():

            # The dominant artifact has a 2-pixel phase, so retain all four
            # parity combinations without increasing inference time.
            shifts = [(0, 0), (0, 1), (1, 0), (1, 1)]
            outputs = []

            for dy, dx in shifts:
                vi_shifted = torch.roll(vi, shifts=(dy, dx), dims=(2, 3))
                ir_shifted = torch.roll(ir, shifts=(dy, dx), dims=(2, 3))
                out = model(vi_shifted, ir_shifted, feature)
                out_unshifted = torch.roll(out, shifts=(-dy, -dx), dims=(2, 3))
                outputs.append(out_unshifted)

            i = torch.stack(outputs).mean(dim=0)
            i = i[:, :, :height, :width]
            # i = model(vi, ir,feature)
            text1 = torch.tensor(0)
            text2 = torch.tensor(0)
            # i = model(vi, ir, feature,text1,text2)
            fused_img_Y = tensor2numpy(i)

            save_pic(fused_img_Y, save_path, img_name)

        print("Save the {}".format(img_name))
    print("Finish! The results are saved in {}.".format(save_path))

def tensor2numpy(img_tensor):
    img = img_tensor[0].squeeze(0).cpu().detach().numpy()
    # img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).detach().cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])
    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])
    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def save_pic(outputpic, path, index : str):
    # Do not stretch every output independently: it amplifies faint periodic
    # decoder errors in low-light images.
    outputpic = np.clip(outputpic, 0.0, 1.0)
    outputpic = np.rint(outputpic * 255.0).astype(np.uint8)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index).replace(".jpg", ".png")
    cv2.imwrite(save_path, outputpic)

def mergy_RGB_to_HSV(img):
    img = img.squeeze(0).cpu().numpy()
    img = np.transpose(img, [1, 2, 0])

    img_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(img_HSV)
    return H,S,V

def rgb_to_gray(rgb_tensor):
    r, g, b = rgb_tensor[:, 0, :, :], rgb_tensor[:, 1, :, :], rgb_tensor[:, 2, :, :]
    gray_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray_tensor.unsqueeze(1)

def calculate_contrast(tensor):
    # 计算每个通道的标准差作为对比度的近似
    mean = tensor.mean(dim=(2, 3), keepdim=True)
    var = ((tensor - mean)**2).mean(dim=(2, 3), keepdim=True)
    std = torch.sqrt(var)
    return std.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='test data root path')
    parser.add_argument('--weights_path', type=str, required=True, help='initial weights path')
    parser.add_argument('--save_path', type=str, default='./results', help='output save image path')
    # parser.add_argument('--input_text', type=str, required=True, help='text control input')

    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()
    main(opt)
