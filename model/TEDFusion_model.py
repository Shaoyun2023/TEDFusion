"""TEDFusion model with checkpoint-compatible attribute-guided inference.

The model architecture and parameter schema are retained.  ``Text_IF1`` fixes
the released inference path so that its eight image attributes form the
256-dimensional ATGFM guidance and that guidance is injected by TIAM at all
four decoder scales.  ``FeatureWiseAffine`` also reuses its registered
projection instead of constructing a random layer during every forward pass.
No parameter names or shapes are changed, therefore the released
``TEDFusion.pth`` checkpoint remains strictly loadable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
from labelme.utils import distance
from sklearn.manifold import TSNE
from prompt_toolkit.filters import has_arg
from sympy.physics.units import percent
from tensorboard.compat.tensorflow_stub.dtypes import float64


from args_fusion import args
from scipy.spatial.distance import jensenshannon


from sklearn.decomposition import PCA


def hyperbolic_projection(x):
    """极简双曲投影：x -> x / (||x|| + 1) 模拟庞加莱球"""
    return x / (x.norm(dim=-1, keepdim=True) + 1)


def visualize_crossmodal_similarity(img1,img2, text1,text2, title):
    image = torch.sum(img2, dim=0, keepdim=True)
    image = torch.sum(image, dim=1, keepdim=True)

    image = image.squeeze().detach().cpu().numpy()

    image = (image - image.min()) / (image.max() - image.min())
    plt.imshow(image, cmap='jet')
    plt.axis('off')
    plt.show()
    """
    可视化图像局部区域与文本特征的相似度热力图
    img_feat: [2, 384, 12, 12] 图像特征
    text_feat: [2, 384, 1, 1] 文本特征
    title: 图像标题
    """
    hh = img1.shape[2]
    # 将文本特征展平为[2, 384]
    text1 = text1.squeeze(-1).squeeze(-1)  # [2, 384]
    text2 = text2.squeeze(-1).squeeze(-1)  # [2, 384]

    # 计算欧式空间相似度（余弦相似度）
    img1 = img1.flatten(2).permute(0, 2, 1)  # [2, 144, 384]
    img2 = img2.flatten(2).permute(0, 2, 1)  # [2, 144, 384]
    # print(img1.shape,"img1")
    # print(text1.shape,"text1")
    euclidean_sim = F.cosine_similarity(img1, text1.unsqueeze(1), dim=-1)  # [2, 144]

    # 计算双曲空间相似度（负距离）

    c1 = torch.tensor(0.1, device=img2.device, dtype=img2.dtype)
    c1 = torch.clamp_min(c1, 1e-8)

    c2 = torch.tensor(0.1, device=text2.device, dtype=text2.dtype)
    c2 = torch.clamp_min(c2, 1e-8)

    # img2 = safe_project(img2,c1)
    # text2 = safe_project(text2,c2)

    # print(img2,"img")
    # print(text2,"text")
    hyperbolic_sim = F.cosine_similarity(img2, text2.unsqueeze(1),dim=-1)  # [2, 144]

    # 调整形状为[2, 12, 12]
    euclidean_sim = euclidean_sim.view(2, hh,hh)
    hyperbolic_sim = hyperbolic_sim.view(2, hh,hh)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i in range(2):
        # 欧式空间相似度热力图
        axes[0, i].imshow(euclidean_sim[i].detach().cpu().numpy(), cmap='viridis')
        axes[0, i].set_title(f"样本{i} - 欧式空间相似度")
        axes[0, i].axis('off')

        # 双曲空间相似度热力图
        axes[1, i].imshow(hyperbolic_sim[i].detach().cpu().numpy(), cmap='viridis')
        axes[1, i].set_title(f"样本{i} - 双曲空间相似度")
        axes[1, i].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print(f"欧式空间相似度范围: [{euclidean_sim.min():.3f}, {euclidean_sim.max():.3f}]")
    print(f"双曲空间相似度范围: [{hyperbolic_sim.min():.3f}, {hyperbolic_sim.max():.3f}]")
    print(f"双曲空间相似度方差: {hyperbolic_sim.var():.3f} (通常更高，表示更清晰的关联)")


import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def perform_sensitivity_analysis(h_attr, h_text):
    """
    灵敏度分析：计算不同 dropout 概率下，融合特征与原始文本特征的偏离程度
    """
    probs = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    distances = []
    cos_sims = []

    # 转换到同一维度进行对比 (假设对比第一个分支 h_text11)
    # 为了简化，我们观察 mask 逻辑对特征分布的影响
    with torch.no_grad():
        for p in probs:
            # 模拟多次随机实验取平均值
            temp_dist = []
            temp_sim = []
            for _ in range(100):
                mask = (torch.rand(h_text.size(0), 1).to(h_text.device) < p).float()
                # 模拟 forward 中的融合逻辑
                h_fused = mask * h_attr + (1 - mask) * h_text

                # 计算欧式距离 (数值稳定性)
                dist = torch.norm(h_fused - h_text, p=2, dim=1).mean().item()
                # 计算余弦相似度 (语义一致性)
                sim = F.cosine_similarity(h_fused, h_text).mean().item()

                temp_dist.append(dist)
                temp_sim.append(sim)

            distances.append(np.mean(temp_dist))
            cos_sims.append(np.mean(temp_sim))

    # --- 可视化部分 ---
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:red'
    ax1.set_xlabel('Dropout Probability (p)')
    ax1.set_ylabel('L2 Distance (Deviation)', color=color)
    ax1.plot(probs, distances, marker='o', color=color, label='L2 Distance')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Cosine Similarity (Consistency)', color=color)
    ax2.plot(probs, cos_sims, marker='s', color=color, label='Cosine Similarity')
    ax2.tick_params(axis='y', labelcolor=color)

    # 标注我们的设置 0.3
    plt.axvline(x=0.3, color='green', linestyle='--', label='Selected p=0.3')

    plt.title('Sensitivity Analysis of Text Dropout Rate')
    fig.tight_layout()
    plt.show()

    # 返回数值用于日志记录
    return dict(zip(probs.round(2), cos_sims))


def poincare_distance(u, v):
    c = torch.tensor(1.0).to(args.device)
    eps = torch.tensor(1e-5).to(args.device)
    """
    计算庞加莱球中的双曲距离（公式8）
    Args:
        u: [..., D] 输入向量1
        v: [..., D] 输入向量2
    Returns:
        [...,] 双曲距离
    """
    # 稳定化计算（防止除零）
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(eps)
    v_norm = v.norm(dim=-1, p=2, keepdim=True).clamp_min(eps)

    # 计算欧式距离差
    delta = torch.norm(u - v, dim=-1, p=2, keepdim=True)

    # 双曲距离公式
    denominator = (1 - c * u_norm ** 2) * (1 - c * v_norm ** 2)
    return (2 / torch.sqrt(c)) * torch.acosh(
        1 + 2 * c * delta ** 2 / denominator.clamp_min(eps)
    )


def hyperbolic_norm(x):
    c = torch.tensor(1.0).to(args.device)
    eps = torch.tensor(1e-5).to(args.device)
    """计算双曲范数（到原点的距离）"""
    x_norm = x.norm(dim=-1, p=2).clamp_min(eps)
    return (2 / torch.sqrt(c)) * torch.atanh(torch.sqrt(c) * x_norm)

def safe_project(x, c=1.0):
    """Project features to Poincaré ball to avoid NaN."""
    c = torch.clamp_min(c, 1e-8)
    max_norm = (1 - 1e-5) / torch.sqrt(c)  # Slightly below boundary
    norm = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.minimum(max_norm / norm, torch.ones_like(norm))
    return x * scale


def poincare_loss(img_feats, txt_feats):
    # 调整tau值避免权重极端化
    tau = torch.tensor(0.5).to(img_feats.device)  # 从0.1增加到0.5，避免权重极端

    # 检查输入维度
    if len(img_feats.shape) != 3 or len(txt_feats.shape) != 3:
        raise ValueError(f"Expected 3D tensors, got img_feats:{img_feats.shape}, txt_feats:{txt_feats.shape}")

    # 使用相同的曲率参数c，确保特征在同一双曲空间
    c = torch.tensor(1.0, device=img_feats.device, dtype=img_feats.dtype)
    c = torch.clamp_min(c, 1e-4)  # 增加最小值避免数值问题

    # 投影到有效范围
    img_feats = safe_project(img_feats, c)
    txt_feats = safe_project(txt_feats, c)

    # 1. 计算双曲范数差
    img_norms = hyperbolic_norm(img_feats)  # [B,N]
    txt_norms = hyperbolic_norm(txt_feats)  # [B,1]
    norm_diff = torch.abs(img_norms - txt_norms)  # [B,N]

    # 2. 计算自适应权重（添加数值稳定性）
    norm_diff = torch.clamp(norm_diff, min=1e-8, max=5.0)  # 限制范围避免极端值
    weights = torch.exp(-norm_diff / tau)  # [B,N]

    # 检查权重是否正常
    if torch.any(weights.isnan()) or torch.any(weights.isinf()):
        print(f"Warning: Invalid weights detected. min:{weights.min():.6f}, max:{weights.max():.6f}")
        weights = torch.clamp(weights, min=1e-8, max=1.0)

    # 3. 计算双曲距离
    # 扩展文本特征以匹配图像特征维度
    txt_feats_expanded = txt_feats.expand_as(img_feats)
    distances = poincare_distance(img_feats, txt_feats_expanded)  # [B,N,1] 或 [B,N]

    # 如果距离是3D，压缩最后一维
    if distances.dim() == 3 and distances.shape[-1] == 1:
        distances = distances.squeeze(-1)  # [B,N]

    # 检查距离是否正常
    if torch.any(distances.isnan()) or torch.any(distances.isinf()):
        print(f"Warning: Invalid distances detected. min:{distances.min():.6f}, max:{distances.max():.6f}")
        distances = torch.clamp(distances, min=0.0, max=10.0)

    # 4. 加权平均（添加权重归一化）
    weights_sum = weights.sum(dim=1, keepdim=True)  # [B,1]
    weights_sum = torch.clamp(weights_sum, min=1e-8)  # 避免除零
    normalized_weights = weights / weights_sum  # [B,N]

    # 计算加权损失
    loss = (normalized_weights * distances).sum(dim=1).mean()  # 标量

    # 调试输出（训练初期可开启）
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"DEBUG - loss:{loss:.6f}, weights mean:{weights.mean():.6f}, "
              f"distances mean:{distances.mean():.6f}, "
              f"norm_diff mean:{norm_diff.mean():.6f}")
        # 返回一个安全值
        return torch.tensor(0.0, device=img_feats.device, requires_grad=True)

    return loss


##########################################################################
## Feature Modulation
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, c_I, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels * 2, out_channels * (1 + self.use_affine_level))
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.img_proj = None

        self.log_c = nn.Parameter(torch.tensor(math.log(1.0)))

        self.image_proj = nn.Linear(c_I, c_I)
        self.text_proj = nn.Linear(c_I, c_I)
        # self.attn_proj = nn.Linear(c_I, c_I)

        self.eps = 1e-5
        self.c = torch.tensor(1).to(args.device)
        self.ttt = 1

    def mobius_add(self, x, y, c=1.0, eps=1e-5):
        """Möbius 加法: x ⊕_c y"""
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        xy_dot = torch.sum(x * y, dim=-1, keepdim=True)
        numerator = (1 + 2 * c * xy_dot + c * y_norm ** 2) * x + (1 - c * x_norm ** 2) * y
        denominator = 1 + 2 * c * xy_dot + c ** 2 * x_norm ** 2 * y_norm ** 2 + eps
        return numerator / denominator

    def mobius_translation(self, x, t, c=1.0):
        """改进后的Möbius平移（保护值域）"""
        # 1. 记录原始值范围（按通道）
        min_val = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_val = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        # 2. 归一化到双曲空间安全范围（-0.9, 0.9）
        x_normed = (x - min_val) / (max_val - min_val + 1e-7) * 1.8 - 0.9

        # 3. 执行Möbius平移（保持原逻辑）
        neg_t = -t.unsqueeze(2).permute(0, 2, 1)  # [batch, 1, D]

        translated = self.mobius_add(x_normed, neg_t, c=c)

        # 4. 反归一化到原始范围
        translated = (translated + 0.9) / 1.8 * (max_val - min_val) + min_val
        return translated.permute(0, 2, 1).clamp(min_val, max_val)  # 硬截断保护

    def mobius_mul(self, r, x, c=1.0):
        c = torch.tensor(1.0, device=x.device, dtype=x.dtype)
        c = torch.clamp_min(c, 1e-8)

        # Project inputs and clamp r
        x = safe_project(x, c)
        r = torch.clamp(r, -5.0, 5.0)  # Prevent extreme scaling

        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
        if r.dim() > 0:
            r = r.view(*r.shape, *([1] * (x.dim() - r.dim())))

        sqrt_c = torch.sqrt(c)
        atanh_input = torch.clamp(sqrt_c * x_norm, -0.999, 0.999)  # Avoid atanh(1)=inf
        atanh_term = torch.atanh(atanh_input)
        scale = torch.tanh(r * atanh_term) / (sqrt_c * x_norm)

        return scale * x

    def exp_map(self, v):
        """指数映射 exp_0^c(v)"""
        v_norm = torch.norm(v, dim=-1, keepdim=True)  # [..., 1]
        sqrt_c = torch.sqrt(
            torch.as_tensor(self.c, device=v.device, dtype=v.dtype)
        )

        # 计算缩放因子
        scale = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm + self.eps)

        # 归一化处理
        return scale * v

    def log_map(self, y):
        """对数映射 log_0^c(y)"""
        y_norm = torch.norm(y, dim=-1, keepdim=True)  # [..., 1]
        sqrt_c = torch.sqrt(
            torch.as_tensor(self.c, device=y.device, dtype=y.dtype)
        )

        # 计算缩放因子
        atanh_term = torch.atanh(sqrt_c * y_norm.clamp(max=1 - self.eps))  # 防止输入接近1
        scale = atanh_term / (sqrt_c * y_norm + self.eps)

        # 归一化处理
        return scale * y

    def forward(self, x, text_embed):

        # PyTorch image tensors are [B, C, H, W].  The released draft named
        # these dimensions w/h and later reshaped them in the opposite order,
        # which silently transposed every non-square TIAM feature map.
        b_image, c_image, height, width = x.shape

        patch_size = 12
        if x.shape[2] % 12 != 0:  # 判断 a 是否可以被 12 整除
            if x.shape[2] % 10 != 0:
                patch_size = 8
            else:
                patch_size = 10
        else:
            if x.shape[3] % 10 == 0:
                patch_size = 10

        image_patches_E = x

        image_patches = x

        b_I, c_I, np_I, p2_I = image_patches.shape

        image_patches = self.image_proj(image_patches.permute(0, 2, 3, 1))
        image_patches = image_patches.permute(0, 3, 1, 2)


        text_embed = text_embed.unsqueeze(1)
        batch = x.shape[0]

        if self.use_affine_level:
            gamma_E, beta_E = self.MLP(text_embed).view(batch, -1, 1, 1).chunk(2, dim=1)

            b_T, c_T, np_T, p2_T = gamma_E.shape
            # Reuse the trained projection created in __init__.  Recreating it
            # here would silently discard TEDFusion.pth's TIAM weights and make
            # every inference call depend on a new random initialization.
            gamma_H = self.text_proj(gamma_E.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            beta_H = self.text_proj(beta_E.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            b_T, c_T = gamma_H.shape[0], gamma_H.shape[1]
            # gamma_H = self.exp_map(gamma_H.view(b_T, c_T, -1).permute(0, 2, 1))
            # beta_H = self.exp_map(beta_H.view(b_T, c_T, -1).permute(0, 2, 1))

            gamma_H = self.exp_map(gamma_H)
            image_patches = self.exp_map(image_patches)

            gamma_H = gamma_H.squeeze(-1).squeeze(-1)
            image_patches = image_patches.view(b_I, c_I, -1)

            image_patches = self.mobius_translation(image_patches.permute(0, 2, 1), gamma_H, c=self.c)

            image_patches = image_patches.reshape(
                b_I, c_I, height, width
            )

            image_patches = image_patches.permute(0, 2, 3, 1)
            image_patches = image_patches.reshape(
                b_I, height * width, c_I
            )

            # x = self.mobius_add(self.mobius_mul(gamma_H, image_patches, self.c), beta_H, self.c)
            gamma_H = gamma_H.view(b_T, c_T, -1).permute(0, 2, 1)

            distance = poincare_loss(image_patches, gamma_H)

            # 2. 恢复原始维度 [2, 9216, 48] -> [2, 48, 96, 96]
            x = image_patches.reshape(
                b_image, height, width, c_I
            )
            x = x.permute(0, 3, 1, 2)


            x = self.log_map(x)

            gamma_H = gamma_H.permute(0, 2, 1).unsqueeze(3)

        return [x, distance]


def print_euclidean_distances(features):
    """打印特征间的欧式距离矩阵"""
    dist_matrix = torch.cdist(features, features, p=2)
    print("距离矩阵形状:", dist_matrix.shape)
    print("样本0到其他样本的距离:", dist_matrix[0])


def safe_project(x, c=1.0):
    """Project features to Poincaré ball to avoid NaN."""
    c = torch.clamp_min(c, 1e-8)
    max_norm = (1 - 1e-5) / torch.sqrt(c)  # Slightly below boundary
    norm = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.minimum(max_norm / norm, torch.ones_like(norm))
    return x * scale

def custom_attention(image_feature, text_feature,patch_size):

    b, c, w, h = image_feature.shape
    # patch_size = 8

    num_patches = (w // patch_size) * (h // patch_size)
    # print(image_feature.shape)
    # 图像特征的 patch embedding
    image_patches = image_feature.view(b, c, w // patch_size, patch_size, h // patch_size, patch_size)
    image_patches = image_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    image_patches = image_patches.view(b, c, num_patches, patch_size * patch_size )

    return image_patches




def restore_image_feature(image_patches, patch_size,w,h):

    b, c, num_patches, patch_dim = image_patches.shape
    # w = h = int(patch_size * (num_patches ** 0.5))  # 假设宽度和高度可以被 patch_size 整除


    # 将 patches 视图调整为 [batch_size, num_patches, channels, patch_size, patch_size]
    image_patches = image_patches.reshape(b, num_patches, c, patch_size, patch_size)

    # 重新排列维度以匹配原始图像的结构
    image_patches = image_patches.permute(0, 1, 3, 4, 2).contiguous()
    # print(image_patches.shape)

    # 将 patches 展平为一维，准备重新组合成原始图像
    # print(w)
    image_patches = image_patches.view(b, c, w,h)

    # 由于 patches 是以 row-major order 存储的，因此直接 view 为原始图像的形状
    restored_image_feature = image_patches

    return restored_image_feature

    return restored_image_feature

def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out

class PatchFlattener1(nn.Module):
    def __init__(self, patch_size):
        super(PatchFlattener1, self).__init__()
        self.patch_size = patch_size

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Invalid patch size"
        num_patches = (height // self.patch_size) * (width // self.patch_size)
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        # Rearrange patches into columns
        x = x.view(batch_size, channels, patch_height, self.patch_size, patch_width, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(batch_size, num_patches, -1)
        return x

#

class TEDFusion(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(TEDFusion, self).__init__()

        self.model_clip = model_clip
        self.model_clip.eval()

        self.encoder_A = Encoder_A(inp_channels=inp_A_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.encoder_B = Encoder_B(inp_channels=inp_B_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.attention_spatial = Attention_spatial(dim * 2 ** 3)

        self.feature_fusion_4 = Fusion_Embed(embed_dim=dim * 2 ** 3)
        self.prompt_guidance_4 = FeatureWiseAffine(in_channels=256, out_channels=dim * 2 ** 3,c_I=384)
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.feature_fusion_3 = Fusion_Embed(embed_dim = dim * 2 ** 2)
        self.prompt_guidance_3 = FeatureWiseAffine(in_channels=256, out_channels=dim * 2 ** 2,c_I=192)
        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.feature_fusion_2 = Fusion_Embed(embed_dim = dim * 2 ** 1)
        self.prompt_guidance_2 = FeatureWiseAffine(in_channels=256, out_channels=dim * 2 ** 1,c_I=96)
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.feature_fusion_1 = Fusion_Embed(embed_dim = dim)
        self.prompt_guidance_1 = FeatureWiseAffine(in_channels=256, out_channels=dim,c_I=48)
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level1 = nn.Conv2d(432, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # self.decoder_level1 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(96, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=96, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.output1 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output4 = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.conv48to384 = nn.Conv2d(48, 384, kernel_size=3, stride=1, padding=1, bias=bias)

        self.decoder = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=bias)

        self.isss = 0

        # self.fc1 = nn.Linear(8, 16)
        # self.fc2 = nn.Linear(16, 4)

        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

        # self.MLP = nn.Sequential(
        #     nn.Linear(4, 4 * 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(4 * 2, 4)
        # )


        inp_channels = 1
        out_channels = 1
        dim = 64
        num_blocks = [4, 4]
        heads = [8, 8, 8]
        ffn_expansion_factor = 2
        bias = False
        LayerNorm_type = 'WithBias'

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        stride = 1
        output_nc = 1
        # self.conve1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1)
        #
        # self.conve6 = ConvLayer(1, 128, 3, stride)
        # self.convd6 = ConvLayer(256, output_nc, 3, stride)

        self.attr_proj1 = nn.Linear(2, 64)
        self.attr_proj2 = nn.Linear(2, 64)
        self.attr_proj3 = nn.Linear(2, 64)
        self.attr_proj4 = nn.Linear(2, 64)

        self.attr_proj5 = nn.Linear(1, 256)
        # 文本语义映射网络
        self.text_proj = nn.Linear(154, 256)
        # 动态门控融合器
        self.fusion_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
        )

        self.text_proj21 = nn.Linear(512, 32)
        self.text_proj22 = nn.Linear(512, 32)

        self.is_training = 1
        self.drop_prob = 0.3
        # self.drop_prob = 1
        # self.dummy_text = nn.Parameter(torch.randn(1, 64))

    def forward(self, inp_img_A, inp_img_B, text1,text2,feature):
        b = inp_img_A.shape[0]
        # text1 = text1.float()
        # text2 = text2.float()
        # text = text.mean(dim=0)

        # feature = torch.relu(self.fc1(feature))
        # feature = self.fc2(feature)
        # feature = torch.sigmoid(feature)

        feature = feature.unsqueeze(0).expand(text1.size(0), -1)

        f1,f2,f3,f4 = feature.chunk(4, dim=1)



        # print(f1.shape)
        # 属性特征映射
        h_attr1 = self.attr_proj1(f1).expand(b, -1)
        h_attr2 = self.attr_proj2(f2).expand(b, -1)
        h_attr3 = self.attr_proj3(f3).expand(b, -1)
        h_attr4 = self.attr_proj4(f4).expand(b, -1)
        # h_attrir = self.attr_proj1(fir).expand(b, -1)
        # h_attrvi = self.attr_proj1(fvi).expand(b, -1)

        # fuse_all = torch.cat([h_attr1, h_attr2, h_attr3, h_attr4], dim=0)
        # heatmap = fuse_all.cpu().detach().numpy()
        # print(feature.shape)
        # # 绘制热力图
        # plt.imshow(heatmap, cmap='viridis')
        # plt.colorbar()
        # plt.title('Heatmap of [1, 256] Tensor')
        # plt.show()

        # 文本特征映射
        # h_text = self.text_proj(text)


        h_text1 = self.get_text_feature(text1.expand(b, -1)).to(inp_img_A.dtype)
        h_text2 = self.get_text_feature(text2.expand(b, -1)).to(inp_img_A.dtype)

        # h_text1 = self.text_proj21(h_text1)
        # h_text2 = self.text_proj22(h_text2)
        h_text1 = self.text_proj21(h_text1)
        h_text2 = self.text_proj22(h_text2)
        h_text = torch.cat([h_text1,h_text2],dim=1)
        # h_text = self.dummy_text.expand(b, -1)
        h_attr = torch.cat([h_attr1, h_attr2], dim=1)


        if self.is_training:
            # 生成伯努利掩码 [B,1]
            mask = (torch.rand_like(h_text[:, :1]) < self.drop_prob).float()

            h_text11 = mask * h_attr1 + (1 - mask) * h_text  # [B, text_dim]
            h_text22 = mask * h_attr2 + (1 - mask) * h_text  # [B, text_dim]
            h_text33 = mask * h_attr3 + (1 - mask) * h_text  # [B, text_dim]
            h_text44 = mask * h_attr4 + (1 - mask) * h_text  # [B, text_dim]
        else:
            # 测试时完全用属性特征替代文本
            h_text11 = h_attr1
            h_text22 = h_attr2
            h_text33 = h_attr3
            h_text44 = h_attr4

        # 动态门控融合
        gate1 = self.fusion_gate(torch.cat([h_attr1, h_text11], dim=1))  # [B,128]

        # 残差融合
        fused1 = h_attr1 * gate1 + h_text * (1 - gate1)


        # 动态门控融合
        gate2 = self.fusion_gate(torch.cat([h_attr2, h_text22], dim=1))  # [B,128]
        # 残差融合
        fused2 = h_attr2 * gate2 + h_text * (1 - gate2)

        # 动态门控融合
        gate3 = self.fusion_gate(torch.cat([h_attr3, h_text33], dim=1))  # [B,128]
        # 残差融合
        fused3 = h_attr3 * gate3 + h_text * (1 - gate3)

        # 动态门控融合
        gate4 = self.fusion_gate(torch.cat([h_attr4, h_text44], dim=1))  # [B,128]
        # 残差融合
        fused4 = h_attr4 * gate4 + h_text * (1 - gate4)


        fused = torch.cat([fused1,fused2,fused3,fused4],dim=1)
        fusedd = fused1 + fused2 + fused3 + fused4

        self.isss = self.isss + 1
        h_texttt = torch.cat([h_text, h_text,h_text,h_text], dim=1)

        x4_A, x3_A, x2_A, x1_A = self.encoder_A(inp_img_A) #4:[384,12,12] 3:[192,24,24] 2:[96,48,48] 1:[48,96,96]
        x4_B, x3_B, x2_B, x1_B = self.encoder_B(inp_img_B)

        xxx = self.prompt_guidance_1(x1_A, fused)[0]

        x4_A = self.attention_spatial(x4_A)
        x4_B = self.attention_spatial(x4_B)



        x4_A, x4_B = self.cross_attention(x4_A, x4_B)

        x4 = self.feature_fusion_4(x4_A, x4_B)
        # x4 = self.attention_spatial(x4)

        x4 = self.prompt_guidance_4(x4, fused)[0]
        inp4 = x4
        outd4 = self.decoder_level4(inp4)

        inp3 = self.up4_3(outd4)
        inp3 = self.prompt_guidance_3(inp3, fused)[0]
        x3 = self.feature_fusion_3(x3_A, x3_B)
        inp31 = torch.cat([inp3, x3], 1)
        inp31 = self.reduce_chan_level3(inp31)
        outd31 = self.decoder_level3(inp31)

        inp2 = self.up3_2(outd31)
        inp2 = self.prompt_guidance_2(inp2, fused)[0]
        x2 = self.feature_fusion_2(x2_A, x2_B)
        inp21 = torch.cat([x2, self.up3_2(x3)], 1)
        inp22 = torch.cat([inp21,x2,inp2], 1)
        inp22 = self.reduce_chan_level2(inp22)
        outd22 = self.decoder_level2(inp22)

        inp1 = self.up2_1(outd22)
        inp1 = self.prompt_guidance_1(inp1, fused)[0]
        x1 = self.feature_fusion_1(x1_A, x1_B)
        inp11 = torch.cat([x1, self.up2_1(x2)], 1)

        inp12 = torch.cat([x1, inp11, self.up2_1_2(inp21)], 1)
        inp13 = torch.cat([x1, inp11, inp12, inp1], 1)

        inp13 = self.reduce_chan_level1(inp13)

        outd1 = self.decoder_level1(inp13)

        outd1 = self.refinement(outd1)



        outd1 = self.output1(outd1)
        outd1 = self.output2(outd1)
        outd1 = self.output3(outd1)
        outd1 = self.output4(outd1)

        distance1 = self.prompt_guidance_4(x4, h_texttt)[1]
        distance2 = self.prompt_guidance_3(inp3, h_texttt)[1]
        distance3 = self.prompt_guidance_2(inp2, h_texttt)[1]
        distance4 = self.prompt_guidance_1(inp1, h_texttt)[1]
        distance_all = (distance1 + distance2 + distance3 + distance4)/4.0
        # distance_all = torch.tensor(0)

        return [outd1,h_text,fusedd,distance_all]

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature

class TEDFusion1(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2, 2, 2, 2],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(TEDFusion1, self).__init__()
        # self.dummy_text = nn.Parameter(torch.randn(1, 64))
        self.model_clip = model_clip
        self.model_clip.eval()

        self.encoder_A = Encoder_A(inp_channels=inp_A_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.encoder_B = Encoder_B(inp_channels=inp_B_channels, dim=dim, num_blocks=num_blocks, heads=heads,
                                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        self.cross_attention = Cross_attention(dim * 2 ** 3)
        self.attention_spatial = Attention_spatial(dim * 2 ** 3)

        self.feature_fusion_4 = Fusion_Embed(embed_dim=dim * 2 ** 3)
        self.prompt_guidance_4 = FeatureWiseAffine(in_channels=256, out_channels=dim * 2 ** 3,c_I=384)
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.feature_fusion_3 = Fusion_Embed(embed_dim = dim * 2 ** 2)
        self.prompt_guidance_3 = FeatureWiseAffine(in_channels=256, out_channels=dim * 2 ** 2,c_I=192)
        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.feature_fusion_2 = Fusion_Embed(embed_dim = dim * 2 ** 1)
        self.prompt_guidance_2 = FeatureWiseAffine(in_channels=256, out_channels=dim * 2 ** 1,c_I=96)
        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.feature_fusion_1 = Fusion_Embed(embed_dim = dim)
        self.prompt_guidance_1 = FeatureWiseAffine(in_channels=256, out_channels=dim,c_I=48)
        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.up2_1_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level1 = nn.Conv2d(432, int(dim * 2 ** 1), kernel_size=1, bias=bias)
        # self.decoder_level1 = nn.Sequential(*[
        #     TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
        #                      bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(96, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=96, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.output1 = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output4 = nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.decoder = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=bias)

        self.isss = 0

        # self.fc1 = nn.Linear(8, 16)
        # self.fc2 = nn.Linear(16, 4)

        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

        # self.MLP = nn.Sequential(
        #     nn.Linear(4, 4 * 2),
        #     nn.LeakyReLU(),
        #     nn.Linear(4 * 2, 4)
        # )


        inp_channels = 1
        out_channels = 1
        dim = 64
        num_blocks = [4, 4]
        heads = [8, 8, 8]
        ffn_expansion_factor = 2
        bias = False
        LayerNorm_type = 'WithBias'

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        stride = 1
        output_nc = 1
        # self.conve1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1)
        #
        # self.conve6 = ConvLayer(1, 128, 3, stride)
        # self.convd6 = ConvLayer(256, output_nc, 3, stride)

        self.attr_proj1 = nn.Linear(2, 64)
        self.attr_proj2 = nn.Linear(2, 64)
        self.attr_proj3 = nn.Linear(2, 64)
        self.attr_proj4 = nn.Linear(2, 64)
        self.attr_proj5 = nn.Linear(1, 256)

        # 文本语义映射网络
        self.text_proj = nn.Linear(154, 256)
        # 动态门控融合器
        self.fusion_gate = nn.Sequential(
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
        # self.fusion_gate = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.Sigmoid()
        # )

        self.text_proj21 = nn.Linear(512, 32)
        self.text_proj22 = nn.Linear(512, 32)
        # self.text_proj21 = nn.Linear(512, 128)
        # self.text_proj22 = nn.Linear(512, 128)
        self.is_training = 0
        self.temp_storage = []
        self.all_samples = []
        self.sample_names = []

    def forward(self, inp_img_A, inp_img_B, feature):
        """Fuse a source-image pair using image attributes instead of text.

        Args:
            inp_img_A: First source image, shaped ``[B, 3, H, W]``.
            inp_img_B: Second source image, shaped ``[B, 3, H, W]``.
            feature: Eight attributes in the same order as the released test
                script, shaped ``[8]``, ``[1, 8]`` or ``[B, 8]``.
        """
        if inp_img_A.ndim != 4 or inp_img_B.ndim != 4:
            raise ValueError("inp_img_A and inp_img_B must both be 4-D tensors")
        if inp_img_A.shape != inp_img_B.shape:
            raise ValueError(
                "The two source images must have identical shapes, got "
                f"{tuple(inp_img_A.shape)} and {tuple(inp_img_B.shape)}"
            )

        batch = inp_img_A.shape[0]
        feature = torch.as_tensor(
            feature, device=inp_img_A.device, dtype=inp_img_A.dtype
        )
        if feature.ndim == 1:
            feature = feature.unsqueeze(0)
        elif feature.ndim != 2:
            raise ValueError(
                "feature must have shape [8], [1, 8] or [B, 8], got "
                f"{tuple(feature.shape)}"
            )
        if feature.shape[1] != 8:
            raise ValueError(
                f"feature must contain exactly 8 attributes, got {feature.shape[1]}"
            )
        if feature.shape[0] == 1 and batch != 1:
            feature = feature.expand(batch, -1)
        elif feature.shape[0] != batch:
            raise ValueError(
                f"feature batch size {feature.shape[0]} does not match image "
                f"batch size {batch}"
            )

        # ATGFM: preserve the checkpoint's training-time order and split the
        # eight values into four consecutive 2-D groups:
        # [A saturation, A brightness], [A texture, A contrast],
        # [B saturation, B brightness], [B texture, B contrast].
        # Each group is projected to 64 dimensions.  At inference text is
        # unavailable, so the attribute embedding substitutes for the text
        # branch; concatenation then gives the 256-D TIAM guidance.
        attribute_pairs = feature.chunk(4, dim=1)
        attribute_projectors = (
            self.attr_proj1,
            self.attr_proj2,
            self.attr_proj3,
            self.attr_proj4,
        )
        fused_attributes = []
        for attribute_pair, projector in zip(
                attribute_pairs, attribute_projectors):
            h_attr = projector(attribute_pair)
            h_text_substitute = h_attr
            gate = self.fusion_gate(
                torch.cat([h_attr, h_text_substitute], dim=1)
            )
            # This intentionally follows the published ATGFM equation.  With
            # text replaced by the same attribute embedding, the expression
            # is algebraically h_attr; changing it to gate * h_attr would feed
            # an untrained distribution into the fixed checkpoint.
            h_fused = (
                gate * h_attr + (1.0 - gate) * h_text_substitute
            )
            fused_attributes.append(h_fused)

        fused = torch.cat(fused_attributes, dim=1)

        x4_A, x3_A, x2_A, x1_A = self.encoder_A(inp_img_A)
        x4_B, x3_B, x2_B, x1_B = self.encoder_B(inp_img_B)

        x4_A = self.attention_spatial(x4_A)
        x4_B = self.attention_spatial(x4_B)
        x4_A, x4_B = self.cross_attention(x4_A, x4_B)

        # TIAM level 4: inject attribute guidance before the deepest decoder.
        x4 = self.feature_fusion_4(x4_A, x4_B)
        x4 = self.prompt_guidance_4(x4, fused)[0]
        outd4 = self.decoder_level4(x4)

        # TIAM level 3.
        inp3 = self.up4_3(outd4)
        inp3 = self.prompt_guidance_3(inp3, fused)[0]
        x3 = self.feature_fusion_3(x3_A, x3_B)
        inp31 = self.reduce_chan_level3(torch.cat([inp3, x3], dim=1))
        outd31 = self.decoder_level3(inp31)

        # TIAM level 2.
        inp2 = self.up3_2(outd31)
        inp2 = self.prompt_guidance_2(inp2, fused)[0]
        x2 = self.feature_fusion_2(x2_A, x2_B)
        inp21 = torch.cat([x2, self.up3_2(x3)], dim=1)
        inp22 = torch.cat([inp21, x2, inp2], dim=1)
        inp22 = self.reduce_chan_level2(inp22)
        outd22 = self.decoder_level2(inp22)

        # TIAM level 1 and the unchanged dense reconstruction head.
        inp1 = self.up2_1(outd22)
        inp1 = self.prompt_guidance_1(inp1, fused)[0]
        x1 = self.feature_fusion_1(x1_A, x1_B)
        inp11 = torch.cat([x1, self.up2_1(x2)], dim=1)
        inp12 = torch.cat([x1, inp11, self.up2_1_2(inp21)], dim=1)
        inp13 = torch.cat([x1, inp11, inp12, inp1], dim=1)
        inp13 = self.reduce_chan_level1(inp13)

        outd1 = self.decoder_level1(inp13)
        outd1 = self.refinement(outd1)
        outd1 = self.output1(outd1)
        outd1 = self.output2(outd1)
        outd1 = self.output3(outd1)
        outd1 = self.output4(outd1)
        return outd1

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature

class Cross_attention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()
        self.n_head = n_head
        self.norm_A = nn.GroupNorm(norm_groups, in_channel)
        self.norm_B = nn.GroupNorm(norm_groups, in_channel)
        self.qkv_A = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_A = nn.Conv2d(in_channel, in_channel, 1)

        self.qkv_B = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out_B = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x_A, x_B):
        batch, channel, height, width = x_A.shape

        n_head = self.n_head
        head_dim = channel // n_head

        x_A = self.norm_A(x_A)
        qkv_A = self.qkv_A(x_A).view(batch, n_head, head_dim * 3, height, width)
        query_A, key_A, value_A = qkv_A.chunk(3, dim=2)

        x_B = self.norm_B(x_B)
        qkv_B = self.qkv_B(x_B).view(batch, n_head, head_dim * 3, height, width)
        query_B, key_B, value_B = qkv_B.chunk(3, dim=2)

        attn_A = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_B, key_A
        ).contiguous() / math.sqrt(channel)
        attn_A = attn_A.view(batch, n_head, height, width, -1)
        attn_A = torch.softmax(attn_A, -1)
        attn_A = attn_A.view(batch, n_head, height, width, height, width)

        out_A = torch.einsum("bnhwyx, bncyx -> bnchw", attn_A, value_A).contiguous()
        out_A = self.out_A(out_A.view(batch, channel, height, width))
        out_A = out_A + x_A

        attn_B = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query_A, key_B
        ).contiguous() / math.sqrt(channel)
        attn_B = attn_B.view(batch, n_head, height, width, -1)
        attn_B = torch.softmax(attn_B, -1)
        attn_B = attn_B.view(batch, n_head, height, width, height, width)

        out_B = torch.einsum("bnhwyx, bncyx -> bnchw", attn_B, value_B).contiguous()
        out_B = self.out_B(out_B.view(batch, channel, height, width))
        out_B = out_B + x_B

        return out_A, out_B

class Attention_spatial(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=16):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head
        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)
        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input











class Encoder_A(nn.Module):
    def __init__(self, inp_channels=3, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_A, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img_A):
        inp_enc_level1_A = self.patch_embed(inp_img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A)

        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A)

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A)

        inp_enc_level4_A = self.down3_4(out_enc_level3_A)
        out_enc_level4_A = self.encoder_level4(inp_enc_level4_A)

        return out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A


class Encoder_B(nn.Module):
    def __init__(self, inp_channels=1, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Encoder_B, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img_B):
        inp_enc_level1_B = self.patch_embed(inp_img_B)
        out_enc_level1_B = self.encoder_level1(inp_enc_level1_B)

        inp_enc_level2_B = self.down1_2(out_enc_level1_B)
        out_enc_level2_B = self.encoder_level2(inp_enc_level2_B)

        inp_enc_level3_B = self.down2_3(out_enc_level2_B)
        out_enc_level3_B = self.encoder_level3(inp_enc_level3_B)

        inp_enc_level4_B = self.down3_4(out_enc_level3_B)
        out_enc_level4_B = self.encoder_level4(inp_enc_level4_B)

        return out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super(Fusion_Embed, self).__init__()

        self.fusion_proj = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x_A, x_B):
        x = torch.concat([x_A, x_B], dim=1)
        x = self.fusion_proj(x)
        return x

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
