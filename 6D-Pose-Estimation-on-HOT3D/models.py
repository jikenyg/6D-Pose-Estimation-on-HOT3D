import utils
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as model_zoo
import config as cfg
from functools import partial
from spatial_correlation_sampler import SpatialCorrelationSampler



class EMA(nn.Module):
    def __init__(self, channels, factor=32): 
        super(EMA, self).__init__()
        self.groups = factor
        assert channels % self.groups == 0, \
            f"EMA Error: Input channels ({channels}) must be divisible by groups ({self.groups})"
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) 
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class ConvGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, norm_groups=32):
        super(ConvGnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=False)
        self.norm = nn.GroupNorm(norm_groups, out_channels)
        self.relu = nn.SiLU(inplace=True)

    def forward(self, input):
        x = input
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class CondConv(nn.Module):
    def __init__(self, n_channels, n_filters, kernel_size,
                 stride=1, padding=None, norm=False,
                 n_class=1, activation='swish'):
        super(CondConv, self).__init__()
        self.weights = nn.Parameter(torch.empty(
            n_class, n_filters, n_channels, kernel_size, kernel_size))
        if norm:
            self.norm = nn.GroupNorm(num_groups=32, num_channels=n_filters)
            self.bias = None
        else:
            self.norm = None
            self.bias = nn.Parameter(torch.empty(
                n_class, n_filters, 1, 1))
        self.stride = stride
        self.kernel_size = kernel_size
        self.activation = activation
        self.padding = padding if padding is not None else kernel_size // 2

        for i, weight in enumerate(self.weights):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            if fan_in != 0 and self.bias is not None:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input, decoder_idx):
        N, Cin, Hi, Wi = input.shape
        _, Cout, Cin, Hk, Wk = self.weights.shape
        x = torch.reshape(input, (1, N*Cin, Hi, Wi))
        weight = torch.reshape(
            self.weights[decoder_idx], (N*Cout, Cin, Hk, Wk))
        x = F.conv2d(x, weight, bias=None, stride=self.stride,
                     padding=self.padding, groups=N)
        x = torch.reshape(x, (N, Cout, *x.shape[2:]))
        if self.bias is not None:
            bias = self.bias[decoder_idx]
            x = x + bias
        if self.norm is not None:
            x = self.norm(x)
        if self.activation == 'relu':
            x = F.relu(x, inplace=True)
        elif self.activation == 'swish':
            x = F.silu(x, inplace=True)
        return x


class ResNet34_AsymUNet(nn.Module):
    def __init__(self, out_feat_dim=64, rgb_input_dim=3,
                 n_decoders=1, track_running_stats=False):
        super().__init__()
        self.out_feat_dim = out_feat_dim
        norm_layer = partial(
            nn.BatchNorm2d, track_running_stats=track_running_stats)
        model = model_zoo.resnet34(norm_layer=norm_layer)
        # 修正：更新加载模型权重的方式，使用get_state_dict_from_url以避免警告
        # 且确保在config中定义MODEL_URL或直接使用硬编码URL
        try:
            ResNet34_Weights_IMAGENET1K_V1 = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet34-b627a593.pth', map_location='cpu')
        except RuntimeError:
            print("Failed to download ResNet34 weights. Please check network connection.")
            # 如果下载失败，可以尝试在本地加载或者抛出错误
            raise
            
        model.load_state_dict(ResNet34_Weights_IMAGENET1K_V1, strict=False)
        
        self.base_model = torch.nn.Sequential(*(list(model.children())[:-2]))

        if rgb_input_dim != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(rgb_input_dim, 64, kernel_size=7,
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64, track_running_stats=False),
                nn.SiLU(inplace=True))
        else:
            self.layer0 = nn.Sequential(
                *list(self.base_model.children())[0:3])

        self.layer1 = nn.Sequential(*list(self.base_model.children())[4:5])
        self.layer2 = nn.Sequential(*list(self.base_model.children())[5:6])
        self.layer3 = nn.Sequential(*list(self.base_model.children())[6:7])
        self.layer4 = nn.Sequential(*list(self.base_model.children())[7:8])
        self.pool = nn.Sequential(
            AtrousSpatialPyramidPooling(
                n_features=512, dilation_rates=[2, 3, 4, 6, 8, 12]),
            nn.Conv2d(512 * 8, 512, kernel_size=1, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.SiLU(inplace=True))

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        
        self.decoder = nn.ModuleDict(dict(
            layer2_1x1=CondConv(128, 128, kernel_size=1, stride=1, n_class=n_decoders),
            layer3_1x1=CondConv(256, 256, kernel_size=1, stride=1, n_class=n_decoders),
            layer4_1x1=CondConv(512, 512, kernel_size=1, stride=1, n_class=n_decoders),
            conv_up3=CondConv(256 + 512, 512, kernel_size=3, stride=1, padding=1, n_class=n_decoders),
            conv_up2=CondConv(128 + 512, 256, kernel_size=3, stride=1, padding=1, n_class=n_decoders),
            conv_last=CondConv(256, self.out_feat_dim + 1, kernel_size=1, stride=1, activation=None, n_class=n_decoders)))
        
       
        self.rgb_emb_ema = EMA(channels=self.out_feat_dim)

    def forward(self, input, decoder_idx):
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        stem_x = self.pool(layer4)
        stem_x = self.decoder['layer4_1x1'](stem_x, decoder_idx)
        stem_x = self.upsample(stem_x)
        layer_slice = layer3        
        layer_projection = self.decoder['layer3_1x1'](layer_slice, decoder_idx)
        stem_x = torch.cat([stem_x, layer_projection], dim=1)
        stem_x = self.decoder['conv_up3'](stem_x, decoder_idx)
        stem_x = self.upsample(stem_x)

        layer_slice = layer2
        layer_projection = self.decoder['layer2_1x1'](layer_slice, decoder_idx)
        stem_x = torch.cat([stem_x, layer_projection], dim=1)
        stem_x = self.decoder['conv_up2'](stem_x, decoder_idx)
        x_out = self.decoder['conv_last'](stem_x, decoder_idx)

        rgb_emb = x_out[:, :-1]         # BxCx64x64
        visib_msk = x_out[:, -1:]       # Bx1x64x64

        # 最小改动：对 rgb_emb 经过 EMA 增强，visib_msk 不变
        rgb_emb = self.rgb_emb_ema(rgb_emb)

        return rgb_emb, visib_msk


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, n_features=128, dilation_rates=[2, 3, 4, 5, 6, 7]):
        super(AtrousSpatialPyramidPooling, self).__init__()

        self.convs = nn.ModuleList([
            ConvGnReLU(n_features, n_features, kernel_size=3,
                       stride=1, dilation=rate, padding=rate)
            for rate in dilation_rates])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        x = inputs
        x_pool = self.global_pool(x)
        x_pool = x_pool.repeat(1, 1, *x.shape[-2:])
        x_out = torch.cat(
            [x, x_pool] + [conv(x) for conv in self.convs], dim=1)
        return x_out


class Correlation(nn.Module):
    def __init__(self, patch_size, input_dim=128, feature_dim=128):
        super(Correlation, self).__init__()
        self.proj_in = nn.Conv2d(input_dim, feature_dim, kernel_size=1,
                                 stride=1, padding=0)
        self.corr = SpatialCorrelationSampler(patch_size=patch_size)
        self.feature_dim = feature_dim

    def forward(self, input1, input2):
        x1 = self.proj_in(input1)
        x2 = self.proj_in(input2)
        output = self.corr(x1, x2)
        output = output.view(
            len(input1), -1, *input1.shape[-2:])
        output = output / math.sqrt(self.feature_dim)

        return output


class PoseDecoder(nn.Module):
    def __init__(self):
        super(PoseDecoder, self).__init__()
        self.conv_down_x8 = ConvGnReLU(
            64 + 1, 128, kernel_size=3, stride=2, padding=1)
        self.conv_down_x16 = ConvGnReLU(
            128, 128, kernel_size=3, stride=2, padding=1)

        self.clf_pool = AtrousSpatialPyramidPooling(
            n_features=128, dilation_rates=[2, 3, 4, 6, 8, 12])

        self.corr_x4 = Correlation(
            patch_size=11, input_dim=64 + 1, feature_dim=64)
        self.reg_conv_x8 = ConvGnReLU(
            64 + 1 + 121, 128, kernel_size=3, stride=2, padding=1)
        self.corr_x8 = Correlation(
            patch_size=11, input_dim=128, feature_dim=128)
        self.reg_conv_x16 = ConvGnReLU(
            128 + 121 + 128, 256, kernel_size=3, stride=2, padding=1)
        self.corr_x16 = Correlation(
            patch_size=11, input_dim=128, feature_dim=128)
        self.reg_conv_add = ConvGnReLU(
            128 + 121 + 256, 256, kernel_size=3, stride=1, padding=1)

        self.downsample = nn.AvgPool2d(2)
        self.reg_pool = AtrousSpatialPyramidPooling(
            n_features=256, dilation_rates=[2, 3, 4, 6, 8, 12])

    def forward(self, x_real, x_synt=None):
        if x_synt is None:
            x_real_x4 = x_real
            x_real_x8 = self.conv_down_x8(x_real_x4)
            x_real_x16 = self.conv_down_x16(x_real_x8)
            x_clf_pool = self.clf_pool(x_real_x16)
            predictions = {
                'real_x4': x_real_x4,
                'real_x8': x_real_x8,
                'real_x16': x_real_x16,
                'clf_pool': x_clf_pool}
            return predictions
        else:
            x_real_x4 = x_real['real_x4']
            x_real_x8 = x_real['real_x8']
            x_real_x16 = x_real['real_x16']
            x_synt_x4 = x_synt

            x_corr_x4 = self.corr_x4(x_synt_x4, x_real_x4)
            x_reg_x8 = self.reg_conv_x8(
                torch.concat([x_real_x4, x_corr_x4], dim=1))

            x_synt_x8 = self.conv_down_x8(x_synt_x4)
            x_corr_x8 = self.corr_x8(x_synt_x8, x_real_x8)

            x_reg_x16 = self.reg_conv_x16(
                torch.concat([x_real_x8, x_corr_x8, x_reg_x8], dim=1))
            x_synt_x16 = self.conv_down_x16(x_synt_x8)
            x_corr_x16 = self.corr_x16(x_synt_x16, x_real_x16)
            x_reg_add = self.reg_conv_add(
                torch.concat([x_real_x16, x_corr_x16, x_reg_x16], dim=1))
            x_reg_pool = self.reg_pool(x_reg_add)
            predictions = {
                'synt_x4': x_synt_x4,
                'synt_x8': x_synt_x8,
                'reg_x8': x_reg_x8,
                'synt_x16': x_synt_x16,
                'reg_x16': x_reg_x16,
                'reg_pool': x_reg_pool}
            return predictions


class TaskHead(nn.Module):
    def __init__(self, n_features, n_class, hidden_size=256):
        super(TaskHead, self).__init__()
        self.conv_in = ConvGnReLU(
            n_features, n_features // 8, kernel_size=3, stride=2, padding=1)
        self.flat = nn.Flatten(1)

    
        
        # 计算 conv_in 后的特征图尺寸
        feature_map_size_after_conv_in = cfg.INPUT_IMG_SIZE // 16 // 2 # 例如 256 // 16 // 2 = 8
        
        LINEAR1_INPUT_DIM_CALCULATED = (n_features // 8) * feature_map_size_after_conv_in * feature_map_size_after_conv_in

        self.linear1 = nn.Linear(LINEAR1_INPUT_DIM_CALCULATED + (3), hidden_size)

        self.relu = nn.SiLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, n_class)

    def forward(self, inputs, aux_feats):
        x = self.conv_in(inputs)
        x = self.flat(x)
        # 假设 aux_feats 可能是 Bx3x1x1，所以需要 squeeze 到 Bx3
        x = torch.concat([x, aux_feats.squeeze(-1).squeeze(-1)], dim=1) 

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, n_pose_bins=300, n_depth_bins=500):
        super(ClassificationHead, self).__init__()
        self.n_depth_bins = n_depth_bins
        # 注意：TaskHead 的 `n_features` 参数应该是其 `inputs` 的通道数
        self.pose_clf = TaskHead(128 * 8, n_pose_bins) # PoseDecoder['clf_pool']的通道数是 128*8
        self.depth_clf = TaskHead(128 * 8, n_depth_bins)
        self.trans_clf = TaskHead(128 * 8, cfg.CELL_SIZE**2)

    def forward(self, features):
        pose_logits = self.pose_clf(features['clf_pool'], features['fov'])
        depth_logits = self.depth_clf(features['clf_pool'], features['fov'])
        trans_logits = self.trans_clf(features['clf_pool'], features['fov'])
        return pose_logits, depth_logits, trans_logits


class RegressionHead(nn.Module):
    def __init__(self, dim_pose=4, dim_trans=3):
        super(RegressionHead, self).__init__()
        # 注意：TaskHead 的 `n_features` 参数应该是其 `inputs` 的通道数
        self.pose_reg = TaskHead(256 * 8, dim_pose) # PoseDecoder['reg_pool']的通道数是 256*8
        self.depth_reg = TaskHead(256 * 8, 1)
        self.trans_reg = TaskHead(256 * 8, 2)

        nn.init.xavier_uniform_(self.depth_reg.linear2.weight,
                                gain=1.0 / cfg.Tz_BINS_NUM) # 假设 cfg.Tz_BINS_NUM 存在
        nn.init.zeros_(self.depth_reg.linear2.bias)
        nn.init.xavier_uniform_(self.trans_reg.linear2.weight,
                                gain=1.0 / cfg.CELL_SIZE) # 假设 cfg.CELL_SIZE 存在
        nn.init.zeros_(self.trans_reg.linear2.bias)

    def forward(self, features):
        pose_res = self.pose_reg(features['reg_pool'], features['fov'])
        depth_res = self.depth_reg(features['reg_pool'], features['fov'])
        trans_res = self.trans_reg(features['reg_pool'], features['fov'])
        return pose_res, depth_res, trans_res


class MRCNet(nn.Module):
    def __init__(self, dataset='tless', n_decoders=1, depth_min=0.05,
                 depth_max=2.0, n_depth_bin=500):
        super(MRCNet, self).__init__()
        self.backbone = ResNet34_AsymUNet(
            n_decoders=n_decoders, track_running_stats=False, rgb_input_dim=4)
        self.decoder = PoseDecoder()
        self.classifier = ClassificationHead(
            n_pose_bins=cfg.N_POSE_BIN, n_depth_bins=n_depth_bin) # 假设 cfg.N_POSE_BIN 存在
        self.regressor = RegressionHead(
            dim_pose=6 if cfg.USE_6D else 4) # 假设 cfg.USE_6D 存在
        self.renderer = utils.SyntheticRenderer(
            cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE, dataset=dataset)

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.n_depth_bin = n_depth_bin

    def _make_predictions(self, mask_real, mask_synt, pose_logits,
                          depth_logits, trans_logits, pose_res,
                          depth_res, trans_res):
        depth = utils.depth_from_bin_logits(
            depth_logits, depth_res, self.depth_min, self.depth_max,
            self.n_depth_bin)
        trans_xy = utils.trans_from_bins(trans_logits, trans_res)
        trans_persp = torch.concat([trans_xy, depth], dim=-1)
        if cfg.USE_6D: # 假设 cfg.USE_6D 存在
            rot_ego = utils.rotation_from_prototypes(pose_logits, pose_res)
        else:
            quaternions = utils.quaternion_from_prototypes(
                pose_logits, pose_res)
            rot_ego = utils.quaternion_to_rotation(quaternions)
        predictions = {'roi_mask': mask_real,
                       'roi_mask_synt': mask_synt,
                       'roi_obj_R': rot_ego,
                       'quat_bin': pose_logits,
                       'quat_res': pose_res,
                       'depth_bin': depth_logits,
                       'depth_res': depth_res,
                       'trans_xy': trans_xy,
                       'trans_logits': trans_logits,
                       'trans_res': trans_res,
                       'translation': trans_persp}
        return predictions

    def forward(self, inputs, aux, targets=None):
        # MRCNet 的前向逻辑现在更简洁，因为 EMA 处理已在 backbone 内部完成
        # x1 已经是经过 EMA 增强的 rgb_emb
        x1, mask_real_raw = self.backbone(inputs, aux['obj_cls'])
        x_real_in = torch.concat([x1, mask_real_raw.detach().sigmoid()], dim=1)
        
        x_clf = self.decoder(x_real_in)
        x_clf['fov'] = aux['fov']
        pose_logits, depth_logits, trans_logits = self.classifier(x_clf)

        with torch.no_grad():
            depth_id = torch.argmax(depth_logits, dim=-1)
            depth_bins = utils.make_depth_bins(
                self.depth_min, self.depth_max, self.n_depth_bin,
                device=depth_logits.device)
            depth_bins = torch.repeat_interleave(
                depth_bins, len(depth_id), dim=0)
            depth = torch.gather(
                depth_bins, dim=-1, index=depth_id.unsqueeze(-1))

            zero_res = torch.zeros(len(trans_logits), 2).to(trans_logits)
            trans_xy = utils.trans_from_bins(trans_logits, zero_res)
            trans_persp = torch.cat([trans_xy, depth], dim=-1)
            image_size = (cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE) # 假设 cfg.INPUT_IMG_SIZE 存在
            trans_3d = utils.perspective_to_trans_3d(
                trans_persp, image_size, aux['intrinsics'])

            pose_id = torch.argmax(pose_logits, dim=-1)
            quat_init = utils.quaternion_from_bin_logits(F.one_hot(
                pose_id, num_classes=cfg.N_POSE_BIN)) # 假设 cfg.N_POSE_BIN 存在
            rot_init = utils.quaternion_to_rotation(quat_init)

            render, render_mask, render_map = self.renderer(
                aux['obj_cls'], rot_init, trans_3d, aux['intrinsics'])

        # x2 同理，已经是经过 EMA 增强的 rgb_emb
        sync_inputs = torch.concat([render, render_map], dim=1)
        x2, mask_synt_raw = self.backbone(sync_inputs, aux['obj_cls'])
        x_synt_in = torch.concat([x2, mask_synt_raw.detach().sigmoid()], dim=1)
        
        x_reg = self.decoder(x_clf, x_synt_in)
        predictions = {'render': render,
                       'mask_synt': mask_synt_raw}
        x_reg['fov'] = torch.concat([
            trans_3d[..., :2] / trans_3d[..., -1:],
            aux['fov'][..., 2:]], dim=1)
        pose_res, depth_res, trans_res = self.regressor(x_reg)
        predictions.update(self._make_predictions(
            mask_real_raw, mask_synt_raw, pose_logits, depth_logits,
            trans_logits, pose_res, depth_res, trans_res))

        if targets:
            targets['quat_init'] = quat_init
            targets['trans_2d'] = trans_xy
            targets['depth_id'] = depth_id
            targets['mask_synt'] = render_mask
            predictions['losses'] = self._compute_loss(predictions, targets)
        return predictions

    def _compute_loss(self, predictions, targets):
        assert 'quat_ego' in targets

        loss_dict = {}
        image_size = (cfg.INPUT_IMG_SIZE, cfg.INPUT_IMG_SIZE) # 假设 cfg.INPUT_IMG_SIZE 存在

        quat_gt = targets['quat_ego']
        trans_gt = targets['roi_obj_t']
        quat_pred = utils.rotation_to_quaternion(predictions['roi_obj_R'])
        trans_pred = utils.perspective_to_trans_3d(
            predictions['translation'], image_size, targets['roi_camK'])
        quat_gt, trans_gt = utils.top_matched_pose(
            quat_gt, trans_gt, quat_pred, trans_pred,
            targets['vertices_correlation'], targets['vertices_mean'],
            targets['quaternion_symmetries'],
            targets['translation_symmetries'], targets['symmetries_mask'])

        loss_dict = utils.vertex_rotation_loss(
            predictions['quat_bin'], predictions['quat_res'],
            quat_gt, targets['quat_bin'], targets['quat_init'],
            targets['vertices'], targets['vertices_mask'],
            image_size, targets['roi_camK'])

        loss_trans_dict = utils.vertex_translation_loss(
            predictions['trans_logits'], predictions['trans_res'],
            targets['trans_2d'], trans_gt, predictions['depth_bin'],
            predictions['depth_res'], targets['depth_id'], targets['roi_camK'],
            targets['roi_obj_R'], image_size,
            self.depth_min, self.depth_max, self.n_depth_bin)
        loss_dict.update(loss_trans_dict)

        bce_fn = nn.BCEWithLogitsLoss()
        mask_gt = targets['roi_mask'].unsqueeze(1)
        loss_dict['mask_visib'] = cfg.WEIGHT_MASK * bce_fn( # 假设 cfg.WEIGHT_MASK 存在
            predictions['roi_mask'], mask_gt)
        return loss_dict

