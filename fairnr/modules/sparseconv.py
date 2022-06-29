import torch.nn as nn
import torch.nn.functional as F
import torch


class SparseConvNet(nn.Module):
    def __init__(self, voxel_size=0.005):
        super(SparseConvNet, self).__init__()
        self.voxel_size = voxel_size

        self.conv0 = double_conv(16, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def grid_sample(self, net_maps, grid_coords):
        net1, net2, net3, net4 = net_maps
        feature_1 = F.grid_sample(net1,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        feature_2 = F.grid_sample(net2,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        feature_3 = F.grid_sample(net3,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        feature_4 = F.grid_sample(net4,
                                  grid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        features = torch.cat((feature_1, feature_2, feature_3, feature_4), dim=1)
        features = features.view(features.size(0), -1, features.size(4))
        return features

    def preprocess_vertices(self, xyz, feats):
        # prepare valid volume
        min_xyz = torch.min(xyz,0).values
        max_xyz = torch.max(xyz,0).values
        min_xyz -= 0.05
        max_xyz += 0.05
        coords = ((xyz - min_xyz[None,:]) / self.voxel_size).round_().long()
        coords = torch.cat([coords.new_zeros(coords.size(0), 1), coords], 1)
        out_sh = ((max_xyz - min_xyz) / self.voxel_size).ceil_().long()

        x = 32
        out_sh = (out_sh | (x - 1)) + 1

        import spconv
        sp_input = spconv.SparseConvTensor(feats, coords.int(), out_sh.cpu().tolist(), 1)
        return sp_input, [min_xyz, max_xyz]
        
    def forward(self, xyz, feats):
        x, bounds = self.preprocess_vertices(xyz, feats)

        net_maps = []
        net = self.conv0(x)
        net = self.down0(net)
        net = self.conv1(net)
        net_maps.append(net.dense())
        
        net = self.down1(net)
        net = self.conv2(net)
        net_maps.append(net.dense())
        
        net = self.down2(net)
        net = self.conv3(net)
        net_maps.append(net.dense())
        
        net = self.down3(net)
        net = self.conv4(net)
        net_maps.append(net.dense())
        
        return {'net_maps': net_maps, 'sp_input': x, 'bounds': bounds}


def single_conv(in_channels, out_channels, indice_key=None):
    import spconv

    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    import spconv

    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    import spconv

    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    import spconv

    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
