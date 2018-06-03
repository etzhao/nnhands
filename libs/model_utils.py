import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def makePosList(h_pred, l_pred, params):

    num_joints = params['num_joints']
    image_size = params['image_size']

    p2d_y, p2d_x = np.unravel_index(torch.argmax(
        h_pred.view(num_joints, -1), dim=1).data.numpy(),
        (image_size, image_size))
    p2d = np.stack((p2d_x, p2d_y), axis=-1)

    p3d_x = l_pred[0].data.numpy()
    p3d_x = p3d_x[p2d_y, p2d_x]

    p3d_y = l_pred[1].data.numpy()
    p3d_y = p3d_y[p2d_y, p2d_x]

    p3d_z = l_pred[2].data.numpy()
    p3d_z = p3d_z[p2d_y, p2d_x]

    p3d = np.stack((p3d_x, p3d_y, p3d_z), axis=-1)
    return p2d, p3d


def show_joints(image, pos_2d, pos_3d):
    image = image/255
    fig = plt.figure(figsize=plt.figaspect(2.))
    ax = fig.add_subplot(2,1,1)
    height, width, depth = image.shape
    ax.imshow(image)
    ax.scatter(pos_2d[:,0], pos_2d[:, 1], s=10, marker='.', c='r')
    plt.axis('off')
    ax = fig.add_subplot(2,1,2, projection="3d")
    ax.view_init(-90,-90)
    plt.axis('off')
    ax.scatter(pos_3d[:,0], pos_3d[:,1], pos_3d[:,2], s=30)
    plt.show()


def makeHeatMapOneHot(pos2d_list, params):
    batch_size = params['batch_size']
    num_joints = params['num_joints']
    image_size = params['image_size']
    device = params['device']
    dtype = params['dtype']
    g_heatmap_size = params['g_heatmap_size']
    b_idx = params['b_idx']
    j_idx = params['j_idx']

    one_hot = torch.zeros(batch_size, num_joints, image_size, image_size,
        device=device, dtype=dtype) # size (N, 21, 224, 224)
    heatmap = torch.zeros(batch_size, num_joints,
        image_size + g_heatmap_size - 1, image_size + g_heatmap_size - 1,
        device=device, dtype=dtype)
    # size (N, 21, 224 + 8, 224 + 8)
    # Hao: Took me a while, but eventually figured out a way to do 4D indexing
    h_idx = pos2d_list[:, :, 1].view(-1).long()
    w_idx = pos2d_list[:, :, 0].view(-1).long()
    one_hot[b_idx, j_idx, h_idx, w_idx] = 1.0
    padding = int((g_heatmap_size - 1)/2)
    # still need some loops, but at least we only need a double loop, yeahhh, I guess

    for dh in range(-4, 5):
        for dw in range(-4, 5):
            cur_h_idx = h_idx + dh + padding
            cur_w_idx = w_idx + dw + padding
            cur_h_idx = cur_h_idx.long()
            cur_w_idx = cur_w_idx.long()
            heatmap[b_idx, j_idx, cur_h_idx, cur_w_idx] = math.exp(-1.0 * (dw**2 + dh**2))
    heatmap = heatmap[:, :, padding:-padding, padding:-padding]
    return heatmap, one_hot


def makeMaps(pos2d_list, pos3d_list, params):
    batch_size = params['batch_size']
    num_joints = params['num_joints']
    image_size = params['image_size']
    device = params['device']
    dtype = params['dtype']
    g_heatmap_size = params['g_heatmap_size']

    loc_map_x = torch.zeros(batch_size, num_joints, image_size,
            image_size, device=device, dtype=dtype)
    loc_map_y = torch.zeros(batch_size, num_joints, image_size,
            image_size, device=device, dtype=dtype)
    loc_map_z = torch.zeros(batch_size, num_joints, image_size,
            image_size, device=device, dtype=dtype)
    loc_map_x += pos3d_list[:, :, 0].view(batch_size, num_joints, 1, 1).float()
    loc_map_y += pos3d_list[:, :, 1].view(batch_size, num_joints, 1, 1).float()
    loc_map_z += pos3d_list[:, :, 2].view(batch_size, num_joints, 1, 1).float()

    heatmap, one_hot = makeHeatMapOneHot(pos2d_list, params)

    loc_map = torch.cat((loc_map_x, loc_map_y, loc_map_z), dim=1)
    return loc_map, heatmap, one_hot

def generate_blw(params):
    device = params['device']
    dtype = params['dtype']
    blw = torch.zeros(22, 21, device=device, dtype=dtype)
    bone_list = [(16, 18), (18, 19), (1, 0), (0, 2), (2, 3), (5, 4),
            (4, 6), (6, 7), (13, 12), (12, 14), (14, 15), (9, 8),
            (8, 10), (10, 11), (17, 16), (17, 1), (17, 5), (17, 13),
            (17, 9), (1, 5), (5, 13), (13, 9)]

    for idx, b in enumerate(bone_list):
        blw[idx, b[0]] = 1.0
        blw[idx, b[1]] = -1.0

    blw = blw.transpose(0, 1)
    return blw

class ComputeLoss(object):
    def __init__(self, params, two_stage=True, stack_size=2000):
        self.h_losses = []
        self.params = params
        self.two_stage = two_stage
        self.stack_size = stack_size
        self.stage = 0

    def __call__(self, heatmap, one_hot, loc_map, hmap_pred, l_pred, blw):
        batch_size = self.params['batch_size']
        num_joints = self.params['num_joints']
        image_size = self.params['image_size']
        USE_GPU = self.params['USE_GPU']
        g_heatmap_size = self.params['g_heatmap_size']

        epsilon = 1e-8
        loss_scale = 1.0 / (batch_size * num_joints * g_heatmap_size**2 * 3)

        global modelLocmap
        global optimizer

        hmap_pred = nn.functional.softmax(hmap_pred.view(batch_size, num_joints, -1), dim=2)
        h_loss = torch.sum(hmap_pred * one_hot.view(batch_size, num_joints, -1), dim=2) + epsilon
        h_loss = torch.sum(-1.0 * h_loss.log()) / (batch_size * num_joints)

        j_idx_flat = torch.argmax(hmap_pred.view(batch_size, num_joints, -1), dim=2)
        # because j_idx_flat is already a long tensor, the decimal truncation is implicit, no need to floor()
        p2d_y = j_idx_flat / image_size
        p2d_x = torch.remainder(j_idx_flat, image_size)
        p2d = torch.cat((p2d_x.view(batch_size, num_joints, 1), p2d_y.view(batch_size, num_joints, 1)), dim=2)

        hp, one_hot_pred = makeHeatMapOneHot(p2d, self.params)

        # total heatmap
        t_heatmap = hp + heatmap
        t_heatmap = t_heatmap.repeat(1, 3, 1, 1) # shape(N, 63, 224, 224)

        l_pred = torch.cat((l_pred[:, 0:1, :, :].expand(-1, 21, -1, -1),
                            l_pred[:, 1:2, :, :].expand(-1, 21, -1, -1),
                            l_pred[:, 2:3, :, :].expand(-1, 21, -1, -1)), dim=1)
        l_loss = torch.sum(torch.pow(t_heatmap * (l_pred - loc_map), 2)) * loss_scale

        # Skeleton (bone length) constraint
        # get a weighted average of predicted joint coordinates
        coord_p = torch.sum(torch.sum(one_hot_pred.repeat(1, 3, 1, 1) * l_pred, dim=3), dim=2)
        x_p = coord_p[:, 0:21]
        y_p = coord_p[:, 21:42]
        z_p = coord_p[:, 42:63]

        # GT bone length
        blb = (torch.matmul(loc_map[:, 0:num_joints, 0, 0], blw)**2 + \
            torch.matmul(loc_map[:, num_joints:2*num_joints, 0, 0], blw)**2 + \
            torch.matmul(loc_map[:, 2*num_joints:3*num_joints, 0, 0], blw)**2)

        # predicted bone length, using predicted joint 2D location as filter
        bl_x = torch.matmul(x_p, blw)**2
        bl_y = torch.matmul(y_p, blw)**2
        bl_z = torch.matmul(z_p, blw)**2

        # add epsilon for numerical stability, since sqrt(x)'s derivative is 0.5/sqrt(x)
        bone_diff = torch.sqrt(bl_x + bl_y + bl_z + epsilon) - torch.sqrt(blb + epsilon)

        bl_loss = torch.sum(bone_diff.abs()) / (batch_size * num_joints)

        # first minimize the 2D prediction h_loss
        # then focus on the 3D location loss
        loss = 0.0

        if self.two_stage:
            if USE_GPU == True:
                h_loss = h_loss.cpu()
                self.h_losses.append(h_loss.data.numpy())
                h_loss = h_loss.cuda()
            else:
                self.h_losses.append(h_loss.data.numpy())

            if len(self.h_losses) > self.stack_size:
                self.h_losses = self.h_losses[1:]

            if self.stage == 0 and (len(self.h_losses) < self.stack_size or \
                    np.mean(self.h_losses[0:int(self.stack_size/2)]) > np.mean(self.h_losses[int(self.stack_size/2):]) ):
                l_loss = 0.05 * l_loss
                bl_loss = 0.05 * bl_loss
            else:
                # move on to stage 1, we won't go back to stage 0
                # focus on 3D location loss
                # freeze model basis portion and the 2D heatmap portion
                if self.stage == 0:
                    self.stage = 1
                    print("###### Stage 1 Start (Second Stage) ######")
                    # only need to freeze the parameters once
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in modelHeatmap.parameters():
                        param.requires_grad = False

                    optimizer = torch.optim.Adam(modelLocmap.parameters(), lr=1.0e-4)
        loss = h_loss + l_loss + bl_loss

        return loss, [h_loss.data.cpu().numpy().tolist(),
                l_loss.data.cpu().numpy().tolist(),
                bl_loss.data.cpu().numpy().tolist(),
                self.stage]


def get_loss(model, modelHeatmap, modelLocmap, image, pos2d_list, pos3d_list, blw, params):
    USE_GPU = params['USE_GPU']
    if USE_GPU:
        pos2d_list = pos2d_list.cuda()
        pos3d_list = pos3d_list.cuda()
        image = image.cuda()

    loc_map, heatmap, one_hot = makeMaps(pos2d_list, pos3d_list, params)
    y_pred = model(image)
    h_pred = modelHeatmap(y_pred)
    l_pred = modelLocmap(y_pred)
    loss, loss_detailed = ComputeLoss(params)(heatmap, one_hot, loc_map, h_pred, l_pred, blw)
    return loss, loss_detailed


def load_model(model, modelHeatmap, modelLocmap, optimizer, fp_head, fp_tail, params):
    device = params['device']
    model.load_state_dict(torch.load('{}/model_param{}.pt'.format(fp_head, fp_tail),
        map_location=device))
    modelHeatmap.load_state_dict(torch.load('{}/modelHeatmap_param{}.pt'.format(fp_head, fp_tail),
        map_location=device))
    modelLocmap.load_state_dict(torch.load('{}/modelLocmap_param{}.pt'.format(fp_head, fp_tail),
        map_location=device))
    if optimizer != None:
        optimizer.load_state_dict(torch.load('{}/optimizer_param{}.pt'.format(fp_head,
            fp_tail), map_location=device))
    return model, modelHeatmap, modelLocmap, optimizer

def save_model(epoch, idx, model, modelHeatmap, modelLocmap, optimizer):
    torch.save(model.state_dict(), 'model_param_e{}_i{}.pt'.format(epoch, idx))
    torch.save(modelHeatmap.state_dict(), 'modelHeatmap_param_e{}_i{}.pt'.format(epoch, idx))
    torch.save(modelLocmap.state_dict(), 'modelLocmap_param_e{}_i{}.pt'.format(epoch, idx))
    torch.save(optimizer.state_dict(), 'optimizer_param_e{}_i{}.pt'.format(epoch, idx))

