import torch
from torch import nn
import torch.distributed as dist
from seg_opr.metric import hist_info, compute_score
import numpy as np
from score_utils import score_to_pred
from sc_utils import export_grid
import pdb
import pprint


def compute_loss(
    label_weight,
    label,
    output,
    criterion,
    sketch_gt,
    pred_sketch_raw,
    pred_sketch,
    pred_sketch_gsnn,
    pred_log_var,
    pred_mean,
    engine=None,
    config=None,
):
    """
    semantic loss
    """
    selectindex = torch.nonzero(label_weight.view(-1)).view(-1)
    label[label == 3] = 1
    label[label == 5] = 2
    filterLabel = torch.index_select(label.view(-1), 0, selectindex)

    filterOutput = torch.index_select(
        output.permute(0, 2, 3, 4, 1).contiguous().view(-1, config.num_classes),
        0,
        selectindex,
    )
    loss_semantic = criterion(filterOutput, filterLabel)
    loss_semantic = torch.mean(loss_semantic)

    """
    sketch loss
    """
    # pdb.set_trace()
    filter_sketch_gt = torch.index_select(sketch_gt.view(-1), 0, selectindex)
    filtersketch_raw = torch.index_select(
        pred_sketch_raw.permute(0, 2, 3, 4, 1).contiguous().view(-1, 2), 0, selectindex
    )
    filtersketch = torch.index_select(
        pred_sketch.permute(0, 2, 3, 4, 1).contiguous().view(-1, 2), 0, selectindex
    )
    filtersketchGsnn = torch.index_select(
        pred_sketch_gsnn.permute(0, 2, 3, 4, 1).contiguous().view(-1, 2), 0, selectindex
    )
    # print((filter_sketch_gt == filtersketch_raw).sum(),sketch_gt.shape)

    criterion_sketch = nn.CrossEntropyLoss(
        ignore_index=255, reduction="none"
    )  # .cuda()
    loss_sketch = criterion_sketch(filtersketch, filter_sketch_gt)
    loss_sketch = torch.mean(loss_sketch)
    loss_sketch_gsnn = criterion_sketch(filtersketchGsnn, filter_sketch_gt)
    loss_sketch_gsnn = torch.mean(loss_sketch_gsnn)
    loss_sketch_raw = criterion_sketch(filtersketch_raw, filter_sketch_gt)
    loss_sketch_raw = torch.mean(loss_sketch_raw)

    """ KLD loss """
    KLD = -0.5 * torch.mean(1 + pred_log_var - pred_mean.pow(2) - pred_log_var.exp())

    # reduce the whole loss over multi-gpu
    # if engine is not None:
    #     if engine.distributed:
    #         dist.all_reduce(loss_semantic, dist.ReduceOp.SUM)
    #         loss_semantic = loss_semantic / engine.world_size
    #         dist.all_reduce(loss_sketch, dist.ReduceOp.SUM)
    #         loss_sketch = loss_sketch / engine.world_size
    #         dist.all_reduce(loss_sketch_raw, dist.ReduceOp.SUM)
    #         loss_sketch_raw = loss_sketch_raw / engine.world_size
    #         dist.all_reduce(loss_sketch_gsnn, dist.ReduceOp.SUM)
    #         loss_sketch_gsnn = loss_sketch_gsnn / engine.world_size
    #         dist.all_reduce(KLD, dist.ReduceOp.SUM)
    #         KLD = KLD / engine.world_size
    #         loss = (
    #             loss_semantic
    #             + (loss_sketch + loss_sketch_raw) * config.sketch_weight
    #             + loss_sketch_gsnn * config.sketch_weight_gsnn
    #             + KLD * config.kld_weight
    #         )
    #     else:
    #         loss = (
    #             loss_semantic
    #             + (loss_sketch + loss_sketch_raw) * config.sketch_weight
    #             + loss_sketch_gsnn * config.sketch_weight_gsnn
    #             + KLD * config.kld_weight
    #         )
    loss = (
        loss_semantic
        + (loss_sketch + loss_sketch_raw) * config.sketch_weight
        + loss_sketch_gsnn * config.sketch_weight_gsnn
        + KLD * config.kld_weight
    )

    # loss_sketch_raw = torch.Tensor([0]).cuda()

    return loss, loss_semantic, loss_sketch_raw, loss_sketch, loss_sketch_gsnn, KLD


# def process_one_batch(dataloader, model, config, engine):
#     minibatch = dataloader.next()
#     img = minibatch["data"]
#     hha = minibatch["hha_img"]
#     label = minibatch["label"]
#     label_weight = minibatch["label_weight"]
#     tsdf = minibatch["tsdf"]
#     depth_mapping_3d = minibatch["depth_mapping_3d"]

#     sketch_gt = minibatch["sketch_gt"]
#     # res = np.zeros(129600)
#     # # label_weight = torch.ones(129600)
#     # # label_weight[(tsdf == 0) | (tsdf > 0) | (tsdf  == -1)] = 0

#     # occluded = (depth_mapping_3d == 307200) & (label_weight > 0)   & (label.flatten() != 255)
#     # res = res + occluded.detach().cpu().numpy()
#     # # print(res.shape,occluded.shape)
#     # grid_shape = [60,36,60]
#     # export_grid("label_sc.ply",label.reshape(grid_shape))
#     # export_grid("label_weight_sc.ply",label_weight.reshape(grid_shape).int())
#     # export_grid("occ_sc.ply",res.reshape(grid_shape).astype(int))
#     # raise NotADirectoryError
#     # print(img.shape)
#     # print(hha.shape)
#     # print(label.shape)
#     # print(label_weight.shape)
#     # print(tsdf.shape)
#     # print(depth_mapping_3d.shape)
#     # # print(img.shape)
#     # print("===============")
#     img = img.cuda(non_blocking=True).float()
#     label = label.cuda(non_blocking=True).long()
#     hha = hha.cuda(non_blocking=True)
#     tsdf = tsdf.cuda(non_blocking=True).float()
#     depth_mapping_3d = depth_mapping_3d.cuda(non_blocking=True).long()
#     # device = torch.device('cuda:0')
#     label_weight = label_weight.cuda(non_blocking=True).long()
#     sketch_gt = sketch_gt.cuda(non_blocking=True).long()
#     # img = np.ascontiguousarray(img[ :, :, :], dtype=np.float32)
#     # img = torch.FloatTensor(img).cuda(device).float()

#     # # hha = np.ascontiguousarray(hha[ :, :, :], dtype=np.float32)
#     # # hha = torch.FloatTensor(hha).cuda(device)

#     # # depth_mapping_3d = np.ascontiguousarray(depth_mapping_3d[ :], dtype=np.int32)
#     # # depth_mapping_3d = torch.LongTensor(depth_mapping_3d).cuda(device).long()

#     # # tsdf = np.ascontiguousarray(tsdf[ :], dtype=np
#     # # .float32)
#     # # tsdf = torch.FloatTensor(tsdf).cuda(device).float()

#     # # print(img.shape,hha.shape,tsdf.shape,depth_mapping_3d.shape,sketch_gt.shape)

#     if model.training:
#         (
#             output,
#             _,
#             pred_sketch_raw,
#             pred_sketch_gsnn,
#             pred_sketch,
#             pred_mean,
#             pred_log_var,
#         ) = model(img, depth_mapping_3d, tsdf, sketch_gt)
#     else:
#         output, _, pred_sketch_raw = model(img, depth_mapping_3d, tsdf, sketch_gt)
#     if output.requires_grad == False:
#         score = output[0]

#         score = torch.exp(score).detach().cpu()
#         pred = score_to_pred(score)

#         results_dict = {
#             "pred": pred,
#             "label": label.detach().cpu(),
#             "label_weight": label_weight.detach().cpu(),
#             "name": minibatch["fn"],
#             "mapping": depth_mapping_3d.detach().cpu(),
#         }
#     else:
#         results_dict = None
#     # pred_sketch_raw = torch.cat([torch.Tensor(np.logical_not(sketch_gt.cpu().numpy()).astype(np.float)).cuda(),sketch_gt.float()], dim=1)
#     cri_weights = torch.FloatTensor(torch.ones(config.num_classes))
#     cri_weights[0] = config.empty_loss_weight
#     criterion = nn.CrossEntropyLoss(
#         ignore_index=255, reduction="none", weight=cri_weights
#     ).cuda()
#     if model.training:
#         (
#             loss,
#             loss_semantic,
#             loss_sketch_raw,
#             loss_sketch,
#             loss_sketch_gsnn,
#             KLD,
#             results_dict,
#         ) = compute_loss(
#             label_weight,
#             label,
#             output,
#             criterion,
#             sketch_gt,
#             pred_sketch_raw,
#             pred_sketch,
#             pred_sketch_gsnn,
#             pred_log_var,
#             pred_mean,
#             engine,
#             config,
#         )

#         del (
#             img,
#             hha,
#             tsdf,
#             label_weight,
#             label,
#             output,
#             criterion,
#             sketch_gt,
#             pred_sketch_raw,
#             pred_sketch,
#             pred_sketch_gsnn,
#         )
#         return (
#             loss,
#             loss_semantic,
#             loss_sketch_raw,
#             loss_sketch,
#             loss_sketch_gsnn,
#             KLD,
#             results_dict,
#         )
#     else:
#         del (
#             img,
#             minibatch,
#             hha,
#             tsdf,
#             label_weight,
#             label,
#             output,
#             criterion,
#             sketch_gt,
#             pred_sketch_raw,
#         )
#         return results_dict
