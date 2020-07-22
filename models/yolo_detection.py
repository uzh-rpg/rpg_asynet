import torch
import torchvision


def yoloParseOutput(model_output, nr_bbox=2):
    """Parses the dense ouput to the predicted values"""
    # Get outputs
    x_norm_rel = model_output[..., 0:nr_bbox]  # Center x
    y_norm_rel = model_output[..., nr_bbox:nr_bbox*2]  # Center y
    h_norm_sqrt = model_output[..., nr_bbox*2:nr_bbox*3]  # Height
    w_norm_sqrt = model_output[..., nr_bbox*3:nr_bbox*4]  # Width
    pred_conf = torch.sigmoid(model_output[..., nr_bbox * 4:nr_bbox * 5])  # Object Confidence
    pred_cls = model_output[..., nr_bbox * 5:]  # Class Score

    return x_norm_rel, y_norm_rel, h_norm_sqrt, w_norm_sqrt, pred_conf, pred_cls


def yoloDetect(model_output, input_shape, threshold=None):
    """Computes the detections used in YOLO: https://arxiv.org/pdf/1506.02640.pdf"""
    cell_map_shape = torch.tensor(model_output.shape[1:3], device=model_output.device)
    cell_shape = input_shape / cell_map_shape
    x_norm_rel, y_norm_rel, h_norm_sqrt, w_norm_sqrt, pred_conf, pred_cls_conf = yoloParseOutput(model_output)

    h = h_norm_sqrt**2 * input_shape[0]
    w = w_norm_sqrt**2 * input_shape[1]

    x_rel = x_norm_rel * cell_shape[0]
    y_rel = y_norm_rel * cell_shape[1]
    cell_top_left = getGrid(input_shape, cell_map_shape)

    bbox_center = cell_top_left[None, :, :, None, :] + torch.stack([x_rel, y_rel], dim=-1)
    bbox_top_left_corner = bbox_center - torch.stack([h, w], dim=-1) // 2

    if threshold is None:
        return torch.cat([bbox_top_left_corner, h.unsqueeze(-1), w.unsqueeze(-1), pred_conf.unsqueeze(-1)], dim=-1)

    detected_bbox_idx = (pred_conf > threshold).nonzero().split(1, dim=-1)
    batch_idx = detected_bbox_idx[0]

    if batch_idx.shape[0] == 0:
        return torch.zeros([0, 7])

    detected_top_left_corner = bbox_top_left_corner[detected_bbox_idx].squeeze(1)
    detected_h = h[detected_bbox_idx]
    detected_w = w[detected_bbox_idx]
    pred_conf = pred_conf[detected_bbox_idx]

    pred_cls = torch.argmax(pred_cls_conf[detected_bbox_idx[:-1]], dim=-1)
    pred_cls_conf = pred_cls_conf[detected_bbox_idx[:-1]].squeeze(1)
    pred_cls_conf = pred_cls_conf[torch.arange(pred_cls.shape[0]), pred_cls.squeeze(-1)]

    # Convert from x, y to u, v
    det_bbox = torch.cat([batch_idx.float(), detected_top_left_corner[:, 1, None].float(),
                         detected_top_left_corner[:, 0, None].float(), detected_w.float(), detected_h.float(),
                         pred_cls.float(), pred_cls_conf[:, None].float(), pred_conf], dim=-1)

    return cropToFrame(det_bbox, input_shape)


def getGrid(input_shape, cell_map_shape):
    """Constructs a 2D grid with the cell center coordinates."""
    cell_shape = input_shape / cell_map_shape
    cell_top_left = torch.meshgrid([torch.arange(start=0, end=cell_map_shape[0]*cell_shape[0], step=cell_shape[0],
                                                 device=cell_shape.device),
                                    torch.arange(start=0, end=cell_map_shape[1]*cell_shape[1], step=cell_shape[1],
                                                 device=cell_shape.device)])
    return torch.stack(cell_top_left, dim=-1)


def cropToFrame(bbox, image_shape):
    """Checks if bounding boxes are inside frame. If not crop to border"""
    array_width = torch.ones_like(bbox[:, 1]) * image_shape[1] - 1
    array_height = torch.ones_like(bbox[:, 2]) * image_shape[0] - 1

    bbox[:, 1:3] = torch.max(bbox[:, 1:3], torch.zeros_like(bbox[:, 1:3]))
    bbox[:, 1] = torch.min(bbox[:, 1], array_width)
    bbox[:, 2] = torch.min(bbox[:, 2], array_height)

    bbox[:, 3] = torch.min(bbox[:, 3], array_width - bbox[:, 1])
    bbox[:, 4] = torch.min(bbox[:, 4], array_height - bbox[:, 2])

    return bbox


def nonMaxSuppression(detected_bbox, iou=0.6):
    """
    Iterates over the bboxes to peform non maximum suppression within each batch.

    :param detected_bbox[0, :]: [batch_idx, top_left_corner_u,  top_left_corner_v, width, height, predicted_class,
                                 predicted class confidence, object_score])
    :param iou: intersection over union, threshold for which the bbox are considered overlapping
    """
    i_sample = 0
    keep_bbox = []

    while i_sample < detected_bbox.shape[0]:
        same_batch_mask = detected_bbox[:, 0] == detected_bbox[i_sample, 0]
        nms_input = detected_bbox[same_batch_mask][:, [1, 2, 3, 4, 7]].clone()
        nms_input[:, [2, 3]] += nms_input[:, [0, 1]]

        # (u, v) or (x, y) should not matter
        keep_idx = torchvision.ops.nms(nms_input[:, :4], nms_input[:, 4], iou)
        keep_bbox.append(detected_bbox[same_batch_mask][keep_idx])
        i_sample += same_batch_mask.sum()

    if len(keep_bbox) != 0:
        filtered_bbox = torch.cat(keep_bbox, dim=0)
    else:
        filtered_bbox = torch.zeros([0, 8])

    return filtered_bbox

