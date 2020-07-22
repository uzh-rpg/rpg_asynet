import torch

from models.yolo_detection import yoloParseOutput
from models.yolo_detection import yoloDetect
from models.yolo_detection import getGrid


def yoloLoss(model_output, bounding_box, input_shape):
    """Computes the loss used in YOLO: https://arxiv.org/pdf/1506.02640.pdf"""
    lambda_coord = 5
    lambda_no_object = 0.5
    lambda_class = 1

    batch_s, gt_nr_bbox, _ = bounding_box.shape
    input_shape = input_shape.to(model_output.device)
    cell_map_shape = torch.tensor(model_output.shape[1:3], device=model_output.device)
    
    x_offset_norm, y_offset_norm, h_norm_sqrt, w_norm_sqrt, pred_conf, pred_cls = yoloParseOutput(model_output)

    out = processGroundTruth(bounding_box, input_shape, cell_map_shape)
    gt_cell_corner_offset_norm, gt_bbox_shape_norm_sqrt, gt_cell_x, gt_cell_y = out

    # Get IoU at gt_bbox_position
    bbox_detection = yoloDetect(model_output, input_shape)
    batch_indices = torch.arange(batch_s)[:, None].repeat([1, gt_nr_bbox])
    bbox_detection = bbox_detection[batch_indices, gt_cell_x, gt_cell_y, :, :]

    iou = computeIoU(bbox_detection[:, :, :, :4], bounding_box[:, :, :4]).detach()
    confidence_score, responsible_pred_bbox_idx = torch.max(iou, dim=-1)

    valid_gt_bbox_mask = bounding_box.sum(-1) > 0

    # ----- Offset Loss -----
    pred_cell_offset_norm = torch.stack([x_offset_norm, y_offset_norm], dim=-1)
    # Get the predictions, which include a object and correspond to the responsible cell
    pred_cell_offset_norm = pred_cell_offset_norm[batch_indices, gt_cell_x, gt_cell_y, responsible_pred_bbox_idx, :]

    offset_delta = pred_cell_offset_norm - gt_cell_corner_offset_norm
    offset_loss = (offset_delta**2).sum(-1)[valid_gt_bbox_mask].mean()

    # ----- Height&Width Loss -----
    pred_cell_shape_norm_sqrt = torch.stack([h_norm_sqrt, w_norm_sqrt], dim=-1)
    # Get the predictions, which include a object and correspond to the responsible cell
    pred_cell_shape_norm_sqrt = pred_cell_shape_norm_sqrt[batch_indices, gt_cell_x, gt_cell_y,
                                                          responsible_pred_bbox_idx, :]

    shape_delta = pred_cell_shape_norm_sqrt - gt_bbox_shape_norm_sqrt
    shape_loss = (shape_delta**2).sum(-1)[valid_gt_bbox_mask].mean()

    # ----- Object Confidence Loss -----
    # Get the predictions, which include a object and correspond to the responsible cell
    pred_conf_object = pred_conf[batch_indices, gt_cell_x, gt_cell_y, responsible_pred_bbox_idx]

    confidence_delta = pred_conf_object - confidence_score
    confidence_loss = (confidence_delta**2)[valid_gt_bbox_mask].mean()

    # ----- No Object Confidence Loss -----
    # Get the predictions, which do not include a object
    no_object_mask = torch.ones_like(pred_conf)
    no_object_mask[batch_indices[valid_gt_bbox_mask], gt_cell_x[valid_gt_bbox_mask], gt_cell_y[valid_gt_bbox_mask],
                   responsible_pred_bbox_idx[valid_gt_bbox_mask]] = 0
    confidence_no_object_loss = (pred_conf[no_object_mask.bool()]**2).mean()

    # ----- Class Prediction Loss -----
    loss_function = torch.nn.CrossEntropyLoss()
    pred_class_bbox = pred_cls[batch_indices, gt_cell_x, gt_cell_y, :]
    class_label = bounding_box[:, :, -1]

    pred_class_bbox = pred_class_bbox[valid_gt_bbox_mask]
    class_label = class_label[valid_gt_bbox_mask]
    class_loss = loss_function(pred_class_bbox, target=class_label)

    offset_loss = lambda_coord * offset_loss
    shape_loss = lambda_coord * shape_loss
    confidence_no_object_loss = lambda_no_object * confidence_no_object_loss
    class_loss = lambda_class * class_loss

    loss = offset_loss + shape_loss + confidence_loss + confidence_no_object_loss + class_loss

    return loss, offset_loss, shape_loss, confidence_loss, confidence_no_object_loss, class_loss


def processGroundTruth(bounding_box, input_shape, cell_map_shape):
    """Normalizes and computes the offset to the grid corner"""
    # Construct normalized, relative ground truth
    cell_corners = getGrid(input_shape, cell_map_shape)
    cell_shape = input_shape / cell_map_shape
    # bounding_box[0, 0, :]: ['u', 'v', 'w', 'h', 'class_id'].  (u, v) is top left point
    gt_bbox_center = bounding_box[:, :, :2] + bounding_box[:, :, 2:4] // 2
    # (u, v) -> (x, y)
    gt_cell_corner_offset_x = gt_bbox_center[:, :, 1, None] - cell_corners[None, None, :, 0, 0]
    gt_cell_corner_offset_x[gt_cell_corner_offset_x < 0] = 999999
    gt_cell_corner_offset_x, gt_cell_x = torch.min(gt_cell_corner_offset_x, dim=-1)

    gt_cell_corner_offset_y = gt_bbox_center[:, :, 0, None] - cell_corners[None, None, 0, :, 1]
    gt_cell_corner_offset_y[gt_cell_corner_offset_y < 0] = 999999
    gt_cell_corner_offset_y, gt_cell_y = torch.min(gt_cell_corner_offset_y, dim=-1)

    gt_cell_corner_offset = torch.stack([gt_cell_corner_offset_x, gt_cell_corner_offset_y], dim=-1)
    gt_cell_corner_offset_norm = gt_cell_corner_offset / cell_shape[None, None, :].float()

    # (width, height) -> (height, width)
    gt_bbox_shape = torch.stack([bounding_box[:, :, 3], bounding_box[:, :, 2]], dim=-1)
    gt_bbox_shape_norm_sqrt = torch.sqrt(gt_bbox_shape / input_shape.float())

    return gt_cell_corner_offset_norm, gt_bbox_shape_norm_sqrt, gt_cell_x, gt_cell_y


def computeIoU(bbox_detection, gt_bbox):
    """
    Computes for bounding boxes in bbox_detection the IoU with the gt_bbox.

    :param bbox_detection: [batch_size, bounding_box_to_compare, nr_pred_bbox, 4]
    :param gt_bbox: [batch_size, bounding_box_to_compare, 4]
    """
    bbox_detection = bbox_detection.long().float()
    gt_bbox = gt_bbox.float()

    # bbox_detection: bounding_box[0, 0, :]: ['u', 'v', 'w', 'h'].  (u, v) is top left point
    # (u, v) -> (x, y)
    intersection_left_x = torch.max(bbox_detection[:, :, :, 0], gt_bbox[:, :, None, 1])
    intersection_left_y = torch.max(bbox_detection[:, :, :, 1], gt_bbox[:, :, None, 0])
    intersection_right_x = torch.min(bbox_detection[:, :, :, 0] + bbox_detection[:, :, :, 2],
                                     gt_bbox[:, :, None, 1] + gt_bbox[:, :, None, 3])
    intersection_right_y = torch.min(bbox_detection[:, :, :, 1] + bbox_detection[:, :, :, 3],
                                     gt_bbox[:, :, None, 0] + gt_bbox[:, :, None, 2])

    intersection_area = (intersection_right_x - intersection_left_x) * (intersection_right_y - intersection_left_y)
    intersection_area = torch.max(intersection_area, torch.zeros_like(intersection_area))

    union_area = bbox_detection[:, :, :, 2:4].prod(axis=-1) + gt_bbox[:, :, None, 2:4].prod(axis=-1) - \
                    intersection_area
    intersection_over_union = intersection_area.float() / (union_area.float() + 1e-9)

    return intersection_over_union

