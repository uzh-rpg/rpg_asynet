import cv2
import matplotlib as mpl
mpl.use('Agg')  # Required to run the script with "screen" command without a X server
import matplotlib.pyplot as plt
import numpy as np


def visualizeEventsTime(events, height, width, path_name=None, last_k_events=None):
    """Visualizes the input events. Different saturation depending on arrival time of the event."""
    np_image = np.zeros([height, width, 3])

    sorted_indices = np.argsort(events[:, -2])
    sorted_events = events[sorted_indices, :]

    if last_k_events is not None:
        sorted_events = sorted_events[-last_k_events:, :]

    event_order = np.arange(sorted_events.shape[0], dtype=np.float)
    temporal_color = event_order / float(sorted_events.shape[0])
    mask_positive_events = sorted_events[:, -1] == 1
    positive_events = sorted_events[mask_positive_events, :3].round().astype(np.int)
    negative_events = sorted_events[np.logical_not(mask_positive_events), :3].round().astype(np.int)

    np_image[positive_events[:, 1], positive_events[:, 0], 1] = temporal_color[mask_positive_events]
    np_image[negative_events[:, 1], negative_events[:, 0], 0] = temporal_color[np.logical_not(mask_positive_events)]

    if path_name is None:
        return np_image
    else:
        fig, ax = plt.subplots()
        ax.imshow(np_image.astype(np.float))
        ax.axis('off')
        fig.savefig(path_name)
        plt.close()


def visualizeLocations(locations, shape, path_name=None, features=None, bounding_box=None, class_name=None):
    """Visualizes changing locations in a histogram. No time dependency"""
    np_image = np.zeros([shape[0], shape[1], 3])

    if features.shape[-1] != 2:
        # If features are not 2 dimensional, the event representation is a even queue
        feature_copy = features.reshape([features.shape[0], 2, -1])[:, 1, :].copy()
        features = np.zeros([feature_copy.shape[0], 2])
        features[:, 0] = (feature_copy == -1).sum(-1)
        features[:, 1] = (feature_copy == 1).sum(-1)

    if features is None:
        np_image[locations[:, 0], locations[:, 1], :] = np.array([0, 0, 1])
    else:
        np_image[locations[:, 0], locations[:, 1], :] = np.concatenate((features, np.zeros([features.shape[0], 1])),
                                                                       axis=-1)

    if bounding_box is not None:
        np_image = drawBoundingBoxes(np_image, bounding_box, class_name)

    if path_name is None:
        return np_image
    else:
        fig, ax = plt.subplots()
        ax.imshow(np_image.astype(np.float))
        ax.axis('off')
        fig.savefig(path_name)
        plt.close()


def visualizeHistogram(histogram, path_name=None):
    """Visualizes the input histogram"""
    height, width, _ = histogram.shape
    np_image = np.zeros([height, width, 3])

    if histogram.shape[-1] != 2:
        height, width, nr_channels = histogram.shape
        # If histogram has not 2 channels, the event representation is a event queue. Get polarity
        polarity_histogram = histogram[:, :, -(nr_channels//2):].copy()
        histogram = np.zeros([height, width, 2])
        histogram[:, :, 0] = np.abs((polarity_histogram * (polarity_histogram == -1)).sum(-1))
        histogram[:, :, 1] = (polarity_histogram * (polarity_histogram == 1)).sum(-1)

    np_image += (histogram[:, :, 1])[:, :, None] * np.array([0, 1, 0])[None, None, :]
    np_image += (histogram[:, :, 0])[:, :, None] * np.array([1, 0, 0])[None, None, :]
    np_image = np_image.clip(0, 1)

    if path_name is None:
        return np_image
    else:
        fig, ax = plt.subplots()
        ax.imshow(np_image.astype(np.float))
        ax.axis('off')
        fig.savefig(path_name)
        plt.close()


def visualizeConfusionMatrix(confusion_matrix, path_name=None):
    """
    Visualizes the confustion matrix using matplotlib.

    :param confusion_matrix: NxN numpy array
    :param path_name: if no path name is given, just an image is returned
    """
    nr_classes = confusion_matrix.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.matshow(confusion_matrix)
    ax.plot([-0.5, nr_classes - 0.5], [-0.5, nr_classes - 0.5], '-', color='grey')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Predicted')

    if path_name is None:
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    else:
        fig.savefig(path_name)
        plt.close()


def drawBoundingBoxes(np_image, bounding_boxes, class_name, ground_truth=True, rescale_image=True):
    """Draws the bounding boxes in the image"""
    resize_scale = 1.5
    bounding_boxes[:, :4] = (bounding_boxes.astype(np.float64)[:, :4] * resize_scale).astype(np.int)
    if rescale_image:
        new_dim = np.array(np_image.shape[:2], dtype=np.float) * resize_scale
        np_image = cv2.resize(np_image, tuple(new_dim.astype(int)[::-1]), interpolation=cv2.INTER_NEAREST)

    for i, bounding_box in enumerate(bounding_boxes):
        if bounding_box.sum() == 0:
            break
        np_image = drawBoundingBox(np_image, bounding_box, class_name[i], ground_truth)

    return np_image


def drawBoundingBox(np_image, bounding_box, class_name=None, ground_truth=False):
    """
    Draws a bounding box in the image.

    :param np_image: [H, W, C]
    :param bounding_box: [u, v, width, height]. (u, v) is bottom top point
    :param class_name: string
    """
    if ground_truth:
        bbox_color = np.array([0, 1, 1])
    else:
        bbox_color = np.array([1, 0, 1])
    width, height = bounding_box[2:4]
    np_image[bounding_box[1], bounding_box[0]:(bounding_box[0] + width)] = bbox_color
    np_image[bounding_box[1]:(bounding_box[1] + height), (bounding_box[0] + width)] = bbox_color
    np_image[(bounding_box[1] + height), bounding_box[0]:(bounding_box[0] + width)] = bbox_color
    np_image[bounding_box[1]:(bounding_box[1] + height), bounding_box[0]] = bbox_color

    if class_name is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)
        font_scale = 0.5
        thickness = 1
        bottom_left = tuple(((bounding_box[:2] + np.array([+1, height - 2]))).astype(int))

        # Draw Box
        (text_width, text_height) = cv2.getTextSize(class_name, font, fontScale=font_scale, thickness=thickness)[0]
        box_coords = ((bottom_left[0], bottom_left[1] + 2),
                      (bottom_left[0] + text_width + 2, bottom_left[1] - text_height - 2 + 2))
        color_format = (int(bbox_color[0]), int(bbox_color[1]), int(bbox_color[2]))
        cv2.rectangle(np_image, box_coords[0], box_coords[1], color_format, cv2.FILLED)

        cv2.putText(np_image, class_name, bottom_left, font, font_scale, font_color, thickness, cv2.LINE_AA)

    return np_image
