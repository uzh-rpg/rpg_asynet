import os
import abc
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import dataloader.dataset
from dataloader.loader import Loader
from models.facebook_sparse_vgg import FBSparseVGG
from models.facebook_sparse_object_det import FBSparseObjectDet
from models.dense_VGG import DenseVGG
from models.dense_object_det import DenseObjectDet
from models.yolo_loss import yoloLoss
from models.yolo_detection import yoloDetect
from models.yolo_detection import nonMaxSuppression
from utils.statistics_pascalvoc import BoundingBoxes, BoundingBox, BBType, VOC_Evaluator, MethodAveragePrecision
import utils.visualizations as visualizations


class AbstractTrainer(abc.ABC):
    def __init__(self, settings):
        self.settings = settings

        self.model = None
        self.scheduler = None
        self.nr_classes = None
        self.val_loader = None
        self.train_loader = None
        self.nr_val_epochs = None
        self.bounding_boxes =None
        self.object_classes = None
        self.nr_train_epochs = None
        self.model_input_size = None

        if self.settings.event_representation == 'histogram':
            self.nr_input_channels = 2
        elif self.settings.event_representation == 'event_queue':
            self.nr_input_channels = 30

        self.dataset_builder = dataloader.dataset.getDataloader(self.settings.dataset_name)
        self.dataset_loader = Loader

        self.writer = SummaryWriter(self.settings.ckpt_dir)
        self.createDatasets()

        self.buildModel()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.settings.init_lr)

        if settings.steps_lr is not None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=settings.steps_lr,
                                                                  gamma=settings.factor_lr)
        if settings.resume_training:
            self.loadCheckpoint(self.settings.resume_ckpt_file)

        self.batch_step = 0
        self.epoch_step = 0
        self.training_loss = 0
        self.val_batch_step = 0
        self.validation_loss = 0
        self.training_accuracy = 0
        self.validation_accuracy = 0
        self.max_validation_accuracy = 0
        self.val_confusion_matrix = np.zeros([self.nr_classes, self.nr_classes])

        # tqdm progress bar
        self.pbar = None

    @abc.abstractmethod
    def buildModel(self):
        """Model is constructed in child class"""
        pass

    def createDatasets(self):
        """
        Creates the validation and the training data based on the lists specified in the config/settings.yaml file.
        """
        train_dataset = self.dataset_builder(self.settings.dataset_path,
                                             self.settings.object_classes,
                                             self.settings.height,
                                             self.settings.width,
                                             self.settings.nr_events_window,
                                             augmentation=True,
                                             mode='training',
                                             event_representation=self.settings.event_representation)

        self.nr_train_epochs = int(train_dataset.nr_samples / self.settings.batch_size) + 1
        self.nr_classes = train_dataset.nr_classes
        self.object_classes = train_dataset.object_classes

        val_dataset = self.dataset_builder(self.settings.dataset_path,
                                           self.settings.object_classes,
                                           self.settings.height,
                                           self.settings.width,
                                           self.settings.nr_events_window,
                                           mode='validation',
                                           event_representation=self.settings.event_representation)
        self.nr_val_epochs = int(val_dataset.nr_samples / self.settings.batch_size) + 1

        self.train_loader = self.dataset_loader(train_dataset, batch_size=self.settings.batch_size,
                                                device=self.settings.gpu_device,
                                                num_workers=self.settings.num_cpu_workers, pin_memory=False)
        self.val_loader = self.dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                              device=self.settings.gpu_device,
                                              num_workers=self.settings.num_cpu_workers, pin_memory=False)

    @staticmethod
    def denseToSparse(dense_tensor):
        """
        Converts a dense tensor to a sparse vector.

        :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
        :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
        :return features: NumberOfActive x FeatureDimension
        """
        non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
        locations = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)

        select_indices = non_zero_indices.split(1, dim=1)
        features = torch.squeeze(dense_tensor[select_indices], dim=-2)

        return locations, features

    def resetValidation(self):
        """Resets all the validation statistics to zero"""
        self.val_batch_step = 0
        self.validation_loss = 0
        self.validation_accuracy = 0
        self.val_confusion_matrix = np.zeros([self.nr_classes, self.nr_classes])

    def saveValidationStatistics(self):
        """Saves the recorded validation statistics to an event file (tensorboard)"""
        self.writer.add_scalar('Validation/Validation_Loss', self.validation_loss, self.epoch_step)
        self.writer.add_scalar('Validation/Validation_Accuracy', self.validation_accuracy, self.epoch_step)

        self.val_confusion_matrix = self.val_confusion_matrix / (np.sum(self.val_confusion_matrix, axis=-1,
                                                                        keepdims=True) + 1e-9)
        plot_confusion_matrix = visualizations.visualizeConfusionMatrix(self.val_confusion_matrix)
        self.writer.add_image('Validation/Confusion_Matrix', plot_confusion_matrix, self.epoch_step,
                              dataformats='HWC')

    def storeLossesObjectDetection(self, loss_list):
        """Writes the different losses to tensorboard"""
        loss_names = ['Overall_Loss', 'Offset_Loss', 'Shape_Loss', 'Confidence_Loss', 'Confidence_NoObject_Loss',
                      'Class_Loss']

        for i_loss in range(len(loss_list)):
            loss_value = loss_list[i_loss].data.cpu().numpy()
            self.writer.add_scalar('TrainingLoss/' + loss_names[i_loss], loss_value, self.batch_step)

    def saveBoundingBoxes(self, gt_bbox, detected_bbox):
        """
        Saves the bounding boxes in the evaluation format

        :param gt_bbox: gt_bbox[0, 0, :]: ['u', 'v', 'w', 'h', 'class_id']
        :param detected_bbox[0, :]: [batch_idx, u, v, w, h, pred_class_id, pred_class_score, object score]
        """
        image_size = self.model_input_size.cpu().numpy()
        for i_batch in range(gt_bbox.shape[0]):
            for i_gt in range(gt_bbox.shape[1]):
                gt_bbox_sample = gt_bbox[i_batch, i_gt, :]
                id_image = self.val_batch_step * self.settings.batch_size + i_batch
                if gt_bbox[i_batch, i_gt, :].sum() == 0:
                    break

                bb_gt = BoundingBox(id_image, gt_bbox_sample[-1], gt_bbox_sample[0], gt_bbox_sample[1],
                                    gt_bbox_sample[2], gt_bbox_sample[3], image_size, BBType.GroundTruth)
                self.bounding_boxes.addBoundingBox(bb_gt)

        for i_det in range(detected_bbox.shape[0]):
            det_bbox_sample = detected_bbox[i_det, :]
            id_image = self.val_batch_step * self.settings.batch_size + det_bbox_sample[0]

            bb_det = BoundingBox(id_image, det_bbox_sample[5], det_bbox_sample[1], det_bbox_sample[2],
                                 det_bbox_sample[3], det_bbox_sample[4], image_size, BBType.Detected,
                                 det_bbox_sample[6])
            self.bounding_boxes.addBoundingBox(bb_det)

    def saveValidationStatisticsObjectDetection(self):
        """Saves the statistice relevant for object detection"""
        evaluator = VOC_Evaluator()
        metrics = evaluator.GetPascalVOCMetrics(self.bounding_boxes,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation)
        acc_AP = 0
        total_positives = 0
        for metricsPerClass in metrics:
            acc_AP += metricsPerClass['AP']
            total_positives += metricsPerClass['total positives']
        mAP = acc_AP / self.nr_classes
        self.validation_accuracy = mAP

        self.writer.add_scalar('Validation/Validation_Loss', self.validation_loss, self.epoch_step)
        self.writer.add_scalar('Validation/Validation_mAP', self.validation_accuracy, self.epoch_step)

    def getLearningRate(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def loadCheckpoint(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.epoch_step = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    def saveCheckpoint(self):
        file_path = os.path.join(self.settings.ckpt_dir, 'model_step_' + str(self.epoch_step) + '.pth')
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch_step}, file_path)


class FBSparseVGGModel(AbstractTrainer):
    def buildModel(self):
        """Creates the specified model"""
        self.model = FBSparseVGG(self.nr_classes, self.nr_input_channels,
                                 vgg_12=(self.settings.dataset_name == 'NCars'))
        self.model.to(self.settings.gpu_device)
        self.model_input_size = self.model.spatial_size

    def train(self):
        """Main training and validation loop"""
        while True:
            if (self.epoch_step % 5) == 0:
                self.validationEpoch()
            self.trainEpoch()

            self.epoch_step += 1
            self.scheduler.step()

    def trainEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_train_epochs, unit='Batch', unit_scale=True)
        self.model = self.model.train()
        self.training_loss = 0
        self.training_accuracy = 0
        loss_function = nn.CrossEntropyLoss()

        for i_batch, sample_batched in enumerate(self.train_loader):
            _, labels, histogram = sample_batched
            self.optimizer.zero_grad()

            # Change size to input size of sparse VGG
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(self.model_input_size))
            histogram = histogram.permute(0, 2, 3, 1)
            locations, features = self.denseToSparse(histogram)

            model_output = self.model([locations, features, histogram.shape[0]])

            loss = loss_function(model_output, target=labels)

            # Save training statistics
            predicted_classes = model_output.argmax(1)
            accuracy = (predicted_classes == labels).float().mean()
            self.training_accuracy += accuracy.data.cpu().numpy()
            self.training_loss += loss.data.cpu().numpy()

            loss.backward()
            self.optimizer.step()

            # Visualization
            # events_sample1 = events[events[:, -1] == 0, :-1]
            # file_path = os.path.join(self.settings.vis_dir, 'image_' + str(self.batch_step) + '.png')
            # visualizations.visualizeEventsTime(events_sample1.data.cpu().numpy(), self.settings.height,
            #                                    self.settings.width, path_name=file_path, last_k_events=None)
            if self.batch_step % (self.nr_train_epochs * 50) == 0:
                batch_one_mask = locations[:, -1] == 0
                vis_locations = locations[batch_one_mask, :2]
                features = features[batch_one_mask, :]

                # file_path = os.path.join(self.settings.vis_dir, 'image_' + str(self.batch_step) + '.png')
                # visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), self.model_input_size,
                #                                   features=features.cpu().numpy(), path_name=file_path)

                image = visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), self.model_input_size,
                                                          features=features.cpu().numpy())
                self.writer.add_image('Training/Input Histogram', image, self.epoch_step, dataformats='HWC')

            self.pbar.set_postfix(TrainLoss=loss.data.cpu().numpy())
            self.pbar.update(1)
            self.batch_step += 1

        self.writer.add_scalar('Training/Training_Accuracy', self.training_accuracy / float(self.nr_train_epochs),
                               self.epoch_step)
        self.writer.add_scalar('Training/Training_Loss', self.training_loss / float(self.nr_train_epochs),
                               self.epoch_step)
        self.writer.add_scalar('Training/Learning_Rate', self.getLearningRate(), self.epoch_step)
        self.pbar.close()

    def validationEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_val_epochs, unit='Batch', unit_scale=True)
        self.resetValidation()
        self.model = self.model.eval()
        loss_function = nn.CrossEntropyLoss()

        for i_batch, sample_batched in enumerate(self.val_loader):
            _, labels, histogram = sample_batched

            # Convert spatial dimension to model input size
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(self.model_input_size))
            histogram = histogram.permute(0, 2, 3, 1)
            locations, features = self.denseToSparse(histogram)

            with torch.no_grad():
                model_output = self.model([locations, features, histogram.shape[0]])

            loss = loss_function(model_output, target=labels)

            # Save validation statistics
            predicted_classes = model_output.argmax(1)
            accuracy = (predicted_classes == labels).float().mean()
            self.validation_loss += loss.data.cpu().numpy()
            self.validation_accuracy += accuracy.data.cpu().numpy()
            np.add.at(self.val_confusion_matrix, (predicted_classes.data.cpu().numpy(), labels.data.cpu().numpy()), 1)

            # Visualization
            # events_sample1 = events[events[:, -1] == 0, :-1]
            # file_path = os.path.join(self.settings.vis_dir, 'image_' + str(self.batch_step) + '.png')
            # visualizations.visualizeEventsTime(events_sample1.data.cpu().numpy(), self.settings.height,
            #                                    self.settings.width, path_name=file_path, last_k_events=None)

            self.pbar.set_postfix(ValLoss=loss.data.cpu().numpy())
            self.pbar.update(1)
            self.val_batch_step += 1

        self.validation_loss = self.validation_loss / float(self.val_batch_step)
        self.validation_accuracy = self.validation_accuracy / float(self.val_batch_step)
        self.saveValidationStatistics()

        if self.max_validation_accuracy < self.validation_accuracy:
            self.max_validation_accuracy = self.validation_accuracy
            self.saveCheckpoint()

        self.pbar.close()


class DenseVGGModel(AbstractTrainer):
    def buildModel(self):
        """Creates the specified model"""
        self.model = DenseVGG(self.nr_classes, in_c=self.nr_input_channels,
                              vgg_12=(self.settings.dataset_name == 'NCars'))
        self.model.to(self.settings.gpu_device)
        # Input size is determined by architecture
        if self.settings.dataset_name == 'NCaltech101':
            self.model_input_size = [191, 255]
        elif self.settings.dataset_name == 'NCars':
            self.model_input_size = [95, 127]

    def train(self):
        """Main training and validation loop"""
        while self.epoch_step <= 6507:
            if (self.epoch_step % 50) == 0:
                self.validationEpoch()
            self.trainEpoch()

            self.epoch_step += 1
            self.scheduler.step()

    def trainEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_train_epochs, unit='Batch', unit_scale=True)
        self.model = self.model.train()
        self.training_loss = 0
        self.training_accuracy = 0
        loss_function = nn.CrossEntropyLoss()

        for i_batch, sample_batched in enumerate(self.train_loader):
            events, labels, histogram = sample_batched
            self.optimizer.zero_grad()

            # Change size to input size of sparse VGG
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size))
            model_output = self.model(histogram)

            loss = loss_function(model_output, target=labels)

            # Save training statistics
            predicted_classes = model_output.argmax(1)
            accuracy = (predicted_classes == labels).float().mean()
            self.training_accuracy += accuracy.data.cpu().numpy()
            self.training_loss += loss.data.cpu().numpy()

            loss.backward()
            self.optimizer.step()

            if self.batch_step % (self.nr_train_epochs * 50) == 0:
                # file_path = os.path.join(self.settings.vis_dir, 'image_' + str(self.batch_step) + '.png')
                # visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), self.model_input_size,
                #                                   features=features.cpu().numpy(), path_name=file_path)
                image = visualizations.visualizeHistogram(histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
                self.writer.add_image('Training/Input Histogram', image, self.epoch_step, dataformats='HWC')

            self.pbar.set_postfix(TrainLoss=loss.data.cpu().numpy())
            self.pbar.update(1)
            self.batch_step += 1

        self.writer.add_scalar('Training/Training_Accuracy', self.training_accuracy / float(self.nr_train_epochs),
                               self.epoch_step)
        self.writer.add_scalar('Training/Training_Loss', self.training_loss / float(self.nr_train_epochs),
                               self.epoch_step)
        self.writer.add_scalar('Training/Learning_Rate', self.getLearningRate(), self.epoch_step)
        self.pbar.close()

    def validationEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_val_epochs, unit='Batch', unit_scale=True)
        self.resetValidation()
        self.model = self.model.eval()
        loss_function = nn.CrossEntropyLoss()

        for i_batch, sample_batched in enumerate(self.val_loader):
            events, labels, histogram = sample_batched

            # Convert spatial dimension to model input size
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(self.model_input_size))

            with torch.no_grad():
                model_output = self.model(histogram)

            loss = loss_function(model_output, target=labels)

            # Save validation statistics
            predicted_classes = model_output.argmax(1)
            accuracy = (predicted_classes == labels).float().mean()
            self.validation_loss += loss.data.cpu().numpy()
            self.validation_accuracy += accuracy.data.cpu().numpy()
            np.add.at(self.val_confusion_matrix, (predicted_classes.data.cpu().numpy(), labels.data.cpu().numpy()), 1)

            self.pbar.set_postfix(ValLoss=loss.data.cpu().numpy())
            self.pbar.update(1)
            self.val_batch_step += 1

        self.validation_loss = self.validation_loss / float(self.val_batch_step)
        self.validation_accuracy = self.validation_accuracy / float(self.val_batch_step)
        self.saveValidationStatistics()

        if self.max_validation_accuracy < self.validation_accuracy:
            self.max_validation_accuracy = self.validation_accuracy
            self.saveCheckpoint()

        self.pbar.close()


class SparseObjectDetModel(AbstractTrainer):
    def buildModel(self):
        """Creates the specified model"""
        self.model = FBSparseObjectDet(self.nr_classes, nr_input_channels=self.nr_input_channels,
                                       small_out_map=(self.settings.dataset_name == 'NCaltech101_ObjectDetection'))
        self.model.to(self.settings.gpu_device)
        self.model_input_size = self.model.spatial_size  # [191, 255]

        if self.settings.use_pretrained and (self.settings.dataset_name == 'NCaltech101_ObjectDetection' or
                                             self.settings.dataset_name == 'Prophesee'):
            self.loadPretrainedWeights()

    def loadPretrainedWeights(self):
        """Loads pretrained model weights"""
        checkpoint = torch.load(self.settings.pretrained_sparse_vgg)
        try:
            pretrained_dict = checkpoint['state_dict']
        except KeyError:
            pretrained_dict = checkpoint['model']

        pretrained_dict_short = {}
        for k, v in pretrained_dict.items():
            if 'sparseModel.25' in k:
                break
            pretrained_dict_short[k] = v

        self.model.load_state_dict(pretrained_dict_short, strict=False)

    def train(self):
        """Main training and validation loop"""
        validation_step = 50 - 48 * (self.settings.dataset_name == 'Prophesee')

        while True:
            self.trainEpoch()
            if (self.epoch_step % validation_step) == (validation_step - 1):
                self.validationEpoch()

            self.epoch_step += 1
            self.scheduler.step()

    def trainEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_train_epochs, unit='Batch', unit_scale=True)
        self.model = self.model.train()
        loss_function = yoloLoss

        for i_batch, sample_batched in enumerate(self.train_loader):
            event, bounding_box, histogram = sample_batched
            self.optimizer.zero_grad()

            # Change size to input size of sparse VGG
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size))
            histogram = histogram.permute(0, 2, 3, 1)
            # Change x, width and y, height
            bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[1].float()
                                       / self.settings.width).long()
            bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[0].float()
                                       / self.settings.height).long()
            locations, features = self.denseToSparse(histogram)

            # Deep Learning Magic
            model_output = self.model([locations, features, histogram.shape[0]])
            out = loss_function(model_output, bounding_box, self.model_input_size)
            loss = out[0]

            # Write losses statistics
            self.storeLossesObjectDetection(out)

            loss.backward()
            self.optimizer.step()

            if self.batch_step % (self.nr_train_epochs * 50) == 0:
                batch_one_mask = locations[:, -1] == 0
                vis_locations = locations[batch_one_mask, :2]
                features = features[batch_one_mask, :]

                with torch.no_grad():
                    detected_bbox = yoloDetect(model_output, self.model_input_size.to(model_output.device),
                                               threshold=0.3).long().cpu().numpy()
                    detected_bbox = detected_bbox[detected_bbox[:, 0] == 0, 1:-2]

                # Visualization
                # file_path = os.path.join(self.settings.vis_dir, 'image_' + str(self.batch_step) + '.png')
                # visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), self.model_input_size,
                #                                   features=features.cpu().numpy(), path_name=file_path,
                #                                   bounding_box=bounding_box[0, :, :].cpu().numpy(),
                #                                   class_name=[self.object_classes[i] for i in bounding_box[0, :, -1]])

                image = visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), self.model_input_size,
                                                          features=features.cpu().numpy(),
                                                          bounding_box=bounding_box[0, :, :].cpu().numpy(),
                                                          class_name=[self.object_classes[i]
                                                                      for i in bounding_box[0, :, -1]])
                image = visualizations.drawBoundingBoxes(image, detected_bbox[:, :-1],
                                                         class_name=[self.object_classes[i]
                                                                     for i in detected_bbox[:, -1]],
                                                         ground_truth=False, rescale_image=False)
                self.writer.add_image('Training/Input Histogram', image, self.epoch_step, dataformats='HWC')

            self.pbar.set_postfix(TrainLoss=loss.data.cpu().numpy())
            self.pbar.update(1)
            self.batch_step += 1

        self.writer.add_scalar('Training/Learning_Rate', self.getLearningRate(), self.epoch_step)
        self.pbar.close()

    def validationEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_val_epochs, unit='Batch', unit_scale=True)
        self.resetValidation()
        self.model = self.model.eval()
        self.bounding_boxes = BoundingBoxes()
        loss_function = yoloLoss
        # Images are upsampled for visualization
        val_images = np.zeros([2, int(self.model_input_size[0]*1.5), int(self.model_input_size[1]*1.5), 3])

        for i_batch, sample_batched in enumerate(self.val_loader):
            event, bounding_box, histogram = sample_batched

            # Convert spatial dimension to model input size
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size))
            histogram = histogram.permute(0, 2, 3, 1)

            # Change x, width and y, height
            bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[1].float()
                                          / self.settings.width).long()
            bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[0].float()
                                          / self.settings.height).long()
            locations, features = self.denseToSparse(histogram)

            with torch.no_grad():
                model_output = self.model([locations, features, histogram.shape[0]])
                loss = loss_function(model_output, bounding_box, self.model_input_size)[0]
                detected_bbox = yoloDetect(model_output, self.model_input_size.to(model_output.device),
                                           threshold=0.3)
                detected_bbox = nonMaxSuppression(detected_bbox, iou=0.6)
                detected_bbox = detected_bbox.cpu().numpy()

            # Save validation statistics
            self.saveBoundingBoxes(bounding_box.cpu().numpy(), detected_bbox)

            if self.val_batch_step % (self.nr_val_epochs - 2) == 0:
                batch_one_mask = locations[:, -1] == 0
                vis_locations = locations[batch_one_mask, :2]
                features = features[batch_one_mask, :]
                vis_detected_bbox = detected_bbox[detected_bbox[:, 0] == 0, 1:-2].astype(np.int)

                image = visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), self.model_input_size,
                                                          features=features.cpu().numpy(),
                                                          bounding_box=bounding_box[0, :, :].cpu().numpy(),
                                                          class_name=[self.object_classes[i]
                                                                      for i in bounding_box[0, :, -1]])
                image = visualizations.drawBoundingBoxes(image, vis_detected_bbox[:, :-1],
                                                         class_name=[self.object_classes[i]
                                                                     for i in vis_detected_bbox[:, -1]],
                                                         ground_truth=False, rescale_image=False)
                val_images[int(self.val_batch_step // (self.nr_val_epochs - 2))] = image

            self.pbar.set_postfix(ValLoss=loss.data.cpu().numpy())
            self.pbar.update(1)
            self.val_batch_step += 1
            self.validation_loss += loss

        self.validation_loss = self.validation_loss / float(self.val_batch_step)
        self.saveValidationStatisticsObjectDetection()
        self.writer.add_image('Validation/Input Histogram', val_images, self.epoch_step, dataformats='NHWC')

        if self.max_validation_accuracy < self.validation_accuracy:
            self.max_validation_accuracy = self.validation_accuracy
            self.saveCheckpoint()

        self.pbar.close()


class DenseObjectDetModel(AbstractTrainer):
    def buildModel(self):
        """Creates the specified model"""
        self.model = DenseObjectDet(self.nr_classes, in_c=self.nr_input_channels,
                                    small_out_map=(self.settings.dataset_name == 'NCaltech101_ObjectDetection'))
        self.model.to(self.settings.gpu_device)

        if self.settings.dataset_name == 'NCaltech101_ObjectDetection':
            self.model_input_size = torch.tensor([191, 255])
        elif self.settings.dataset_name == 'Prophesee':
            self.model_input_size = torch.tensor([223, 287])

        if self.settings.use_pretrained and (self.settings.dataset_name == 'NCaltech101_ObjectDetection' or
                                             self.settings.dataset_name == 'Prophesee'):
            self.loadPretrainedWeights()

    def loadPretrainedWeights(self):
        """Loads pretrained model weights"""
        checkpoint = torch.load(self.settings.pretrained_dense_vgg)
        try:
            pretrained_dict = checkpoint['state_dict']
        except KeyError:
            pretrained_dict = checkpoint['model']

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'conv_layers.' in k and int(k[12]) <= 4}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def train(self):
        """Main training and validation loop"""
        validation_step = 50 - 40 * (self.settings.dataset_name == 'Prophesee')

        while True:
            self.trainEpoch()
            if (self.epoch_step % validation_step) == (validation_step - 1):
                self.validationEpoch()

            self.epoch_step += 1
            self.scheduler.step()

    def trainEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_train_epochs, unit='Batch', unit_scale=True)
        self.model = self.model.train()
        loss_function = yoloLoss

        for i_batch, sample_batched in enumerate(self.train_loader):
            event, bounding_box, histogram = sample_batched
            self.optimizer.zero_grad()

            # Change size to input size of sparse VGG
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size))

            # Change x, width and y, height
            bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[1].float()
                                       / self.settings.width).long()
            bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[0].float()
                                       / self.settings.height).long()

            # Deep Learning Magic
            model_output = self.model(histogram)
            out = loss_function(model_output, bounding_box, self.model_input_size)
            loss = out[0]

            # Write losses statistics
            self.storeLossesObjectDetection(out)

            loss.backward()
            self.optimizer.step()

            if self.batch_step % (self.nr_train_epochs * 50) == 0:
                with torch.no_grad():
                    detected_bbox = yoloDetect(model_output, self.model_input_size.to(model_output.device),
                                               threshold=0.3).long().cpu().numpy()
                    detected_bbox = detected_bbox[detected_bbox[:, 0] == 0, 1:-2]

                image = visualizations.visualizeHistogram(histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
                image = visualizations.drawBoundingBoxes(image, bounding_box[0, :, :].cpu().numpy(),
                                                         class_name=[self.object_classes[i]
                                                                     for i in bounding_box[0, :, -1]],
                                                         ground_truth=True, rescale_image=True)
                image = visualizations.drawBoundingBoxes(image, detected_bbox[:, :-1],
                                                         class_name=[self.object_classes[i]
                                                                     for i in detected_bbox[:, -1]],
                                                         ground_truth=False, rescale_image=False)
                self.writer.add_image('Training/Input Histogram', image, self.epoch_step, dataformats='HWC')

            self.pbar.set_postfix(TrainLoss=loss.data.cpu().numpy())
            self.pbar.update(1)
            self.batch_step += 1

        self.writer.add_scalar('Training/Learning_Rate', self.getLearningRate(), self.epoch_step)
        self.pbar.close()

    def validationEpoch(self):
        self.pbar = tqdm.tqdm(total=self.nr_val_epochs, unit='Batch', unit_scale=True)
        self.resetValidation()
        self.model = self.model.eval()
        self.bounding_boxes = BoundingBoxes()
        loss_function = yoloLoss
        # Images are upsampled for visualization
        val_images = np.zeros([2, int(self.model_input_size[0]*1.5), int(self.model_input_size[1]*1.5), 3])

        for i_batch, sample_batched in enumerate(self.val_loader):
            event, bounding_box, histogram = sample_batched

            # Convert spatial dimension to model input size
            histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2),
                                                        torch.Size(self.model_input_size))

            # Change x, width and y, height
            bounding_box[:, :, [0, 2]] = (bounding_box[:, :, [0, 2]] * self.model_input_size[1].float()
                                          / self.settings.width).long()
            bounding_box[:, :, [1, 3]] = (bounding_box[:, :, [1, 3]] * self.model_input_size[0].float()
                                          / self.settings.height).long()

            with torch.no_grad():
                model_output = self.model(histogram)
                loss = loss_function(model_output, bounding_box, self.model_input_size)[0]
                detected_bbox = yoloDetect(model_output, self.model_input_size.to(model_output.device),
                                           threshold=0.3)
                detected_bbox = nonMaxSuppression(detected_bbox, iou=0.6)
                detected_bbox = detected_bbox.cpu().numpy()

            # Save validation statistics
            self.saveBoundingBoxes(bounding_box.cpu().numpy(), detected_bbox)

            if self.val_batch_step % (self.nr_val_epochs - 2) == 0:
                vis_detected_bbox = detected_bbox[detected_bbox[:, 0] == 0, 1:-2].astype(np.int)

                image = visualizations.visualizeHistogram(histogram[0, :, :, :].permute(1, 2, 0).cpu().int().numpy())
                image = visualizations.drawBoundingBoxes(image, bounding_box[0, :, :].cpu().numpy(),
                                                         class_name=[self.object_classes[i]
                                                                     for i in bounding_box[0, :, -1]],
                                                         ground_truth=True, rescale_image=True)
                image = visualizations.drawBoundingBoxes(image, vis_detected_bbox[:, :-1],
                                                         class_name=[self.object_classes[i]
                                                                     for i in vis_detected_bbox[:, -1]],
                                                         ground_truth=False, rescale_image=False)
                val_images[int(self.val_batch_step // (self.nr_val_epochs - 2))] = image

            self.pbar.set_postfix(ValLoss=loss.data.cpu().numpy())
            self.pbar.update(1)
            self.val_batch_step += 1
            self.validation_loss += loss

        self.validation_loss = self.validation_loss / float(self.val_batch_step)
        self.saveValidationStatisticsObjectDetection()
        self.writer.add_image('Validation/Input Histogram', val_images, self.epoch_step, dataformats='NHWC')

        if self.max_validation_accuracy < self.validation_accuracy:
            self.max_validation_accuracy = self.validation_accuracy
            self.saveCheckpoint()

        self.pbar.close()
