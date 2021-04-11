import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt


available_transformers = ['Dropout', 'Affine', 'DropoutAndAffine', 'Translate']


class TransformerBase:
    def __init__(self, transform_labels=None, **kwargs):
        self.augmenter = self._build_augmenter(**kwargs)
        self.transform_labels = transform_labels

    def _build_augmenter(**kwargs):
        raise NotImplementedError()

    def __call__(self, data_dict):
        if not self.transform_labels or data_dict['label/name'] in self.transform_labels:
            image = data_dict['image']
            label = data_dict['label/truths']
            aug_ = self.augmenter.to_deterministic()
            image = aug_.augment_images([image])
            image = [im.astype(np.float32)/255.0 for im in image]
            truth = [ia.SegmentationMapOnImage(label[i],
                                               shape=image[0].shape,
                                               nb_classes=2)
                     for i in range(len(label))]
            truth = [aug_.augment_segmentation_maps([l])[0]
                     for l in truth]
            truth = [l.get_arr_int().astype(np.float32)
                     for l in truth]

            if('label/masks' in data_dict.keys()):
                masks = data_dict['label/masks']
                masks = [ia.SegmentationMapOnImage(masks[i],
                                                   shape=image[0].shape,
                                                   nb_classes=2)
                        for i in range(len(masks))]
                masks = [aug_.augment_segmentation_maps([l])[0]
                        for l in masks]
                masks = [l.get_arr_int().astype(np.float32)
                        for l in masks]
                data_dict['label/masks'] = masks

            data_dict['image'] = image[0]
            data_dict['label/truths'] = truth
            data_dict['label/masks'] = masks

        return data_dict


class DropoutAndAffine(TransformerBase):
    def _build_augmenter(self, **kwargs):
        affine_xform = Affine()
        dropout_xform = Dropout()
        aug = iaa.Sequential([
           affine_xform.augmenter,
           dropout_xform.augmenter],
           random_order=True)
        return aug


class Translate(TransformerBase):
    def _build_augmenter(self, **kwargs):
        aug = iaa.SomeOf((0, None), [
            iaa.Affine(translate_percent=(0, 0.1)),
            iaa.CropAndPad(percent=(0, -0.1), keep_size=True, sample_independently=True)],
            random_order=True)
        return aug


class Affine(TransformerBase):
    def _build_augmenter(self, **kwargs):
        aug = iaa.SomeOf((0, None), [
            iaa.Affine(translate_percent=(0, 0.1)),
            iaa.Affine(rotate=(0, 45)),
            iaa.Fliplr(1),
            iaa.Flipud(1)],
            random_order=True)
        return aug


class Dropout(TransformerBase):
    def _build_augmenter(self, **kwargs):
        aug = iaa.Sometimes(0.5,
            iaa.CoarseDropout((0.01, 0.05), size_percent=(0.05, 0.25)))
        return aug
