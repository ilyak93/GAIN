from my_voc_loader import VOCSegmentation
from pascal_voc_loader import PascalVOCLoader
import matplotlib.pyplot as plt
import numpy as np
import transform
import data

root = 'G:/VOC - dataset/'

aug_ = getattr(transform, 'DropoutAndAffine')
loader = PascalVOCLoader(root,
                         augmentations=aug_())
                         # augmentations=Affine)
'''
rds = data.RawDataset(
    root,
    num_workers=0,
    output_dims=224,
    augmentations=aug_(),
    batch_size_dict={'train': 1, 'test': 1})
'''
sample = loader[10]

ind = 52
sample = loader[ind]

im_ = sample['image'].numpy().transpose((1, 2, 0))
truth_ = sample['label/truths'][0].numpy().transpose((1, 2, 0))
mask_ = sample['label/masks'][0].numpy().transpose((1, 2, 0))
print(sample['label/name'])

plt.subplot(221)
plt.imshow(im_)
plt.subplot(222)
plt.imshow(truth_[..., 0])
plt.subplot(223)
plt.imshow(mask_[..., 0])
plt.show()
