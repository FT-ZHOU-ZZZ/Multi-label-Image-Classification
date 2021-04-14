import torchvision.transforms as transforms
from coco import *
import torch.utils.data

if __name__ == '__main__':
    # image path
    data = r'E:\Dataset\MS-COCO\tmp'

    # image resize
    image_size = 448

    # batch resize
    batch_size = 32

    # number of workers
    num_worker = 16

    # image normalization
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=image_normalization_mean, std=image_normalization_std)
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        MultiScaleCrop(image_size, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        Warp(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # define dataset
    train_dataset = COCO2014(data, train_transform, 'train')
    val_dataset = COCO2014(data, test_transform, 'val')

    # data loading
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_worker)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=num_worker)