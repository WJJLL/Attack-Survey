import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision.datasets as dset
import torch
class CocoCaptions(data.Dataset):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = self.coco.imgs.keys()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds = img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

class CocoDetection(data.Dataset):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.coco = dset.CocoDetection(root=root, annFile=annFile)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        input= self.coco[index][0]
        if self.transform is not None:
            input = self.transform(input)
        target = torch.randint(0, 89, size=(1,), dtype=torch.long)[0]

        return input, target

    def __len__(self):
        return len(self.coco)