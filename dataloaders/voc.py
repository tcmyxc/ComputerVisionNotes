import collections
import os
from xml.etree.ElementTree import Element as ET_Element
from xml.etree.ElementTree import parse as ET_parse
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Dict, Optional, Tuple, List

from PIL import Image


class VOCDetection(VisionDataset):

    def __init__(
            self,
            root: str = "D:/dataset",
            year: str = "2007",
            image_set: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.year = year
        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")

        voc_root = os.path.join(self.root, os.path.join("VOCdevkit", f"VOC{year}"))

        if not os.path.isdir(voc_root):
            raise RuntimeError("Dataset not found or corrupted.")

        splits_dir = os.path.join(voc_root, "ImageSets", "Main")
        split_f = os.path.join(splits_dir, image_set.rstrip() + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(voc_root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(voc_root, "Annotations")
        self.targets = [os.path.join(target_dir, x + ".xml") for x in file_names]

        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)

    @property
    def annotations(self) -> List[str]:
        return self.targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())  # ET_parse(self.annotations[index] 读取 xml 文件，getroot() 得到根节点

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)  # Element转为list, 每个元素是一个Element。数组里面放的是自己的子节点
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)  # defaultdict 会避免 key 缺少错误
            for dc in map(VOCDetection.parse_voc_xml, children):  # 对 children 里的每一个元素调用 parse_voc_xml 函数
                for ind, v in dc.items():
                    def_dic[ind].append(v)  # 如果有相同的节点，会追加进去(如 object)
            if node.tag == "annotation":  # 根节点
                def_dic["object"] = [def_dic["object"]]  # 多加一个维度。因为可能有多个 object
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:  # 如果没有子节点，直接赋值
                voc_dict[node.tag] = text
        return voc_dict


if __name__ == '__main__':
    voc = VOCDetection()
    # print(voc[0])  # 图片12
    print(voc[1])  # 图片17

