import fnx, eval, re

def test_convnext_imagenet():
    for name, builder in vars(fnx.pretrained.facebookresearch).items():
        if not "ade20k" in name:
            if re.match("convnext_.*_imagenet1k.*", name):
                accurate = eval.imagenet1k(builder)
                assert accurate
                break
            elif re.match("convnext_.*_imagenet22k.*", name):
                accurate = eval.imagenet22k(builder)
                assert accurate

def test_convnext_upernet_ade20k():
    for name, builder in vars(fnx.pretrained.facebookresearch).items():
        if re.match("convnext_.*_ade20k.*", name):
            accuracy = eval.ade20k(builder)
            assert accuracy > 0.8

def test_detr_coco():
    for name, builder in vars(fnx.pretrained.facebookresearch).items():
        if re.match("detr_.*_coco", name):
            accuracy, mean_iou = eval.coco(builder, bbox_format="center-size")
            assert accuracy > 0.6
            assert mean_iou > 0.3

def test_dino_imagenet():
    for name, builder in vars(fnx.pretrained.facebookresearch).items():
        if re.match("dino_.*_imagenet1k.*", name):
            accurate = eval.imagenet1k(builder, resolution=(516, 516))
            assert accurate