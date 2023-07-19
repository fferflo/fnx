import fnx, eval, re

def test_convnext_imagenet():
    for name, builder in vars(fnx.pretrained.timm).items():
        if re.match("convnext_.*_imagenet1k.*", name):
            accurate = eval.imagenet1k(builder)
            assert accurate
        elif re.match("convnext_.*_imagenet12k.*", name):
            accurate = eval.imagenet12k(builder)
            assert accurate

def test_vit_imagenet():
    for name, builder in vars(fnx.pretrained.timm).items():
        if re.match("vit_.*_in1k$", name):
            accurate = eval.imagenet1k(builder)
            assert accurate

def test_efficientnet_imagenet():
    for name, builder in vars(fnx.pretrained.timm).items():
        if re.match("efficientnet_.*_imagenet1k$", name):
            accurate = eval.imagenet1k(builder, resolution=builder.resolution)
            assert accurate
