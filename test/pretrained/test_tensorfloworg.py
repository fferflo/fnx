import fnx, eval, re

def test_resnet_v1_imagenet():
    for name, builder in vars(fnx.pretrained.tensorfloworg).items():
        if re.match("resnet_v1b_[0-9]*_imagenet$", name):
            accurate = eval.imagenet1k(builder)
            assert accurate
