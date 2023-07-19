import fnx, eval, re

def test_swin_imagenet():
    for name, builder in vars(fnx.pretrained.microsoft).items():
        if re.match("swin_.*_imagenet1k.*", name):
            accurate = eval.imagenet1k(builder)
            assert accurate
        elif re.match("swin_.*_imagenet22k.*", name):
            accurate = eval.imagenet22k(builder)
            assert accurate
