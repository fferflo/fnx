import fnx, eval, re

def test_vit_imagenet():
    for name, builder in vars(fnx.pretrained.googleresearch).items():
        if re.match("vit_.*_imagenet22k.*", name):
            accurate = eval.imagenet22k(builder)
            assert accurate

def test_mlp_mixer_imagenet():
    for name, builder in vars(fnx.pretrained.googleresearch).items():
        if re.match("mlp_mixer_.*_imagenet22k.*", name):
            accurate = eval.imagenet22k(builder)
            assert accurate
