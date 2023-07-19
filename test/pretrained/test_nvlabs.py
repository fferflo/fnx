import fnx, eval, re

def test_segformer_ade20k():
    for name, builder in vars(fnx.pretrained.nvlabs).items():
        if re.match("segformer_.*_ade20k.*", name):
            accuracy = eval.ade20k(builder)
            assert accuracy > 0.6

def test_segformer_cityscapes():
    for name, builder in vars(fnx.pretrained.nvlabs).items():
        if re.match("segformer_.*_cityscapes.*", name):
            accuracy = eval.cityscapes(builder)
            assert accuracy > 0.8
