import fnx, re, eval

def test_segnext_cityscapes():
    for name, builder in vars(fnx.pretrained.jittor).items():
        if re.match("segnext_.*_cityscapes.*", name):
            accuracy = eval.cityscapes(builder)
            assert accuracy > 0.8

def test_segnext_ade20k():
    for name, builder in vars(fnx.pretrained.jittor).items():
        if re.match("segnext_.*_ade20k.*", name):
            accuracy = eval.ade20k(builder)
            assert accuracy > 0.5

def test_segnext_isaid():
    for name, builder in vars(fnx.pretrained.jittor).items():
        if re.match("segnext_.*_isaid.*", name):
            accuracy = eval.isaid(builder)
            assert accuracy > 0.6
