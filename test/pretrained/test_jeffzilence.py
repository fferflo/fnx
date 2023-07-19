import fnx, re, eval

def test_transgeo_cvusa():
    for name, builder in vars(fnx.pretrained.jeffzilence).items():
        if "cvusa" in name:
            accuracy = eval.cvusa(builder, query_shape=(112, 616), reference_shape=(256, 256))
            assert accuracy == 1.0
