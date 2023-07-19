import fnx, re, eval

def test_sc_depth_kitti_depth():
    for name, builder in vars(fnx.pretrained.jiawangbian).items():
        if re.match("sc_depth_.*_kitti", name):
            abs_rel_error = eval.kitti_depth(builder, shape=(256, 832))
            assert abs_rel_error < 0.1
