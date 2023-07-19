import imageio, os, fnx, jax, yaml
import numpy as np
import haiku as hk
from functools import partial

datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def classification(builder, color, true_class):
    color_preprocessed = builder.preprocess(color)
    x0 = color_preprocessed[np.newaxis]
    rng = jax.random.PRNGKey(42)

    def model_fn(x, is_training):
        with fnx.set_is_training(is_training):
            with fnx.scope("test_additional_name"):
                x = builder(x)
        return x
    model = hk.transform_with_state(model_fn)

    params, state = model.init(rng, x0, is_training=True)

    f = jax.jit(partial(model.apply, is_training=False))
    x, new_state = f(params, state, rng, x0)
    x = x[0]

    return np.argmax(x, axis=0) == true_class

def segmentation(builder, color, true_labels):
    color_preprocessed = builder.preprocess(color)

    x0 = color_preprocessed[np.newaxis]
    rng = jax.random.PRNGKey(42)

    def model_fn(x, is_training):
        with fnx.set_is_training(is_training):
            with fnx.scope("test_additional_name"):
                x = builder(x)
        return x
    model = hk.transform_with_state(model_fn)

    params, state = model.init(rng, x0, is_training=True)

    f = jax.jit(partial(model.apply, is_training=False))
    x, new_state = f(params, state, rng, x0)
    x = x[0]

    classes_num = x.shape[-1]
    return float(np.count_nonzero(true_labels == np.argmax(x, axis=-1))) / np.count_nonzero(np.logical_and(0 <= true_labels, true_labels < classes_num)) # Compute accuracy

def detection(builder, color, true_labels, bbox_format):
    color_preprocessed = builder.preprocess(color)

    x0 = color_preprocessed[np.newaxis]
    rng = jax.random.PRNGKey(42)

    def model_fn(x, is_training):
        with fnx.set_is_training(is_training):
            with fnx.scope("test_additional_name"):
                x = builder(x)
        return x
    model = hk.transform_with_state(model_fn)

    params, state = model.init(rng, x0, is_training=True)

    f = jax.jit(partial(model.apply, is_training=False))
    (probs, pred_bbox), new_state = f(params, state, rng, x0)
    classes_num = probs.shape[-1]
    pred_classes = np.argmax(probs[0], axis=-1)
    pred_bbox = np.copy(pred_bbox[0])

    mask = pred_classes < classes_num - 1
    pred_classes = pred_classes[mask]
    pred_bbox = pred_bbox[mask]

    if bbox_format == "center-size":
        pred_bbox[..., (0, 1)] -= pred_bbox[..., (2, 3)] * 0.5
        pred_bbox[..., (2, 3)] += pred_bbox[..., (0, 1)]
    elif bbox_format == "topleft-bottomright":
        pass
    else:
        raise ValueError(f"Invalid bbox_format {bbox_format}")
    pred_bbox[..., :2] *= np.asarray(color.shape[:2])
    pred_bbox[..., 2:] *= np.asarray(color.shape[:2])

    gt_bbox = np.asarray([x["bbox"] for x in true_labels])
    gt_classes = np.asarray([x["category_id"] for x in true_labels])


    # Find matches between gt and pred boxes via very simple criterion
    index = (pred_bbox[:, np.newaxis, :2] + pred_bbox[:, np.newaxis, 2:]) * 0.5 - (gt_bbox[np.newaxis, :, :2] + gt_bbox[np.newaxis, :, 2:]) * 0.5
    index = np.linalg.norm(index, axis=-1)
    index = np.argmin(index, axis=0)
    pred_bbox = pred_bbox[index]
    pred_classes = pred_classes[index]

    # Compute IoU
    gt_bbox_area = (gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1])
    pred_bbox_area = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])

    max_topleft = np.maximum(pred_bbox[:, :2], gt_bbox[:, :2])
    min_bottomright = np.minimum(pred_bbox[:, 2:], gt_bbox[:, 2:])
    intersection = np.maximum(min_bottomright - max_topleft, 0)
    intersection = intersection[..., 0] * intersection[..., 1]

    union = pred_bbox_area + gt_bbox_area - intersection

    mean_iou = np.mean(intersection / union)
    accuracy = np.mean(np.where(gt_classes == pred_classes, 1.0, 0.0))

    return accuracy, mean_iou

def nearest_neighbors(builder, query, reference):
    q0 = [builder.preprocess(q)[np.newaxis] for q in query]
    r0 = [builder.preprocess(r)[np.newaxis] for r in reference]
    rng = jax.random.PRNGKey(42)

    def model_fn(x, is_training):
        with fnx.set_is_training(is_training):
            with fnx.scope("test_additional_name"):
                x = builder(x, mode="reference")
        return x
    model = hk.transform_with_state(model_fn)

    params, state = model.init(rng, r0[0], is_training=True)
    f = jax.jit(partial(model.apply, is_training=False))
    r_out = []
    for r in r0:
        r, new_state = f(params, state, rng, r)
        r_out.append(r[0])
    r_out = np.asarray(r_out)

    def model_fn(x, is_training):
        with fnx.set_is_training(is_training):
            with fnx.scope("test_additional_name"):
                x = builder(x, mode="query")
        return x
    model = hk.transform_with_state(model_fn)

    params, state = model.init(rng, q0[0], is_training=True)
    f = jax.jit(partial(model.apply, is_training=False))
    q_out = []
    for q in q0:
        q, new_state = f(params, state, rng, q)
        q_out.append(q[0])
    q_out = np.asarray(q_out)

    q_out = q_out / np.linalg.norm(q_out, axis=1, keepdims=True)
    r_out = r_out / np.linalg.norm(r_out, axis=1, keepdims=True)

    similarity = np.matmul(r_out, q_out.T)
    return float(np.count_nonzero(np.argmax(similarity, axis=0) == np.arange(similarity.shape[0]))) / similarity.shape[0]

def depth(builder, color, true_depth):
    color_preprocessed = builder.preprocess(color)

    x0 = color_preprocessed[np.newaxis]
    rng = jax.random.PRNGKey(42)

    def model_fn(x, is_training):
        with fnx.set_is_training(is_training):
            with fnx.scope("test_additional_name"):
                x = builder(x)
        return x
    model = hk.transform_with_state(model_fn)

    params, state = model.init(rng, x0, is_training=True)

    f = jax.jit(partial(model.apply, is_training=False))
    x, new_state = f(params, state, rng, x0)
    x = x[0]
    pred_depth = x

    pred_depth = pred_depth.reshape([-1])
    true_depth = true_depth.reshape([-1])
    mask = true_depth > 0
    pred_depth = pred_depth[mask]
    true_depth = true_depth[mask]

    pred_depth = pred_depth * (np.median(true_depth) / np.median(pred_depth)) # Align scale

    return np.mean(np.abs(pred_depth - true_depth) / true_depth) # Compute abs_rel_error



def imagenet(builder, true_class, resolution=None):
    color = imageio.imread(os.path.join(datadir, "weimaraner.png")).astype("float32")[:, :, :3]
    if not resolution is None:
        color = jax.image.resize(color, (resolution[0], resolution[1], 3), method="bilinear")
    return classification(
        builder=builder,
        color=color,
        true_class=true_class,
    )

def imagenet1k(builder, **kwargs):
    return imagenet(builder, true_class=178, **kwargs) # https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/

def imagenet12k(builder, **kwargs):
    return imagenet(builder, true_class=1462, **kwargs) # https://github.com/rwightman/pytorch-image-models/blob/main/results/imagenet12k_synsets.txt

def imagenet22k(builder, **kwargs):
    return imagenet(builder, true_class=2219, **kwargs) # https://github.com/google-research/big_transfer/issues/7

def ade20k(builder): # ADE_val_00000004
    return segmentation(
        builder=builder,
        color=imageio.imread(os.path.join(datadir, "ade20k", "color.jpg")).astype("float32"),
        true_labels=(imageio.imread(os.path.join(datadir, "ade20k", "labels.png")) - 1).astype("uint8"),
    )

def cityscapes(builder): # frankfurt_000000_000294_leftImg8bit
    return segmentation(
        builder=builder,
        color=imageio.imread(os.path.join(datadir, "cityscapes", "color.png")).astype("float32"),
        true_labels=imageio.imread(os.path.join(datadir, "cityscapes", "labels.png")).astype("uint8"),
    )

def isaid(builder):
    return segmentation(
        builder=builder,
        color=imageio.imread(os.path.join(datadir, "isaid", "P0003.png")).astype("float32")[:896, :896],
        true_labels=imageio.imread(os.path.join(datadir, "isaid", "P0003_trainids.png")).astype("uint8")[:896, :896],
    )

def coco(builder, bbox_format):
    color = imageio.imread(os.path.join(datadir, "coco", "000000002473.jpg")).astype("float32")[:, :, :3]
    with open(os.path.join(datadir, "coco", "000000002473.yaml"), "r") as f:
        true_labels = yaml.safe_load(f)
    return detection(
        builder=builder,
        color=color,
        true_labels=true_labels,
        bbox_format=bbox_format,
    )

def cvusa(builder, query_shape=None, reference_shape=None):
    if query_shape is None:
        resize_query = lambda image: image
    else:
        resize_query = lambda image: jax.image.resize(image, (query_shape[0], query_shape[1], 3), "bilinear")
    if reference_shape is None:
        resize_reference = lambda image: image
    else:
        resize_reference = lambda image: jax.image.resize(image, (reference_shape[0], reference_shape[1], 3), "bilinear")
    return nearest_neighbors(
        builder=builder,
        query=[resize_query(image) for image in [
            imageio.imread(os.path.join(datadir, "cvusa", "ground_0041073.jpg")).astype("float32"),
            imageio.imread(os.path.join(datadir, "cvusa", "ground_0000001.jpg")).astype("float32"),
            imageio.imread(os.path.join(datadir, "cvusa", "ground_0000005.jpg")).astype("float32"),
            imageio.imread(os.path.join(datadir, "cvusa", "ground_0000021.jpg")).astype("float32"),
        ]],
        reference=[resize_reference(image) for image in [
            imageio.imread(os.path.join(datadir, "cvusa", "aerial_0041073.jpg")).astype("float32"),
            imageio.imread(os.path.join(datadir, "cvusa", "aerial_0000001.jpg")).astype("float32"),
            imageio.imread(os.path.join(datadir, "cvusa", "aerial_0000005.jpg")).astype("float32"),
            imageio.imread(os.path.join(datadir, "cvusa", "aerial_0000021.jpg")).astype("float32"),
        ]],
    )

def kitti_depth(builder, shape=None):
    color = imageio.imread(os.path.join(datadir, "kitti-depth", "2011_09_26_drive_0002_sync_image_0000000053_image_02.png")).astype("float32")
    true_depth = imageio.imread(os.path.join(datadir, "kitti-depth", "2011_09_26_drive_0002_sync_groundtruth_depth_0000000053_image_02.png")).astype("uint16")

    mask = true_depth == 0
    y1, y2 = int(0.40810811 * true_depth.shape[0]), int(0.99189189 * true_depth.shape[0])
    x1, x2 = int(0.03594771 * true_depth.shape[1]), int(0.96405229 * true_depth.shape[1])
    mask[y1:y2, x1:x2] = False
    true_depth = true_depth.astype("float32") / 256
    true_depth[mask] = -1

    if not shape is None:
        color = jax.image.resize(color, (shape[0], shape[1], 3), "bilinear")
        true_depth = jax.image.resize(true_depth, (shape[0], shape[1]), "nearest")

    return depth(
        builder=builder,
        color=color,
        true_depth=true_depth,
    )



def text_generation(builder):
    with open(os.path.join(datadir, "english.txt"), "r") as f:
        text = f.read()

    tokens = builder.encoder.encode_ordinary(text) # text = builder.encoder.decode(tokens)
    tokens.append(builder.encoder.eot_token)

    rng = jax.random.PRNGKey(42)

    def model_fn(x, is_training):
        with fnx.set_is_training(is_training):
            with fnx.scope("test_additional_name"):
                x = builder(x)
        return x
    model = hk.transform_with_state(model_fn)

    params, state = model.init(rng, np.asarray(tokens)[np.newaxis], is_training=True)
    f = jax.jit(partial(model.apply, is_training=False))

    block_size = 100
    probs = []
    for offset in range(len(tokens) - block_size - 1):
        x, _ = f(params, state, rng, np.asarray(tokens[offset:offset + block_size])[np.newaxis])
        probs.append(x[0, -1, tokens[offset + block_size]])

    return np.exp(-np.mean(np.log(probs))) # Compute perplexity
