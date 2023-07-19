import fnx, jax
import numpy as np

color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])
def preprocess(color):
    color = color / 255.0
    color = (color - color_mean) / color_std
    return color

resnet_config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.batchnorm),
    fnx.intercept.replace(fnx.act, jax.nn.relu),
    fnx.config.pytorch,
)

detr_config = fnx.intercept.chain(
    fnx.intercept.replace(fnx.norm, fnx.layernorm),
    fnx.intercept.replace(fnx.act, jax.nn.relu),
    fnx.config.pytorch,
)

class Builder:
    def __init__(self, resnet_variant, url):
        self.resnet = vars(fnx.resnet)[resnet_variant]
        self.url = url
        self.preprocess = preprocess

    @fnx.module(name="model", is_unbound_method=True)
    def __call__(self, x):
        def model_fn(x):
            with resnet_config:
                x = self.resnet(x)

            # 1. They use mlp_ratio=8
            with fnx.scope("detr"), detr_config, fnx.intercept.defaults(fnx.detr.mlp, mlp_ratio=8):
                x = fnx.linear(x, channels=256, bias=True)
                x = fnx.detr.encode(x, depth=6, heads=8)
                x = fnx.detr.decode(x, depth=6, heads=8, num_queries=100)
                x = fnx.norm(x) # 2. They use norm after decoder, even with post-norm transformer

                with fnx.scope("detector"):
                    probs = fnx.linear(x, channels=92, bias=True, name="logits")
                    probs = jax.nn.softmax(probs, axis=-1)
                    with fnx.scope("bbox"):
                        bbox = x
                        for block_index in range(2):
                            bbox = fnx.linear(bbox, bias=True, name=f"block{block_index + 1}")
                            bbox = fnx.act(bbox)
                        bbox = fnx.linear(bbox, channels=4, bias=True, name="out")
                        bbox = jax.nn.sigmoid(bbox)
            return probs, bbox

        file = fnx.pretrained.weights.download(self.url)
        weights = fnx.pretrained.weights.load_pytorch(file)
        # 2. Transpose weights
        weights["query_embed.weight"] = np.transpose(weights["query_embed.weight"], (1, 0))
        # 3. They use combined QKV matrix in attention, we split into QK, V for self-attention and Q, K, V for cross-attention
        # 4. They predict bbox in w-h order, we predict bbox in h-w order
        for key in list(weights.keys()):
            if key.endswith("self_attn.in_proj_weight"):
                qk, v = np.split(weights[key], indices_or_sections=[512], axis=-1)
                del weights[key]
                key = key[:-len("in_proj_weight")]
                weights[f"{key}qk.weight"] = qk
                weights[f"{key}v.weight"] = v
            elif key.endswith("self_attn.in_proj_bias"):
                qk, v = np.split(weights[key], indices_or_sections=[512], axis=-1)
                del weights[key]
                key = key[:-len("in_proj_bias")]
                weights[f"{key}qk.bias"] = qk
                weights[f"{key}v.bias"] = v
            elif key.endswith("multihead_attn.in_proj_weight"):
                q, k, v = np.split(weights[key], indices_or_sections=3, axis=-1)
                del weights[key]
                key = key[:-len("in_proj_weight")]
                weights[f"{key}q.weight"] = q
                weights[f"{key}k.weight"] = k
                weights[f"{key}v.weight"] = v
            elif key.endswith("multihead_attn.in_proj_bias"):
                q, k, v = np.split(weights[key], indices_or_sections=3, axis=-1)
                del weights[key]
                key = key[:-len("in_proj_bias")]
                weights[f"{key}q.bias"] = q
                weights[f"{key}k.bias"] = k
                weights[f"{key}v.bias"] = v
            elif key == "bbox_embed.layers.2.weight" or key == "bbox_embed.layers.2.bias":
                weights[key][..., (0, 1, 2, 3)] = weights[key][..., (1, 0, 3, 2)]

        model_fn = fnx.pretrained.weights.init(model_fn, weights, hints=[("/reduce/norm", ".bn1"), ("/spatial/embed2", ".bn2"), ("/self_attention/norm", ".norm1"), ("/cross_attention/norm", ".norm2"), ("/q/", ".q."), ("/k/", ".k."), ("/v/", ".v."), ("shortcut/norm", "downsample.1")])

        return model_fn(x)



detr_resnet_v1_50_coco = Builder("resnet_v1_50", f"https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth")
detr_resnet_v1_101_coco = Builder("resnet_v1_101", f"https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth")
