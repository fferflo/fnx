import haiku as hk

from . import intercept
from .module import module, wrap_module, shared, sow, Reaper, init, ModuleCallContext
from .util import remove_unused, flatten, unflatten, subshape, MemoryTracker
from . import util

def __getattr__(name):
    if name == "is_training":
        return util.is_training()
    else:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")
set_is_training = util.set_is_training

scope = lambda name: hk.experimental.name_scope(name)
param = lambda *args, **kwargs: hk.get_parameter(*args, **kwargs)
state = lambda *args, **kwargs: hk.get_state(*args, **kwargs)



from .nets.base import *

from . import config
from .config import norm, act

from .nets import attention, hamburger, resnet, convnext, pspnet, upernet, segformer, segnext, vit, gpt, sc_depth, mlp_mixer, stochastic, swin, efficientnet, detr

from . import pretrained
