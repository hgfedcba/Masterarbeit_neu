from torch.nn.functional import relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink, \
    gumbel_softmax, log_softmax, hardsigmoid
from torch import tanh, sigmoid
from torch.optim import Adam, lr_scheduler

activation_functions = [tanh, sigmoid, relu, hardtanh, relu6, elu, selu, celu, leaky_relu, rrelu, gelu, logsigmoid, hardshrink, tanhshrink, softsign, softplus, softmin, softmax, softshrink,
                        gumbel_softmax, log_softmax, hardsigmoid]

optimizer = [Adam]

"""
Guide to the Definitions below and in Model Definitions:
1. I have global variables foo_dict and foos. The second should always contain all foos that are used in the current runntime. The first should always be a dictonary that maps any foo to
    an expressive description
2. When ConfigInitilaizer is called it is decided what foos are needed. For every needed foo the corresponding add_foo function is called. This function adds foo to the dictonary and to foos.
    If only a single foo is needed i don't need to bother with foos and can use the value returned from add_foo directly.
    
I need this for pretrainfunc, lr_decay, sigma, mu, g
"""


def add_am_put_default_pretrain(K, slope_length):
    def am_put_default_pretrain(x):
        b = K
        a = b - slope_length
        out = (b - x) / slope_length + relu(x - b) / slope_length - relu(a - x) / slope_length
        return out
    f = am_put_default_pretrain
    pretrain_functions.append(f)
    pretrain_func_dict[f] = "am_put_default_pretrain"
    return f


def add_am_call_default_pretrain(K, slope_length):
    def am_call_default_pretrain(x):
        b = K
        a = b - slope_length
        out = (b - x) / slope_length + relu(x - b) / slope_length - relu(a - x) / slope_length
        out = -out+2*K
        return out

    f = am_call_default_pretrain
    pretrain_functions.append(f)
    pretrain_func_dict[f] = "am_call_default_pretrain"
    return f


pretrain_functions = []
pretrain_func_dict = {}


lr_decay_algs = []
lr_decay_dict = {}


def add_multiplicative_lr_sheduler(factor):
    f = lr_scheduler.MultiplicativeLR(optimizer, factor)
    lr_decay_algs[f] = "multiplicative lr_sheduler with factor " + str(factor)
    lr_decay_algs.append(f)
    return f

"""
# has to be done every time a new config is picked
def configure_pretrain_functions(K, slope_length):
    global pretrain_functions
    global pretrain_func_dict
    pretrain_functions = [return_am_put_default_pretrain(K, slope_length)]
    pretrain_func_dict = {
        pretrain_functions[0]: "am_put_default_pretrain"
    }
"""

activation_func_dict = {
    # TODO: commented functions might work in theory
    tanh          : "tanh",
    sigmoid       : "sigmoid",
    # threshold     : "threshold",
    relu          : "relu",
    hardtanh      : "hardtanh",
    # hardswish     : "hardswish",
    relu6         : "relu6",
    elu           : "elu",
    selu          : "selu",
    celu          : "celu",
    leaky_relu    : "leaky_relu",
    # prelu         : "prelu",
    rrelu         : "rrelu",
    # glu           : "glu",
    gelu          : "gelu",
    logsigmoid    : "logsigmoid",
    hardshrink    : "hardshrink",
    tanhshrink    : "tanhshrink",
    softsign      : "softsign",
    softplus      : "softplus",
    softmin       : "softmin",
    softmax       : "softmax",
    softshrink    : "softshrink",
    gumbel_softmax: "gumbel_softmax",
    log_softmax   : "log_softmax",
    hardsigmoid   : "hardsigmoid",
}

optimizer_dict = {
    Adam: "Adam"
}
