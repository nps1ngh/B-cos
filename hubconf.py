dependencies = ['torch', 'torchvision']

from experiments.Imagenet.bcos.experiment_parameters import exps as imn_exps
from experiments.Imagenet.bcos.model import get_model as imn_get_model

from experiments.CIFAR10.bcos.experiment_parameters import exps as c10_exps
from experiments.CIFAR10.bcos.model import get_model as c10_get_model


def densenet121(pretrained=False, **kwargs):
    """ B-cos Densenet-121 model.
    pretrained (bool): load pretrained weights into the model
    kwargs (Any): update model config dict
    """
    exps = imn_exps["densenet_121"]
    exps.update(kwargs)
    exps["load_pretrained"] = pretrained
    return imn_get_model(exps)


def densenet121_cossched(pretrained=False, **kwargs):
    """ B-cos Densenet-121 model trained with a cosine learning rate schedule.
    pretrained (bool): load pretrained weights into the model
    kwargs (Any): update model config dict
    """
    exps = imn_exps["densenet_121_cossched"]
    exps.update(kwargs)
    exps["load_pretrained"] = pretrained
    return imn_get_model(exps)


def resnet34(pretrained=False, **kwargs):
    """ B-cos ResNet-34 model.
    pretrained (bool): load pretrained weights into the model
    kwargs (Any): update model config dict
    """
    exps = imn_exps["densenet_121"]
    exps.update(kwargs)
    exps["load_pretrained"] = pretrained
    return imn_get_model(exps)


def vgg11(pretrained=False, **kwargs):
    """ B-cos VGG-11 model.
    pretrained (bool): load pretrained weights into the model
    kwargs (Any): update model config dict
    """
    exps = imn_exps["densenet_121"]
    exps.update(kwargs)
    exps["load_pretrained"] = pretrained
    return imn_get_model(exps)


def inceptionv3(pretrained=False, **kwargs):
    """ B-cos Inception v3 model.
    pretrained (bool): load pretrained weights into the model
    kwargs (Any): update model config dict
    """
    exps = imn_exps["densenet_121"]
    exps.update(kwargs)
    exps["load_pretrained"] = pretrained
    return imn_get_model(exps)


def cifar10_bcosnet(pretrained=False, b=2, **kwargs):
    """ B-cos 9L model for CIFAR-10.
    pretrained (bool): load pretrained weights into the model
    b (float|int): The B exponent. Can be one of 
        [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5]. Default: 2
    kwargs (Any): update model config dict
    """
    if int(b) == b:
        b = int(b)  # removes .0 if integer float
    
    model_name = "9L-M2-B{}".format(b)
    if model_name not in c10_exps:
        raise ValueError(
            "Model '{}' not found! Maybe the passed b exponent is wrong?".format(model_name)
        )
    exps = c10_exps[model_name]
    exps.update(kwargs)
    exps["load_pretrained"] = pretrained
    return c10_get_model(exps)