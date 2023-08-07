import collections

import torch.nn

"""
Adapted from https://gist.github.com/ferrine/89d739e80712f5549e44b2c2435979ef
"""


class Builder(object):
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)

    def __call__(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e

    def add_namespace(self, namespace, index=-1):
        if index >= 0:
            namespaces = self._namespace.maps
            namespaces.insert(index, namespace)
            self._namespace = collections.ChainMap(*namespaces)
        else:
            self._namespace = self._namespace.new_child(namespace)


def build_network(architecture, builder=Builder(torch.nn.__dict__)):
    """
    Configuration for feedforward networks is list by nature. We can write 
    this in simple data structures. In yaml format it can look like:
    .. code-block:: yaml
        architecture:
            - Conv2d:
                args: [3, 16, 25]
                stride: 1
                padding: 2
            - ReLU:
                inplace: true
            - Conv2d:
                args: [16, 25, 5]
                stride: 1
                padding: 2
    Note, that each layer is a list with a single dict, this is for readability.
    For example, `builder` for the first block is called like this:
    .. code-block:: python
        first_layer = builder("Conv2d", *[3, 16, 25], **{"stride": 1, "padding": 2})
    the simpliest ever builder is just the following function:
    .. code-block:: python
         def build_layer(name, *args, **kwargs):
            return layers_dictionary[name](*args, **kwargs)

    Some more advanced builders catch exceptions and format them in debuggable way or merge 
    namespaces for name lookup

    .. code-block:: python

        extended_builder = Builder(torch.nn.__dict__, mynnlib.__dict__)
        net = build_network(architecture, builder=extended_builder)

    """
    layers = []
    for block in architecture:
        assert len(block) == 1
        name, kwargs = list(block.items())[0]
        if kwargs is None:
            kwargs = {}
        args = kwargs.pop("args", [])
        layers.append(builder(name, *args, **kwargs))
    return torch.nn.Sequential(*layers)


def build_transforms(transforms, builder=Builder(torch.nn.__dict__)):
    transforms_in = {}
    for ds_name in transforms["in"]:
        transforms_in[ds_name] = [builder(name)
                                  for name in transforms["in"][ds_name]]
    transforms_out = [builder(name) for name in transforms["out"]]
    return transforms_in, transforms_out


def build_parameters(parameters, builder=Builder(torch.nn.__dict__)):
    hyper_parameters = {}
    for k, v in parameters.items():
        hyper_parameters[k] = build_parameter(v, builder)
    return hyper_parameters


def build_parameter(parameter, builder=Builder(torch.nn.__dict__)):
    if type(parameter) is list and type(parameter[0]) is dict:
        parameter_list = []
        for block in parameter:
            assert len(block) == 1
            name, kwargs = list(block.items())[0]
            if kwargs is None:
                kwargs = {}
            args = kwargs.pop("args", [])
            parameter_list.append(builder(name, *args, **kwargs))
        return parameter_list
    return parameter
