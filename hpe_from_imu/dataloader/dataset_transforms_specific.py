from .dataset_transforms_in import AddNoise, DownsampleInput

class DownsampleInputHalf(DownsampleInput):
    def __init__(self):
        super().__init__(keep_nth=2)

    def __call__(self, acc, ori, pose):
        return super().__call__(acc, ori, pose)


class DownsampleInputThird(DownsampleInput):
    def __init__(self):
        super().__init__(keep_nth=3)

    def __call__(self, acc, ori, pose):
        return super().__call__(acc, ori, pose)


class Add20Noise(AddNoise):
    def __init__(self):
        super().__init__(sigma=0.2)

    def __call__(self, acc, ori, pose):
        return super().__call__(acc, ori, pose)


class Add05Noise(AddNoise):
    def __init__(self):
        super().__init__(sigma=0.05)

    def __call__(self, acc, ori, pose):
        return super().__call__(acc, ori, pose)
