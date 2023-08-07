import torch
from torch.nn.functional import pad


def expand_to_bigger_tensor(src: torch.Tensor, new_dim_size: int, index: torch.Tensor):
    """
    Expands the given tensor src in dimension=2 such that the src values are now present in locations given by the index.

    B: Batch size
    S: Sequence length

    Args:
        src (torch.Tensor): Input tensor of shape (B, S, len(index), 3, 3)
        new_dim_size (int): New size of dim=2. Previously len(index).
        index (torch.Tensor): The rotation matrices already included in src.

    Returns:
        torch.Tensor: Output tensor that contains the src tensor at specified indexes in dim=2 and identity matrices where no input was given.
        Output tensor has shape (B, S, new_dim_size, 3, 3)
    """
    full = torch.eye(3).repeat(src.size(0), src.size(1), new_dim_size, 1, 1)
    for i in range(src.size(0)):
        for j in range(src.size(1)):
            for k in range(src.size(2)):
                full[i][j][index[k]] = src[i][j][k]
    return full


def unfold_to_sliding_windows(x: torch.Tensor, past_frames=20, future_frames=5) -> torch.Tensor:
    """
    Prepares sliding windows of length (past_frames + future_frames + 1) for batches of tensors.
    For example if given input=tensor([[[1],[2],[3],[4],[5]]]), past_frames=1, future_frames=1 will return the output tensor([[[[1,1,2]],[[1,2,3]],[[2,3,4]],[[3,4,5]],[[4,5,5]]]])

    B: Batch size
    S: Sequence length
    F: Features

    Args:
        x (torch.Tensor): Tensor to unfold of shape (B, S, F)
        past_frames (int, optional): Number of frames in the past per sliding window. Defaults to 20.
        future_frames (int, optional): Number of frames in the future per sliding window. Defaults to 5.

    Returns:
        torch.Tensor: Tensor that contains windows of shape (B, S, F, (past_frames + future_frames + 1))
    """
    padded = pad(x, pad=(0, 0, past_frames, future_frames), mode="replicate")
    return padded.unfold(1, past_frames + future_frames + 1, 1)


def gaussian_noise(x: torch.Tensor, sigma=0.1, is_relative_detach=True):
    """
    Gaussian noise regularizer.

    Based on https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887/4

    Args:
        x (torch.Tensor): The tensor to add noise to.
        sigma (float, optional): Relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector. Defaults to 0.1.
        is_relative_detach (bool, optional): Whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values. Defaults to True.

    Returns:
        torch.Tensor: The given tensor with added sampled noise.
    """
    noise = torch.tensor(0.0).to(x.device)
    if sigma != 0:
        scale = sigma * x.detach() if is_relative_detach else sigma * x
        sampled_noise = noise.repeat(*x.size()).normal_() * scale
        x = x + sampled_noise
    return x
