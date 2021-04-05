import torch

EPSILON = 1e-10


# addition fusion strategy
def addition_fusion(tensor1, tensor2):
    return (tensor1 + tensor2)/2


# attention fusion strategy, average based on weight maps
def attention_fusion_weight(tensor1, tensor2):
    # avg, max, nuclear
    f_spatial = spatial_fusion(tensor1, tensor2)
    tensor_f = f_spatial
    return tensor_f


def spatial_fusion(tensor1, tensor2, spatial_type='sum'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial




