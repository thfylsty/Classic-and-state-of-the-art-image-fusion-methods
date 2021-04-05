# coding: utf-8
# useful transforms in the implementation of our IFCNN

def denorm(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def norms(mean=[0, 0, 0], std=[1, 1, 1], *tensors):
    out_tensors = []
    for tensor in tensors:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        out_tensors.append(tensor)
    return out_tensors

def detransformcv2(img, mean=[0, 0, 0], std=[1, 1, 1]):
    img = denorm(mean, std, img).clamp_(0, 1) * 255
    if img.is_cuda:
        img = img.cpu().data.numpy().astype('uint8')
    else:
        img = img.numpy().astype('uint8')
    img = img.transpose([1,2,0])
    return img
