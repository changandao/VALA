def get_feature_extractor(features, checkpoint=None, device='cuda:0'):
    if features == 'fcn50':
        from autolabel.features import FCN50
        return FCN50()
    elif features == 'dino':
        from autolabel.features import Dino
        return Dino()
    elif features == 'lseg':
        from autolabel.features import lseg
        return lseg.LSegFE(checkpoint)
    elif features == 'clip':
        from eval.openclip_encoder import OpenCLIPNetwork
        return OpenCLIPNetwork(device)
    else:
        raise NotImplementedError()
