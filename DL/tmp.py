def remove_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

# safeload some parameters from pretrained model molscribe
molscribe_path = "/home/wangzixu/MolScribe/ckpts/swin_base_char_aux_200k.pth"
pretrained_dict = torch.load(molscribe_path, map_location=torch.device('cpu'))["encoder"]
model.load_state_dict(remove_prefix(pretrained_dict), strict=False)