"""
GLInT experiments for the SSVM 2027 paper.

Modules:
    exp, polar, tokens: Lie group maths (exponentials, polar decomposition) and geometric tokens, shared by everything else
    glint, vit:         GLInT and the ViT baseline
    config:             loading and printing experiment configurations
    data:               datasets and dataloaders
    training:           generic training loop
Subpackages:
    splatting:          higher order Gaussian splatter used to turn images into tokens
"""
