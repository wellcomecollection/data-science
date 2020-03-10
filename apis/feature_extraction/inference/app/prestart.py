from torchvision.models.vgg import vgg16

feature_extractor = vgg16(pretrained=True, progress=False)
