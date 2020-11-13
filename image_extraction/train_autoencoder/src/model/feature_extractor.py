from torch import nn
from torchvision.models import vgg16
from weco_datascience.logging import get_logger

log = get_logger(__name__)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        log.info("Initialising VGG feature extractor")
        self.vgg = vgg16(pretrained=True, progress=False)
        # cut off the classifier head in order to extract hidden features,
        # not binary labels
        self.vgg.classifier = self.vgg.classifier[:4]
        self.output_dim = self.vgg.classifier[-1].out_features

    def forward(self, x):
        return self.vgg(x)
