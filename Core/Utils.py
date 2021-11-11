import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

class ImgPreproccess():
    def __init__(self, shrink=40):
        self.transform = T.Compose([
        T.ToPILImage(),
        T.Resize(shrink,interpolation=Image.CUBIC),
        T.ToTensor()
        ]
        )
    def __call__(self,x):
        return self.transform(x)

