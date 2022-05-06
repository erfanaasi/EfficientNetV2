
import timm, torch, torch.nn as nn, torchvision.utils
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import matplotlib.pyplot as plt


def visualize_feature_output(t):
    plt.imshow(feature_output[0].transpose(0, 2).sum(-1).detach().numpy())
    plt.show()


# all_model_names = timm.list_models('*')   # List all models
# model_names = timm.list_models(pretrained = True)     # List pretrained models
model = timm.create_model('efficientnetv2_rw_s', pretrained = True)
model.eval()
# print(model.get_classifier())
# print(model.get_classifier().in_features)

config = resolve_data_config({}, model = model)
transform = create_transform(**config)
print(transform)
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = transform(img).unsqueeze(0)
#
with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim = 0)


url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
urllib.request.urlretrieve(url, filename)
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


feature_output = model.forward_features(tensor)

visualize_feature_output(feature_output)
