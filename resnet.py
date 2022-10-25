import torch
import urllib
from PIL import Image
from torchvision import transforms

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)

model.eval()

url, filename = ('https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg', 'dog.jpg')

try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

input_img = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# create a mini-batch as expected by the model
input_tesnor = preprocess(input_img)
input_batch = input_tesnor.unsqueeze(0)  # unsqueeze turns an n.d. tensor into an (n+1).d. one by adding an extra
                                         # dimension of depth 1.

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)