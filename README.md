# Facial Expression Recognition

The FER-2013 dataset, short for Facial Expression Recognition 2013 dataset, created for the Facial Expression Recognition Challenge, consists of 35887 images with a resolution of 48x48 pixels, 28709 images for training and 7178 images for testing, depicting facial expressions corresponding to seven different emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral.  This dataset is often used to train and evaluate machine learning models for facial expression recognition tasks.

## Install Packages and Dataset


```python
!git clone https://github.com/kemalkilicaslan/Facial_Expression_Recognition.git
!pip install timm
```

    Cloning into 'Facial_Expression_Recognition'...
    remote: Enumerating objects: 34063, done.[K
    remote: Total 34063 (delta 0), reused 0 (delta 0), pack-reused 34063[K
    Receiving objects: 100% (34063/34063), 67.10 MiB | 16.03 MiB/s, done.
    Resolving deltas: 100% (5/5), done.
    Updating files: 100% (35898/35898), done.
    Collecting timm
      Downloading timm-0.9.12-py3-none-any.whl (2.2 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.2/2.2 MB[0m [31m20.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: torch>=1.7 in /usr/local/lib/python3.10/dist-packages (from timm) (2.1.0+cu121)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (from timm) (0.16.0+cu121)
    Requirement already satisfied: pyyaml in /usr/local/lib/python3.10/dist-packages (from timm) (6.0.1)
    Requirement already satisfied: huggingface-hub in /usr/local/lib/python3.10/dist-packages (from timm) (0.20.2)
    Requirement already satisfied: safetensors in /usr/local/lib/python3.10/dist-packages (from timm) (0.4.1)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.7->timm) (3.13.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.7->timm) (4.5.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.7->timm) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.7->timm) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.7->timm) (3.1.2)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.7->timm) (2023.6.0)
    Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.7->timm) (2.1.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->timm) (2.31.0)
    Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->timm) (4.66.1)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub->timm) (23.2)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision->timm) (1.23.5)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision->timm) (9.4.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.7->timm) (2.1.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->timm) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->timm) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->timm) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->huggingface-hub->timm) (2023.11.17)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.7->timm) (1.3.0)
    Installing collected packages: timm
    Successfully installed timm-0.9.12


## Imports


```python
import numpy as np
import matplotlib.pyplot as plt
import torch
```

## Configurations


```python
train_img_folder_path = '/content/Facial_Expression_Recognition/train'
test_img_folder_path = '/content/Facial_Expression_Recognition/test'
```


```python
lr = 0.001
batch_size=32
epochs=20
device = 'cuda'
model_name='efficientnet'
```

## Load Dataset


```python
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
```


```python
train_aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=(-20, +20)),
    T.ToTensor() # PIL / numpy arr -> torch tensor -> (h, w,c) -> (c, h, w)
])

test_aug= T.Compose([
    T.ToTensor()
])
```


```python
trainset = ImageFolder(train_img_folder_path, transform = train_aug)
testset = ImageFolder(test_img_folder_path, transform = test_aug)

print(f"Total no. of examples in trainset : {len(trainset)}")
print(f"Total no. of examples in testset : {len(testset)}")
```

    Total no. of examples in trainset : 28709
    Total no. of examples in testset : 7178



```python
print(trainset.class_to_idx)
```

    {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}



```python
image, label = trainset[40]
plt.imshow(image.permute(1,2,0)) #(h, w, c)
plt.title(label)
```




    Text(0.5, 1.0, '0')




    
![trainset40](https://github.com/kemalkilicaslan/Facial_Expression_Recognition/blob/main/trainset40.png)
    



```python
image, label = testset[40]
plt.imshow(image.permute(1,2,0)) #(h, w, c)
plt.title(label)
```




    Text(0.5, 1.0, '0')




    
![testset40](https://github.com/kemalkilicaslan/Facial_Expression_Recognition/blob/main/testset40.png)
    



```python
from torch.utils.data import DataLoader
```


```python
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size, batch_size)

print(f"Total no. of batches in trainloader : {len(trainloader)}")
print(f"Total no. of batches in testloader : {len(testloader)}")
```

    Total no. of batches in trainloader : 898
    Total no. of batches in testloader : 225



```python
for images, labels in trainloader:
  break;

print(f"One image batch shape : {images.shape}")
print(f"One label batch shape : {labels.shape}")
```

    One image batch shape : torch.Size([32, 3, 48, 48])
    One label batch shape : torch.Size([32])


## Create Model


```python
import timm
from torch import nn
```


```python
class FaceRecognitionModel(nn.Module):

  def __init__(self):
    super(FaceRecognitionModel, self).__init__()

    self.eff_net = timm.create_model('efficientnet_b0', pretrained = True, num_classes = 7)

  def forward(self, images, labels = None):
    logits = self.eff_net(images)

    if labels != None:
      loss = nn.CrossEntropyLoss()(logits, labels)
      return logits, loss

    return logits
```


```python
model = FaceRecognitionModel()
model.to(device);
```

    /usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:88: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(



    model.safetensors:   0%|          | 0.00/21.4M [00:00<?, ?B/s]


## Create Train and Eval Function


```python
from tqdm import tqdm
```


```python
def multiclass_accuracy(y_pred,y_true):
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))
```


```python
def train_func(model, dataloader, optimizer, epoch):
  model.train()
  total_loss = 0.0
  total_acc = 0.0
  tk = tqdm(dataloader, desc = "epoch" + "[train]" + str(epoch + 1) +  "/" + str(epochs))

  for t, data in enumerate(tk):
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    optimizer.zero_grad()
    logits, loss = model(images, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    total_acc += multiclass_accuracy(logits, labels)
    tk.set_postfix({'loss' : '%6f' %float(total_loss / (t + 1)), 'acc' : '%6f' %float(total_acc / (t + 1)),})

  return total_loss / len(dataloader), total_acc / len(dataloader)
```


```python
def eval_func(model, dataloader, epoch):
  model.eval()
  total_loss = 0.0
  total_acc = 0.0
  tk = tqdm(dataloader, desc = "epoch" + "[train]" + str(epoch + 1) +  "/" + str(epochs))

  for t, data in enumerate(tk):
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    logits, loss = model(images, labels)

    total_loss += loss.item()
    total_acc += multiclass_accuracy(logits, labels)
    tk.set_postfix({'loss' : '%6f' %float(total_loss / (t + 1)), 'acc' : '%6f' %float(total_acc / (t + 1)),})

  return total_loss / len(dataloader), total_acc / len(dataloader)
```

## Model Training


```python
optimizer = torch.optim.Adam(model.parameters(),lr = lr)
```


```python
best_test_loss = np.Inf

for i in range(epochs):
  train_loss, train_acc = train_func(model, trainloader, optimizer, i)
  test_loss, test_acc = eval_func(model, testloader, i)

  if test_loss < best_test_loss:
    torch.save(model.state_dict(), 'best-weights.pt')
    print("Saved best-weights")
    best_test_loss = test_loss
```

    epoch[train]1/20: 100%|██████████| 898/898 [00:53<00:00, 16.93it/s, loss=1.826254, acc=0.382893]
    epoch[train]1/20: 100%|██████████| 225/225 [00:06<00:00, 35.25it/s, loss=1.361929, acc=0.473222]


    Saved best-weights


    epoch[train]2/20: 100%|██████████| 898/898 [00:51<00:00, 17.60it/s, loss=1.293115, acc=0.508797]
    epoch[train]2/20: 100%|██████████| 225/225 [00:07<00:00, 31.86it/s, loss=1.169665, acc=0.551556]


    Saved best-weights


    epoch[train]3/20: 100%|██████████| 898/898 [00:51<00:00, 17.57it/s, loss=1.193605, acc=0.547146]
    epoch[train]3/20: 100%|██████████| 225/225 [00:06<00:00, 35.52it/s, loss=1.167089, acc=0.560250]


    Saved best-weights


    epoch[train]4/20: 100%|██████████| 898/898 [00:52<00:00, 17.12it/s, loss=1.141166, acc=0.568875]
    epoch[train]4/20: 100%|██████████| 225/225 [00:07<00:00, 31.19it/s, loss=1.124356, acc=0.575806]


    Saved best-weights


    epoch[train]5/20: 100%|██████████| 898/898 [00:51<00:00, 17.47it/s, loss=1.098124, acc=0.589880]
    epoch[train]5/20: 100%|██████████| 225/225 [00:06<00:00, 34.90it/s, loss=1.103011, acc=0.590722]


    Saved best-weights


    epoch[train]6/20: 100%|██████████| 898/898 [00:51<00:00, 17.40it/s, loss=1.066784, acc=0.599374]
    epoch[train]6/20: 100%|██████████| 225/225 [00:07<00:00, 31.27it/s, loss=1.075633, acc=0.594278]


    Saved best-weights


    epoch[train]7/20: 100%|██████████| 898/898 [00:52<00:00, 17.19it/s, loss=1.038542, acc=0.609869]
    epoch[train]7/20: 100%|██████████| 225/225 [00:06<00:00, 34.63it/s, loss=1.037396, acc=0.616028]


    Saved best-weights


    epoch[train]8/20: 100%|██████████| 898/898 [00:51<00:00, 17.56it/s, loss=1.014691, acc=0.620031]
    epoch[train]8/20: 100%|██████████| 225/225 [00:07<00:00, 31.70it/s, loss=1.066045, acc=0.598389]
    epoch[train]9/20: 100%|██████████| 898/898 [00:50<00:00, 17.68it/s, loss=0.987903, acc=0.627499]
    epoch[train]9/20: 100%|██████████| 225/225 [00:06<00:00, 34.18it/s, loss=1.051137, acc=0.605833]
    epoch[train]10/20: 100%|██████████| 898/898 [00:51<00:00, 17.28it/s, loss=0.959626, acc=0.643555]
    epoch[train]10/20: 100%|██████████| 225/225 [00:06<00:00, 33.84it/s, loss=1.057377, acc=0.609861]
    epoch[train]11/20: 100%|██████████| 898/898 [00:50<00:00, 17.73it/s, loss=0.927390, acc=0.653035]
    epoch[train]11/20: 100%|██████████| 225/225 [00:07<00:00, 31.47it/s, loss=1.027363, acc=0.627861]


    Saved best-weights


    epoch[train]12/20: 100%|██████████| 898/898 [00:50<00:00, 17.64it/s, loss=0.900384, acc=0.661546]
    epoch[train]12/20: 100%|██████████| 225/225 [00:06<00:00, 35.34it/s, loss=1.001174, acc=0.634056]


    Saved best-weights


    epoch[train]13/20: 100%|██████████| 898/898 [00:51<00:00, 17.33it/s, loss=0.867132, acc=0.675550]
    epoch[train]13/20: 100%|██████████| 225/225 [00:07<00:00, 31.66it/s, loss=0.992073, acc=0.637111]


    Saved best-weights


    epoch[train]14/20: 100%|██████████| 898/898 [00:50<00:00, 17.68it/s, loss=0.840561, acc=0.688544]
    epoch[train]14/20: 100%|██████████| 225/225 [00:06<00:00, 34.59it/s, loss=1.003687, acc=0.631556]
    epoch[train]15/20: 100%|██████████| 898/898 [00:50<00:00, 17.65it/s, loss=0.812158, acc=0.697112]
    epoch[train]15/20: 100%|██████████| 225/225 [00:07<00:00, 31.50it/s, loss=1.012400, acc=0.644111]
    epoch[train]16/20: 100%|██████████| 898/898 [00:51<00:00, 17.42it/s, loss=0.785455, acc=0.706890]
    epoch[train]16/20: 100%|██████████| 225/225 [00:07<00:00, 31.58it/s, loss=1.022393, acc=0.636167]
    epoch[train]17/20: 100%|██████████| 898/898 [00:51<00:00, 17.46it/s, loss=0.753879, acc=0.717198]
    epoch[train]17/20: 100%|██████████| 225/225 [00:06<00:00, 35.13it/s, loss=1.094419, acc=0.613222]
    epoch[train]18/20: 100%|██████████| 898/898 [00:50<00:00, 17.70it/s, loss=0.725911, acc=0.730310]
    epoch[train]18/20: 100%|██████████| 225/225 [00:07<00:00, 28.97it/s, loss=1.009356, acc=0.655389]
    epoch[train]19/20: 100%|██████████| 898/898 [00:50<00:00, 17.74it/s, loss=0.700277, acc=0.741606]
    epoch[train]19/20: 100%|██████████| 225/225 [00:06<00:00, 36.04it/s, loss=0.991358, acc=0.654111]


    Saved best-weights


    epoch[train]20/20: 100%|██████████| 898/898 [00:50<00:00, 17.68it/s, loss=0.672303, acc=0.749325]
    epoch[train]20/20: 100%|██████████| 225/225 [00:07<00:00, 31.96it/s, loss=1.070224, acc=0.633722]


## Result


```python
def view_classify(img, ps):

    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    ps = ps.data.cpu().numpy().squeeze()
    img = img.numpy().transpose(1,2,0)

    fig, (ax1, ax2) = plt.subplots(figsize=(5,9), ncols=2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None
```


```python
def predict(model, dataloader, num_class=7):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        logits = model(images)
        ps = torch.softmax(logits, dim=1)
        top_p, top_class = ps.topk(1, dim=1)

    for i in range(num_class):
        for j, image in enumerate(images):
            label = labels[j].item()
            predicted_class = top_class[j].item()
            predicted_prob = top_p[j].item()

            predicted_label = trainset.classes[predicted_class]
            true_label = trainset.classes[label]

            if true_label == trainset.classes[i]:
                print(f"True Label: {true_label}, Predicted Label: {predicted_label}, Predicted Probability: {predicted_prob}")
                view_classify(image.cpu(), ps[j])
                break  # Show only the first prediction for each class

# Example usage
predict(model, testloader, num_class=7)
```

    True Label: angry, Predicted Label: disgust, Predicted Probability: 0.7471802830696106
    True Label: disgust, Predicted Label: disgust, Predicted Probability: 0.9846369028091431
    True Label: fear, Predicted Label: fear, Predicted Probability: 0.5837821364402771
    True Label: happy, Predicted Label: happy, Predicted Probability: 0.9693137407302856
    True Label: neutral, Predicted Label: happy, Predicted Probability: 0.8458672761917114
    True Label: sad, Predicted Label: angry, Predicted Probability: 0.6017364263534546
    True Label: surprise, Predicted Label: surprise, Predicted Probability: 0.6609428524971008



    
![angry](https://github.com/kemalkilicaslan/Facial_Expression_Recognition/blob/main/angry.png)

    



    
![disgust](https://github.com/kemalkilicaslan/Facial_Expression_Recognition/blob/main/disgust.png)
    



    
![fear](https://github.com/kemalkilicaslan/Facial_Expression_Recognition/blob/main/fear.png)

    



    
![happy](https://github.com/kemalkilicaslan/Facial_Expression_Recognition/blob/main/happy.png)

    



    
![neutral](https://github.com/kemalkilicaslan/Facial_Expression_Recognition/blob/main/neutral.png)

    



    
![sad](https://github.com/kemalkilicaslan/Facial_Expression_Recognition/blob/main/sad.png)

    



    
![surprise](https://github.com/kemalkilicaslan/Facial_Expression_Recognition/blob/main/surprise.png)