import math
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F

def convSize(Hin,Win,K,S = [1,1],P = [0,0],D = [1,1]):
  Hout = math.floor((Hin + 2*P[0] - D[0]*(K[0]-1) - 1)/S[0] + 1)
  Wout = math.floor((Win + 2*P[1] - D[1]*(K[1]-1) - 1)/S[1] + 1)
  return Hout, Wout



### Plotting
def val_plot(config,model, dataset, device,name):
    model.eval()
    #metric = SegmentationMetric(19)
    #metric.reset()
    lsit_pixAcc = []
    list_mIoU = []
    rand = np.random.randint(0,len(dataset))
    image_standard,target,_,_,_ = dataset[rand]

    mean_standard = np.array([0.485, 0.456, 0.406],dtype = np.float32)
    std_standard = np.array([0.229, 0.224, 0.225],dtype = np.float32)

    with torch.no_grad():
      fig = plt.figure(figsize=(15,5))
      ax = fig.add_subplot(1, 3, 1)
      ax.set_title('Image')
      image_original = (image_standard.copy().transpose(1,2,0)*std_standard + mean_standard)
      ax.imshow(image_original)
      ax = fig.add_subplot(1, 3, 3)
      target[target == 255] = 0
      ax.imshow(target)
      #print(f'Unique value in original target: {np.unique(target)}')
      ax.set_title('Target')

      ## Evaluating the output
      if 'bisenet' in name:
        image_standard = torch.from_numpy(image_standard).unsqueeze(0).to(device)
        outputs = model(image_standard)
        output = outputs[config.test.output_index_bisenet]
      
      if 'pidnet' in name:
        image_standard = torch.from_numpy(image_standard).unsqueeze(0).to(device)
        outputs = model(image_standard)
        size = target.shape
        output = F.interpolate(
                      outputs[config.test.output_index_pidnet], size[-2:],
                      mode='bilinear', align_corners=True
                )
        
      if 'segformer' in name:
        image_standard = torch.from_numpy(image_standard).unsqueeze(0).to(device)
        outputs = model(image_standard)
        size = target.shape
        output = F.interpolate(
                      outputs.logits, size[-2:],
                      mode='bilinear', align_corners=True


                )
      if 'icnet' in name:
        image_standard = torch.from_numpy(image_standard).unsqueeze(0).to(device)
        outputs = model(image_standard)
        output = outputs[config.test.output_index_icnet]
      ax = fig.add_subplot(1, 3, 2)
      ax.set_title('Prediction')
      outputs = output.argmax(1).squeeze(0)
      #print(f'Unique value in predicted target: {torch.unique(outputs)}')
      ax.imshow(outputs.cpu().numpy())
      plt.show()
