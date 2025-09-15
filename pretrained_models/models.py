import sys
import torch
# Save the original sys.path
original_sys_path = sys.path.copy()
sys.path.append("/kaggle/working/adversarial-patch-transferability/")
from pretrained_models.ICNet.icnet import ICNet
from pretrained_models.BisNetV1.model import BiSeNetV1
from pretrained_models.BisNetV2.model import BiSeNetV2
from pretrained_models.PIDNet.model import PIDNet, get_pred_model
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch.nn.functional as F


# Restore original sys.path to avoid conflicts or shadowing
sys.path = original_sys_path

class Models():
  def __init__(self,config, model_name: str=None):
    self.config = config
    self.name = model_name if model_name is not None else config.model.name
    self.device = config.experiment.device
    self.model = None

  def get(self):
    if 'pidnet' in self.config.model.name:
      if '_s' in self.config.model.name:
        model = torch.load('/kaggle/input/pidnet_s_cityscapes_test/pytorch/default/1/PIDNet_S_Cityscapes_test.pt',map_location=self.device)
      if '_m' in self.config.model.name:
        model = torch.load('/kaggle/input/pidnet_m_cityscapes_test/pytorch/default/1/PIDNet_M_Cityscapes_test.pt',map_location=self.device)
      if '_l' in self.config.model.name:
        model = torch.load('/kaggle/input/pidnet-l-weights/PIDNet_L_Cityscapes_test.pt',map_location=self.device)
      
  
      pidnet = get_pred_model(name = self.config.model.name, num_classes = 19).to(self.device)
      if 'state_dict' in model:
          model = model['state_dict']
      model_dict = pidnet.state_dict()
      model = {k[6:]: v for k, v in model.items() # k[6:] to start after model. in key names
                          if k[6:] in model_dict.keys()}

      pidnet.load_state_dict(model)
      self.model = pidnet
      self.model.eval()
      

    if 'bisenet' in self.config.model.name:
      if '_v1' in self.config.model.name:
        model = torch.load('/kaggle/input/bisenetv1/bisenetv1.pth',map_location=self.device)
        bisenet = BiSeNetV1(19,aux_mode = 'eval').to(self.device)
        bisenet.load_state_dict(model, strict=False)
      if '_v2' in self.config.model.name:
        model = torch.load('/kaggle/input/bisenetv2-weights/bisenetv2.pth',map_location=self.device)
        bisenet = BiSeNetV2(19,aux_mode = 'eval').to(self.device)
        bisenet.load_state_dict(model, strict=False)
      self.model = bisenet
      self.model.eval()


    if 'icnet' in self.config.model.name:
      model = torch.load('/kaggle/input/icnet-wts/icnet_resnet50os8_cityscapes.pth',map_location=self.device)
      icnet = ICNet(nclass = 19).to(self.device)
      icnet.load_state_dict(model['model_state_dict'])
      self.model = icnet
      self.model.eval()

    if 'segformer' in self.config.model.name:
      feature_extractor = SegformerFeatureExtractor.from_pretrained("/kaggle/input/segformer-weights/segformer.b0.1024x1024.city.160k.pth")
      segformer = SegformerForSemanticSegmentation.from_pretrained("/kaggle/input/segformer-weights/segformer.b0.1024x1024.city.160k.pth").to(self.device)
      self.model = segformer
      self.model.eval()


  def predict(self,image_standard,size):
    image_standard = image_standard.to(self.device)
    outputs = self.model(image_standard)
    if 'pidnet' in self.config.model.name:
      output = F.interpolate(
                    outputs[self.config.test.output_index_pidnet], size[-2:],
                    mode='bilinear', align_corners=True
                )

    if 'segformer' in self.config.model.name:
      output = F.interpolate(
                    outputs.logits, size[-2:],
                    mode='bilinear', align_corners=True
                )

    if 'icnet' in self.config.model.name:
      output = outputs[self.config.test.output_index_icnet]

    if 'bisenet' in self.config.model.name:
      ## Images needs to be unnormalized and then normalized as:
      ## mean=[0.3257, 0.3690, 0.3223], std=[0.2112, 0.2148, 0.2115]
      ## The it will give 75% miou instead of 71 and to keep things simple keeping it as it
      output = outputs[self.config.test.output_index_bisenet]

    return output

    





