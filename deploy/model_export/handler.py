from ts.torch_handler.vision_handler import VisionHandler
from torchvision import transforms as T
import base64
from PIL import Image
import io
import torch
import imgaug.augmenters as iaa
import numpy as np


class MetricLearningHandler(VisionHandler):
    
    image_processing = T.Compose([     
    iaa.Sequential([
        iaa.size.Resize((240, 240), interpolation='cubic')
    ]).augment_image,     
    T.ToTensor()
    ])
    
    def preprocess(self, data):
        """The preprocess function converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        images = []

        for row in data:

            image = row.get("data") or row.get("body")
        
            image = base64.b64decode(image)
            
            if isinstance(image, (bytearray, bytes)):
                image = np.array(Image.open(io.BytesIO(image)))
                image = self.image_processing(image)
                
            else:
                image = torch.FloatTensor(image)

            images.append(image)

        return torch.stack(images).to(self.device)
    
    def postprocess(self, data):
        
        embedding = data.cpu().numpy().tolist()
        return [{'embed': embedding}]