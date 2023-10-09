# -*- coding: utf-8 -*-

import warnings
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
# import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# %%
# !pip install git+https://github.com/openai/CLIP.git

# %%
import torch
import clip
import pandas as pd
# !pip install git+https://github.com/openai/CLIP.git
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, compose = clip.load('ViT-L/14', device = device)
# text_model = text_model.cpu()
def process(idx_val,arr):
  if idx_val=='0':
    arr.append(0)
  else:
    arr.append(1)

# %%
data = pd.read_csv('XYZ/meme_sarcasm_humor.csv')
data_test = pd.read_csv('XYZ/meme_sarcasm_humor_test.csv')

# %%
data

# %%
data_test

# %%
data.head(10)

# %%
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import clip
from PIL import Image

# %%
model, preprocess = clip.load("ViT-B/32")

# %%
# !pip install multilingual-clip

# %%
from multilingual_clip import pt_multilingual_clip
import transformers

# %%
texts = [
    'Three blind horses listening to Mozart.',
    'Älgen är skogens konung!',
    'Wie leben Eisbären in der Antarktis?',
    'Вы знали, что все белые медведи левши?'
]
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
# Load Model & Tokenizer
model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
embeddings = model.forward(texts, tokenizer)
print(embeddings.shape)

# %%
sample = data['Text_Transcription'][10]

# %%
import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# clip_model, compose = clip.load('RN50x4', device = device)
clip_model, compose = clip.load("ViT-B/32", device = device)
text_inputs = (clip.tokenize(data.Text_Transcription.values[321],truncate=True)).to(device)
print(text_inputs)

# %%
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %%
len(data)

# %%
data.img.values[1:10]

# %%
def get_data(data):
  #data = pd.read_csv(dataset_path)
  text = list(data['text'])
  img_path = list(data['file_name'])
  name = list(data['file_name'])

  # label = list(data['Level1'])
  label = list(data['sarcasm'])
  humor = list(data['humor'])
  text_features,image_features,l,Name,v,humor = [],[],[],[],[]
  # for txt,img,L,A,V in tqdm(zip(text,img_path,label,arousal,valence)):
  for txt,img,L,n,hum in tqdm(zip(text,img_path,label,name,hum)):
    try:
      #img = preprocess(Image.open('/content/drive/.shortcut-targets-by-id/1Z57L19m3ZpJ6bEPdyaIMYuI00Tc2RT1I/memes_our_dataset_hindi/my_meme_data/'+img)).unsqueeze(0).to(device)
      img = Image.open('xyz/test_images/'+img)
    except Exception as e:
      print(e)
      continue

    img = torch.stack([compose(img).to(device)])
    l.append(L)
    humor.append(hum)
    Name.append(n)
    # v.append(V)
    #txt = torch.as_tensor(txt)
    with torch.no_grad():
      temp_txt=model.forward(txt, tokenizer).detach().cpu().numpy()
      text_features.append(temp_txt)
      temp_img = clip_model.encode_image(img).detach().cpu().numpy()
      image_features.append(temp_img)

      del temp_txt
      del temp_img

      torch.cuda.empty_cache()

    del img
    # del txt
    torch.cuda.empty_cache()
  return text_features,image_features,l,Name,hum



# %%
class HatefulDataset(Dataset):

  def __init__(self,data):

    self.t_f,self.i_f,self.label,self.name,self.humor = get_data(data)
    self.t_f = np.squeeze(np.asarray(self.t_f),axis=1)
    self.i_f = np.squeeze(np.asarray(self.i_f),axis=1)
  def __len__(self):
    return len(self.label)

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    #print(idx)
    name=self.name[idx]
    label = self.label[idx]
    T = self.t_f[idx,:]
    humor = self.humor[idx]
    I = self.i_f[idx,:]
    sample = {'label':label,'processed_txt':T,'processed_img':I,'name':name,'humor':humor}
    return sample


# %%
sample_dataset = HatefulDataset(data)


# %%
len(data)

# %%
len(sample_dataset)

# %%
import pytorch_lightning as pl

# %%
torch.manual_seed(123)
t_p,v_p = torch.utils.data.random_split(sample_dataset,[9000,941])
t_p,te_p = torch.utils.data.random_split(t_p,[7000,2000])

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
class M3F_CmU(nn.Module):
    def __init__(self, num_heads, d_model, num_segments):
        super(M3F_CmU, self).__init__()

        # Define the weight matrices for linear transformations
        self.Wt_q = nn.Linear(d_model, d_model)
        self.Wt_k = nn.Linear(d_model, d_model)
        self.Wt_v = nn.Linear(d_model, d_model)
        self.Wv_q = nn.Linear(d_model, d_model)
        self.Wv_k = nn.Linear(d_model, d_model)
        self.Wv_v = nn.Linear(d_model, d_model)

        # Define the bilinear pooling weight matrix
        self.W_pooling = nn.Parameter(torch.rand((d_model, d_model)))

        # Define Layer Normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Number of attention heads and segments
        self.num_heads = num_heads
        self.num_segments = num_segments

    def forward(self, ft, fv):
        # Define the sliding window parameters
        d = 64  # Window size
        s = 32  # Stride size

        # Initialize an empty list to store the attended representations for each segment
        attended_representations = []

        # Loop over segments
        for i in range(self.num_segments):
            # Extract local portions from both textual and visual features
            t_i = ft[:, i * d:i * d + d]
            v_i = fv[:, i * d:i * d + d]

            # Linear transformations for query, key, and value
            Q_t_i = self.Wt_q(t_i)
            K_t_i = self.Wt_k(t_i)
            V_t_i = self.Wt_v(t_i)
            Q_v_i = self.Wv_q(v_i)
            K_v_i = self.Wv_k(v_i)
            V_v_i = self.Wv_v(v_i)

            # Calculate attention scores using scaled dot-product attention
            alpha_t_i = F.softmax(Q_t_i @ K_v_i.transpose(1, 0) / (d ** 0.5), dim=1)
            alpha_v_i = F.softmax(Q_v_i @ K_t_i.transpose(1, 0) / (d ** 0.5), dim=1)

            # Calculate the attended representations for both modalities
            Z_t_i = alpha_t_i @ V_v_i
            Z_v_i = alpha_v_i @ V_t_i

            # Concatenate attended representations from all attention heads
            Z_t_i_concat = torch.cat([Z_t_i[i] for i in range(self.num_heads)], dim=1)
            Z_v_i_concat = torch.cat([Z_v_i[i] for i in range(self.num_heads)], dim=1)

            # Bilinear pooling operation
            M_i = Z_t_i_concat @ self.W_pooling @ Z_v_i_concat.transpose(1, 0)

            # Layer normalization and residual connection
            o_i_tv = self.layer_norm(M_i) + t_i

            # Append the attended representation for this segment
            attended_representations.append(o_i_tv)

        # Stack the attended representations for all segments
        multimodal_representation = torch.stack(attended_representations, dim=1)

        return multimodal_representation

# %%
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from pytorch_lightning import seed_everything

# Define the teacher model
class TeacherModel(pl.LightningModule):

  def __init__(self):
    # super().__init__()
    super(TeacherModel, self).__init__()
    self.M3F_CmU = M3F_CmU(512,640,True,256,64,0.1)
    self.flatten=torch.nn.Flatten()
    # self.MFB = MFB(640,640,True,256,64,0.1)
    self.loss_fn_emotion=torch.nn.KLDivLoss(reduction='batchmean',log_target=True)
    self.encode_text = torch.nn.Linear(1280,64)
    self.fin = torch.nn.Linear(64,3)
    self.fin_img = torch.nn.Linear(512,2)
    self.flatten=torch.nn.Flatten()
    self.fin_inten = torch.nn.Linear(64,3)
    # self.mask= torch.tensor([0, 1]).cuda()
  def forward(self, x,y):
      x_,y_ = x,y
      x = x.float()
      y = y.float()
      z_ = self.M3F_CmU(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))
      # z_= torch.unsqueeze(y,axis=1)
      z = z_
      # print(z.shape)
      c_sar = self.fin(torch.squeeze(z,dim=1))
      c_hum = self.fin_inten(torch.squeeze(z,dim=1))

      # c = self.fin_img(torch.squeeze(z,dim=1))
    #   c = torch.log_softmax(c, dim=1)
      return c_sar,c_hum

# Define the student model
class StudentModel(nn.Module):
    def __init__(self, num_classes=3):
        super(StudentModel, self).__init__()
        self.M3F_CmU = M3F_CmU(512,640,True,256,64,0.1)
        self.fc2 = nn.Linear(128, num_classes)


    def forward(self, x_text, x_visual,perturbation):
        x_text = x_text.float()
        x_visual = x_visual.float()
        x = self.M3F_CmU(torch.unsqueeze(x_visual,axis=1),torch.unsqueeze(x_text,axis=1))
        # Apply M3F_CmU to obtain a multimodal representation
        multimodal_representation = self.M3F_CmU(torch.unsqueeze(x_visual,axis=1),torch.unsqueeze(x_text,axis=1))

        # Apply perturbation
        multimodal_representation += perturbation

        # Apply fully connected layers
        x = F.relu(self.fc1(multimodal_representation))
        x = self.fc2(x)

class SarcasmDetectionModel(pl.LightningModule):
    def __init__(self, teacher_model, student_model, distillation_weight=0.5):
        super(SarcasmDetectionModel, self).__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_weight = distillation_weight
        self.automatic_optimization = False # add this line
        self.alpha=0.5
    def forward(self, x_text, x_visual):
        return self.student_model(x_text, x_visual)

    @staticmethod
    def temperature_scaled_softmax(logits, temperature):
      return F.softmax(logits / temperature, dim=1)

    def training_step(self, batch, batch_idx):
        # x, y, label, name, humor = val_batch

        x_text,x_visual,y_sarcasm,name,y_humor= batch
        y_sarcasm = batch[y_sarcasm]
        x_text = batch[x_text]
        x_visual = batch[x_visual]
        y_humor = batch[y_humor]
        teacher_sarcasm_out, teacher_humor_out = self.teacher_model(x_text, x_visual)
        perturbation = teacher_sarcasm_out - teacher_humor_out
        teacher_loss = F.cross_entropy(teacher_sarcasm_out, y_sarcasm) + F.cross_entropy(teacher_humor_out, y_humor)
        # teacher_loss+= F.cross_entropy(teacher_sarcasm_out, y_humor)
        self.log('train_teacher_loss', teacher_loss)
####################################################USING TALD regularization (START)      ##########################################################
    # Generate pseudo-labels with the teacher model
        with torch.no_grad():
            teacher_sarcasm_out, _ = self.teacher_model(x_text, x_visual)
            pseudo_labels = torch.argmax(teacher_sarcasm_out, dim=1)
        pseudo_labels = pseudo_labels.detach()

        # Train the student model with distillation loss and TALD regularization
        student_out = self.student_model(x_text, x_visual,perturbation)
        student_loss = (1 - self.distillation_weight) * F.cross_entropy(student_out, y_sarcasm)
        student_loss += self.distillation_weight * F.kl_div(F.log_softmax(student_out /3.5, dim=1),\
                                                             F.softmax(teacher_sarcasm_out / 3.5, dim=1), reduction='batchmean') #working good with 2.5
        # student_loss += self.distillation_weight * F.kl_div(F.log_softmax(student_out / 0.5, dim=1),\
        #                                                      F.softmax(teacher_sarcasm_out / 0.5, dim=1), reduction='batchmean')
        self.log('train_student_loss', student_loss)
        self.manual_backward(student_loss)
        self.configure_optimizers().step()
        return {'loss': student_loss + teacher_loss}

    def configure_optimizers(self):
       optimizer = torch.optim.SGD(self.student_model.parameters(), lr=0.01)
       return optimizer
    def validation_step(self, batch, batch_idx):
        x_text,x_visual,y_sarcasm,name,y_humor= batch
        y_sarcasm = batch[y_sarcasm]
        x_text = batch[x_text]
        x_visual = batch[x_visual]
        y_humor = batch[y_humor]
        # y_humor,y_sarcasm, x_text, x_visual, x_cap, name = batch
        student_out = self.student_model(x_text, x_visual)
        student_loss = F.cross_entropy(student_out, y_sarcasm)
        self.log('val_student_loss', student_loss)
        # return student_loss
        # self.log('val_acc', f1_score(lab,tmp,average='macro'))
        # self.log('val_loss', loss)
        tqdm_dict = {'val_student_loss': student_loss}

        return {
                'progress_bar': tqdm_dict,
      }
    def validation_epoch_end(self, validation_step_outputs):
      outs = []
      outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14,outs16,outs17 = \
      [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
      outs15 = []
      outs18 = []
      for out in validation_step_outputs:
        outs.append(out['progress_bar']['val_student_loss'])
      #   outs14.append(out['val_acc intensity'])
      self.log('val_student_loss', sum(outs)/len(outs))
      # self.log('val_acc_all inten', sum(outs14)/len(outs14))
      print(f'***val_student_loss {sum(outs)/len(outs)}****')
      # print(f'***val acc inten at epoch end {sum(outs14)/len(outs14)}****')

    def test_step(self, batch, batch_idx):
        # x_text, x_visual, y_sarcasm, y_humor = batch
        x_text,x_visual,y_sarcasm,name,y_humor= batch
        y_sarcasm = batch[y_sarcasm]
        x_text = batch[x_text]
        x_visual = batch[x_visual]
        y_humor = batch[y_humor]
        student_out = self.student_model(x_text, x_visual)
        student_loss = F.cross_entropy(student_out, y_sarcasm)
        self.log('test_student_loss', student_loss)
        y_sarcasm = y_sarcasm.detach().cpu().numpy()
        tmp = np.argmax(student_out.detach().cpu().numpy(),axis=-1)
        # self.log('test_accuracy', accuracy_score(student_out, y_sarcasm))
        self.log('test_accuracy', accuracy_score(tmp, y_sarcasm))

        # return student_loss
        # y_sarcasm = y_sarcasm.detach().cpu().numpy()
        print(f'confusion matrix sarcasm {confusion_matrix(tmp, y_sarcasm)}')
        return {'test_loss': student_loss,
                # 'test_loss_target': F.binary_cross_entropy_with_logits(logit_target.float(), gt_target.float()),
                # 'test_loss_emotion_multilabel': F.binary_cross_entropy_with_logits(logit_emotion.float(), gt_emotion.float()),
                'test_sarcasm f1_score':f1_score(tmp, y_sarcasm,average='macro'),
                'test_sarcasm acc_sco': accuracy_score(tmp, y_sarcasm)
                  }
    def test_epoch_end(self, outputs):
          # OPTIONAL
          outs = []
          outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
          [],[],[],[],[],[],[],[],[],[],[],[],[],[]
          outs15 = []
          outs16 = []
          outs17 = []
          outs18 = []
          for out in outputs:
            # outs15.append(out['test_loss_target'])
            outs.append(out['test_sarcasm f1_score'])
            # outs13.append(out['test_acc e13'])
            outs14.append(out['test_sarcasm acc_sco'])
          self.log('test_sarcasm f1_score', sum(outs)/len(outs))
          self.log('test_sarcasm acc_sco', sum(outs14)/len(outs14))

    # def configure_optimizers(self):
    #     self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
    #     return self.optim



class HmDataModule(pl.LightningDataModule):

  def setup(self, stage):



    self.hm_train = t_p
    self.hm_val = v_p
    # self.hm_test = te_p
    self.hm_test = te_p



  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=64, drop_last=True)

  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=64, drop_last=True)

  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128)

data_module = HmDataModule()


checkpoint_callback = ModelCheckpoint(
     monitor='val_student_loss',  #val_acc_all inten
    #  monitor='val_acc_all_offn',  #
     dirpath='ckpts_political/',
     filename='simple_teac_stu_new{epoch:02d}-val_acc{val_student_loss:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="min",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)
seed_everything(seed=1234, workers=True)
# Create the teacher model
teacher_model = TeacherModel()

# Create the student model
student_model = StudentModel()
model = SarcasmDetectionModel(teacher_model, student_model, distillation_weight=0.5)
# model.automatic_optimization = False
# hm_model_pol = SarcasmDetectionModel()
gpus = 1 if torch.cuda.is_available() else 0
trainer = pl.Trainer(gpus=gpus,max_epochs=60,callbacks=all_callbacks)
#trainer = pl.Trainer(gpus=gpus,deterministic=True,max_epochs=60,callbacks=all_callbacks)
trainer.fit(model, data_module)

# %%
test_dataloader = DataLoader(dataset=te_p, batch_size=1478)
ckpt_path = 'XYZ/simple_teac_stu_new_val_f1_all.ckpt' # put ckpt_path according to the path output in the previous cell
trainer.test(dataloaders=test_dataloader,ckpt_path=ckpt_path)