import os
import clip
import torch
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from PIL import Image
import requests
from flair.data import Sentence
from flair.models import SequenceTagger
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--stablediffusion",type=str)

#####################CLIP部分#####################
print("####### CLIP ########")
# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
print(f'Using device:{device}')
# 获取文件夹中所有的图片文件
folder_path = './image'  
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# 处理图片并获取CLIP视觉特征，保存在clip_feature文件夹中
image_features = []
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image_input)
    image_features.append(image_feature)
    image_file_without_extension = os.path.splitext(image_file)[0]
    torch.save(image_features, f'./clip_feature/{image_file_without_extension}.pt')
    print(f'Complete save clip feature of {image_file}')


#####################SAM部分#####################
print("####### SAM ########")
print(f'Using device:{device}')
#储存并可视化mask
def show_anns(anns,name):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)#对注释列表进行排序，按照区域面积从大到小排序。
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0#创建一个全为1的RGBA图像（四通道，包括红、绿、蓝和透明度），初始透明度设置为0。
    count=0
    for ann in sorted_anns:
        count+=1
        m = ann['segmentation']
        np.save(f'./sam_segmentation_feature/npy/{name}/{name}_{count}.npy',m)
        plt.imsave(f'./sam_segmentation_feature/visible/{name}/{name}_{count}.png',m,cmap='gray')
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    plt.imsave(f'./sam_segmentation_feature/visible/{name}/{name}.png',img)

#加载SAM模型
sam_checkpoint = "./weight_sam/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

#读取图片,生成segmentation_mask，并可视化
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    image_file_without_extension = os.path.splitext(image_file)[0]
    os.makedirs(f'./sam_segmentation_feature/npy/{image_file_without_extension}',exist_ok=True)
    os.makedirs(f'./sam_segmentation_feature/visible/{image_file_without_extension}',exist_ok=True)
    show_anns(masks,image_file_without_extension)
    print(f'Complete save segmentation_mask of {image_file}')


#####################Image Caption部分#####################
print("####### Image Caption ########")
#加载OFA模型
model = Model.from_pretrained('damo/multi-modal_gemm-vit-large-patch14_generative-multi-modal-embedding')
p = pipeline(task=Tasks.generative_multi_modal_embedding, model=model)

#读取图片，生成image_caption，并储存为txt文档
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)
    image_caption = p.forward({'image': image, 'captioning': True})['caption']
    image_file_without_extension = os.path.splitext(image_file)[0]
    print('image caption: {}'.format(image_caption))
    os.makedirs('./image_caption',exist_ok=True)
    txt_path='./image_caption'
    with open(f'{txt_path}/{image_file_without_extension}.txt', 'w') as f:
         f.write(image_caption)
    print(f'Complete save image caption of {image_file}')


#####################Part of Speech部分#####################


print("####### Part of Speech ########")

# 加载预训练的词性标注模型
tagger = SequenceTagger.load('pos')
#加载txt文档，生成prompt，并储存为txt文档
txt_files=[f for f in os.listdir(txt_path) if f.endswith('.txt')]
for txt_file in txt_files:
    txt_file_path = os.path.join(txt_path, txt_file)
    with open(txt_file_path, 'r') as f:
        sentence = Sentence(f.read())
        # 应用模型
        tagger.predict(sentence)
        # 提取名词
        nouns = [token.text for token in sentence.tokens if token.tag.startswith('N')]
        #将结果变为prompt
        prompt=', '.join(['a ' + word for word in nouns]) + ', top down'
        print(f'Prompts of {txt_file} are {prompt}')
        os.makedirs('./prompts',exist_ok=True)
        prompt_path='./prompts'
        txt_file_without_extension = os.path.splitext(txt_file)[0]
        with open(f'{prompt_path}/{txt_file_without_extension}.txt', 'w') as f:
         f.write(prompt)
        print(f'Complete save prompt of {txt_file}')

#####################stablediffusion部分#####################
print("####### stable diffusion ########")
#加载prompts文档
prompt_files=[f for f in os.listdir(prompt_path) if f.endswith('.txt')]
for prompt_file in prompt_files:
    txt_file_path = os.path.join(txt_path, txt_file)
    with open(txt_file_path, 'r') as f:
        prompt_sd = Sentence(f.read())
        parser.parse_args().stablediffusion=f'./scripts/txt2img.py --prompt {prompt_sd} --ckpt <path/to/768model.ckpt/> --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --device cuda'
        os.system(parser.parse_args().stablediffusion)