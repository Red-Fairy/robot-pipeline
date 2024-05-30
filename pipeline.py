import json
import argparse
import yaml
import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tokenizer import VQGANVisionActionEval, VideoData, get_image_action_dataloader, count_parameters
# from vla import get_VLA_dataset
from configs import H4ArgumentParser, DataArguments, VLAModelArguments, TATSModelArguments
from pytorch_lightning.strategies import DeepSpeedStrategy
from torchvision import transforms

import logging
import random
import sys
import transformers
import os
import json
import time
import mii

@torch.no_grad()
def encode(instance_data, model, tats_args, device):

    transform = transforms.Compose([
        transforms.Resize((tats_args.resolution, tats_args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
    ])
    video = []
    for img_path in instance_data['image_paths']:
        img = Image.open(img_path)
        img = transform(img)
        video.append(img)
    video = torch.stack(video).permute(1,0,2,3).to(device) # [C, T, H, W]
    action = torch.tensor(instance_data['actions']).to(device) # [T, 7]

    _, _, vq_output, vq_output_action = model(video.unsqueeze(0), action.unsqueeze(0))
    video_tokens, action_tokens = vq_output['encodings'].reshape(-1), vq_output_action['encodings'].reshape(-1) # video tokens: 3*256=768, action tokens: 6*7=42

    return video_tokens, action_tokens

@torch.no_grad()
def call_vla(video_tokens: torch.Tensor, action_tokens: torch.Tensor, 
             vla_pipe: mii.pipeline, data_args: DataArguments, device):

    video_tokens = video_tokens.cpu().numpy().tolist()
    action_tokens = action_tokens.cpu().numpy().tolist()

    input_text = '<bov_i>' + ''.join([f'<va{str(x)}>' for x in video_tokens]) + '<eov_i>' + \
                '<boa_i>' + ''.join([f'<va{str(x)}>' for x in action_tokens]) + '<eoa_i>'

    if data_args.action_before_vision:
        input_text += '<boa_i>' + ''.join([f'<va{str(x)}>' for x in action_tokens]) + '<eoa_i>' + \
                '<bov_i>' + ''.join([f'<va{str(x)}>' for x in video_tokens]) + '<eov_i>'
    else:
        input_text += '<bov_i>' + ''.join([f'<va{str(x)}>' for x in video_tokens]) + '<eov_i>' + \
                '<boa_i>' + ''.join([f'<va{str(x)}>' for x in action_tokens]) + '<eoa_i>'

    output = vla_pipe([input_text], max_new_tokens=1024)
    output_text = output[0].generated_text

    output_action_tokens_pred = [int(x[:-1]) for x in output_text.split('<eoa_o>')[0].split('<boa_o>')[-1].split('<va') if x != '']
    output_action_tokens_pred = torch.tensor(output_action_tokens_pred, device=device).unsqueeze(0).reshape(1, 6, 7)

    output_clip_description_pred = output_text.split('<eotp_o>')[0].split('<botp_o>')[-1]

    return output_action_tokens_pred, output_clip_description_pred

def call_models(instance_data, model_vq: VQGANVisionActionEval, vla_pipe: mii.pipeline, 
                tats_args: TATSModelArguments, data_args: DataArguments, device):
    
    video_tokens, action_tokens = encode(instance_data, model_vq, tats_args, device=device)

    output_action_tokens_pred, output_clip_description_pred = call_vla(video_tokens, action_tokens, vla_pipe, data_args, device)

    output_action_pred = model_vq.decode_action(output_action_tokens_pred).squeeze(0).detach().cpu() # 6, 7

    instance_data['clip_description'] = output_clip_description_pred
    instance_data['actions'] = output_action_pred.tolist()

    return instance_data

def call_robot(instance_data, robot) -> dict:
    '''
    use the predicted actions to call the robot
    should override the image_paths with the observation after movement, 
    and override the given actions with the actual ones (as the robot may not be able to follow the predicted actions exactly)
    return the new instance_data 
    '''
    pass

def main():

    parser = H4ArgumentParser((VLAModelArguments, DataArguments, TATSModelArguments))
    vla_args, data_args, tats_args = parser.parse()

    local_rank = os.getenv('LOCAL_RANK', 0)
    device = f'cuda:{local_rank}'

    assert tats_args.sequence_length == 6

    # 0. define the vq model and vla model
    model_vq = VQGANVisionActionEval(tats_args)
    state_dict = torch.load(tats_args.weight_path, map_location='cpu')['state_dict']
    result = model_vq.load_state_dict(state_dict, strict=False)
    for k in result.missing_keys:
        assert 'discriminator' in k or 'perceptual_model' in k
    model_vq = model_vq.eval().to(device)

    vla_pipe = mii.pipeline(vla_args.model_name_or_path)

    # 1. encode the images and actions
    with open(data_args.src_filepath, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 1
        line = lines[0]

        instance_data = json.loads(line)

    # call the models, override original actions and clip description with the predicted ones
    instance_data = call_models(instance_data, model_vq, vla_pipe, tats_args, data_args, device)

    # call the robot, override the image_paths and actions with the actual ones
    instance_data = call_robot(instance_data, robot)

if __name__ == '__main__':
    main()



