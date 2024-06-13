import json
import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

import os
import argparse
from tokenizer import VQGANVisionActionEval
from configs import H4ArgumentParser, DataArguments, VLAModelArguments, TATSModelArguments
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from time import time

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

    action = (action - torch.tensor(instance_data['mean']).to(device)) / torch.tensor(instance_data['std']).to(device)

    _, _, vq_output, vq_output_action = model(video.unsqueeze(0), action.unsqueeze(0))
    video_tokens, action_tokens = vq_output['encodings'].reshape(-1), vq_output_action['encodings'].reshape(-1) # video tokens: 3*256=768, action tokens: 6*7=42

    return video_tokens, action_tokens

@torch.no_grad()
def call_vla(instance_data: dict,
             video_tokens: torch.Tensor, action_tokens: torch.Tensor, 
             model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
             data_args: DataArguments, device):

    video_tokens = video_tokens.cpu().numpy().tolist()
    action_tokens = action_tokens.cpu().numpy().tolist()

    input_text = '<bott_i>' + instance_data['task_description'] + '<eott_i>' + \
                 '<bots_i>' + instance_data['scene_description'] + '<eots_i>' + \
                 '<botp_i>' + instance_data['clip_description'] + '<eotp_i>'

    if data_args.action_before_vision:
        input_text += '<boa_i>' + ''.join([f'<va{str(x)}>' for x in action_tokens]) + '<eoa_i>' + \
                       '<bov_i>' + ''.join([f'<va{str(x)}>' for x in video_tokens]) + '<eov_i>'
    else:
        input_text += '<bov_i>' + ''.join([f'<va{str(x)}>' for x in video_tokens]) + '<eov_i>' + \
                       '<boa_i>' + ''.join([f'<va{str(x)}>' for x in action_tokens]) + '<eoa_i>'
    
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    output = model.generate(input_ids, max_new_tokens=1024, num_beams=1)
    output_text = tokenizer.decode(output[0], skip_special_tokens=False)
    print(output_text)

    output_video_tokens_pred = [int(x[:-1]) for x in output_text.split('<eov_o>')[0].split('<bov_o>')[-1].split('<va') if x != '']
    output_video_tokens_pred = torch.tensor(output_video_tokens_pred, device=device).reshape(1, 3, 16, 16)

    output_action_tokens_pred = [int(x[:-1]) for x in output_text.split('<eoa_o>')[0].split('<boa_o>')[-1].split('<va') if x != '']
    output_action_tokens_pred = torch.tensor(output_action_tokens_pred, device=device).reshape(1, 6, 7)

    output_clip_description_pred = output_text.split('<eotp_o>')[0].split('<botp_o>')[-1]

    return output_video_tokens_pred, output_action_tokens_pred, output_clip_description_pred

def call_models(instance_data, model_vq: VQGANVisionActionEval,
                model_vla: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                tats_args: TATSModelArguments, data_args: DataArguments, device) -> dict:
    '''
    call the models to predict the actions and clip description
    override the original actions and clip description with the predicted ones
    '''
    
    video_tokens, action_tokens = encode(instance_data, model_vq, tats_args, device=device)

    start = time()
    output_video_tokens_pred, output_action_tokens_pred, output_clip_description_pred = call_vla(instance_data, video_tokens, action_tokens, model_vla, tokenizer, data_args, device)
    print(f'vla time: {time()-start:.2f} sec')

    output_frames = model_vq.decode_video(output_video_tokens_pred).squeeze(0).permute(1,0,2,3).detach().cpu() # 6, 3, 256, 256
    output_action_pred = model_vq.decode_action(output_action_tokens_pred).squeeze(0).detach().cpu() # 6, 7

    instance_data['clip_description'] = output_clip_description_pred
    instance_data['actions'] = output_action_pred.tolist()
    instance_data['images'] = output_frames

    return instance_data

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

    model_kwargs = dict(
        use_flash_attention_2=vla_args.use_flash_attention_2,
        torch_dtype=getattr(torch, vla_args.torch_dtype)
    )
    model_vla = AutoModelForCausalLM.from_pretrained(vla_args.model_name_or_path, **model_kwargs).to(device)
    tokenizer = AutoTokenizer.from_pretrained(vla_args.model_name_or_path)

    f = open(data_args.src_filepath, 'r')
    lines = f.readlines()
    f.close()

    for line in lines:
        '''
        1. encode the images and actions
        src file should contains the following entry
        trajectory_id, frame_number, task_description, scene_description, input_clip_description
        image_indices, actions
        '''
        instance_data = json.loads(line) 

        trajectory_id, view = instance_data['trajectory_id'], instance_data['view']
        save_dir = os.path.join(data_args.save_dir, f'{trajectory_id}_{view}')
        save_path = os.path.join(save_dir, 'results.json')
        save_image_dir = os.path.join(save_dir, 'images_pred')
        save_image_dir_gt = os.path.join(save_dir, 'images_gt')
        os.makedirs(save_image_dir, exist_ok=True)
        os.makedirs(save_image_dir_gt, exist_ok=True)

        output_data = {}
        pred_descriptions = {}
        pred_actions = torch.empty(0, 7)

        image_format = '/home/v-rundongluo/robotdata/' + \
                    ('bridge2/images_bridge' if data_args.dataset_name == 'bridge2' else 'RT1/RT1-images') + \
                    '/outputimage_' + str(instance_data['trajectory_id']) + \
                    (('_{}_' + str(instance_data['view'])) if data_args.dataset_name == 'bridge2' else '_{}') + \
                    '.png'
        cur_instance_data = {}
        cur_instance_data['task_description'] = instance_data['task_description']
        cur_instance_data['scene_description'] = instance_data['scene_description']
        cur_instance_data['mean'] = instance_data['mean']
        cur_instance_data['std'] = instance_data['std']

        for start_frame in [-1] + list(range(0, instance_data['frame_number'], 6))[:-1]:
            if start_frame != -1:
                cur_instance_data['image_paths'] = [image_format.format(x) for x in instance_data['image_indices'][start_frame:start_frame+6]]
                cur_instance_data['actions'] = instance_data['actions'][start_frame-1:start_frame+5] if start_frame > 0 else \
                                                [[0. for _ in range(6)] + [instance_data['actions'][0][-1]]] + instance_data['actions'][start_frame:start_frame+5]
                cur_instance_data['clip_description'] = instance_data['descriptions'][str(start_frame+5)]
            else:
                cur_instance_data['image_paths'] = [image_format.format(instance_data['image_indices'][0])] * 6
                cur_instance_data['actions'] = [[0. for _ in range(6)] + [instance_data['actions'][0][-1]]] * 6
                cur_instance_data['clip_description'] = ''
            
            # call the models, override original actions and clip description with the predicted ones
            cur_instance_data = call_models(cur_instance_data, model_vq, model_vla, tokenizer, tats_args, data_args, device)
            pred_descriptions[start_frame+11 if start_frame!=-1 else 5] = cur_instance_data['clip_description']
            pred_actions = torch.cat((pred_actions, (torch.tensor(cur_instance_data['actions']) * torch.tensor(instance_data['std']) + torch.tensor(instance_data['mean']))), dim=0)

            # save the frames
            for i, img in enumerate(cur_instance_data['images']):
                img = (img + 0.5).clamp(0,1).numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                img.save(os.path.join(save_image_dir, f'{start_frame+6+i if start_frame!=-1 else i}.png'))

        output_data['trajectory_id'] = instance_data['trajectory_id']
        output_data['view'] = instance_data['view']
        output_data['task_description'] = instance_data['task_description']
        output_data['scene_description'] = instance_data['scene_description']
        # stack the pred_description and gt_description
        stacked_descriptions = {}
        for key, value in pred_descriptions.items():
            stacked_descriptions[int(key)] = {'gt': instance_data['descriptions'][str(key)], 'pred': value}
        output_data['descriptions'] = stacked_descriptions
        # output_data['pred_descriptions'] = pred_descriptions
        output_data['pred_actions'] = pred_actions.tolist()

        with open(save_path, 'w') as f:
            json.dump(output_data, f)

if __name__ == '__main__':
    main()



