import json
import os
from datasets import Dataset, DatasetDict, IterableDataset, Dataset
from torch.utils.data import DataLoader
import random
import numpy as np
import glob

def VLA_dataset_generator(shards, eos_token, static_video_description, return_info, action_before_vision):
    '''
    each shard is a jsonl file, with each line containing a json object
    the json object contains the following fields:
    - trajectory_id: a integer that identifies the trajectory
    - view: a string that describes the view
    - start_frame: the start frame of the clip, -1 means it is 6 duplicate first frames
    - task_description: a string that describes the task, identical for clips with the same trajectory_id
    - scene_description: a string that describes the initial scene, identical for clips with the same trajectory_id and view
    - input_clip_description: a string that describes the frame difference in the input clip
    - output_clip_description: a string that describes the frame difference in the output clip
    - input_video_tokens: a 2D array of size 768 (256 * 3),
        256 * 3 is because each clip has 6 frames and downsamples by factor 2
    - output_video_tokens: a 2D array of size 768 (256 * 3),
    - input_action_tokens: a 2D array of size 42 (6 * 7),
    - output_action_tokens: a 2D array of size 42 (6 * 7),
    
    output:
    a generator that yields a dictionary with only the 'text' field

    text = '<bott_i>' + data['task_description'] + '<eott_i>' + \
            '<bots_i>' + data['scene_description'] + '<eots_i>' + \
            '<botp_i>' + data['input_clip_description'] + '<eotp_i>' + \ 
            '<bov_i>' + ''.join([f'<va{str(x)}>' for x in data['input_video_tokens']]) + '<eov_i>' + \
            '<boa_i>' + ''.join([f'<va{str(x)}>' for x in data['input_action_tokens']]) + '<eoa_i>' + \
            '<botp_o>' + data['output_clip_description'] + '<eotp_o>' + \
            '<bov_o>' + ''.join([f'<va{str(x)}>' for x in data['output_video_tokens']]) + '<eov_o>' + \
            '<boa_o>' + ''.join([f'<va{str(x)}>' for x in data['output_action_tokens']) + '<eoa_o>' + eos_token
    length: 14 special tokens + 
            768 * 2 video tokens +
            42 * 2 action tokens +
            200 task description, scene description, input clip, output clip
            2 eos_token and bos_token (will be automatically added by the tokenizer)
            thus, 2048 sequence length is enough
    '''

    for shard in shards:
        with open(shard, "r") as f:
            for line in f:
                try:
                    instance_data = json.loads(line)
                    if instance_data['input_clip_description'] == '': # sample a description for the input clip
                        instance_data['input_clip_description'] = random.choice(static_video_description)
                    text_input = '<bott_i>' + instance_data['task_description'] + '<eott_i>' + \
                            '<bots_i>' + instance_data['scene_description'] + '<eots_i>' + \
                            '<botp_i>' + instance_data['input_clip_description'] + '<eotp_i>'
                    if action_before_vision:
                        text_input += '<boa_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_action_tokens']]) + '<eoa_i>' + \
                                '<bov_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_video_tokens']]) + '<eov_i>'
                    else:
                        text_input += '<bov_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_video_tokens']]) + '<eov_i>' + \
                                '<boa_i>' + ''.join([f'<va{str(x)}>' for x in instance_data['input_action_tokens']]) + '<eoa_i>'
                    text_output = '<botp_o>' + instance_data['output_clip_description'] + '<eotp_o>'
                    if action_before_vision:
                        text_output += '<boa_o>' + ''.join([f'<va{str(x)}>' for x in instance_data['output_action_tokens']]) + '<eoa_o>' + \
                                        '<bov_o>' + ''.join([f'<va{str(x)}>' for x in instance_data['output_video_tokens']]) + '<eov_o>'
                    else:
                        text_output += '<bov_o>' + ''.join([f'<va{str(x)}>' for x in instance_data['output_video_tokens']]) + '<eov_o>' + \
                                        '<boa_o>' + ''.join([f'<va{str(x)}>' for x in instance_data['output_action_tokens']]) + '<eoa_o>' + eos_token
                except:
                    continue
                if return_info:
                    yield {"input": text_input, "output": text_output, 
                           "trajectory_id": instance_data['trajectory_id'], "view": instance_data['view'],
                           "gt_actions": instance_data['gt_actions']}
                else:
                    yield {"input": text_input, "output": text_output}

def get_VLA_dataset(args, eos_token, split='train', return_info=False):
    root = args.data_root
    shards = glob.glob(os.path.join(root, split, '*_stacked.jsonl'))
    shards = sorted(shards)
    if args.data_debug:
        shards = shards[:1]
    if args.dataset_type == 'dataset':
        ds = Dataset.from_generator(VLA_dataset_generator, gen_kwargs={"shards": shards, 
                                                            "eos_token": eos_token,
                                                            "static_video_description": args.static_video_description,
                                                            "return_info": return_info,
                                                            "action_before_vision": args.action_before_vision
                                                            })
    else: # iterable dataset
        ds = IterableDataset.from_generator(VLA_dataset_generator, gen_kwargs={"shards": shards, 
                                                                "eos_token": eos_token,
                                                                "static_video_description": args.static_video_description,
                                                                "return_info": return_info,
                                                                "action_before_vision": args.action_before_vision
                                                                })
        # ds.column_names = ['text']
    return ds