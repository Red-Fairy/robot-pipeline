import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import argparse
import json
import copy

class ImageActionDataset(Dataset):
    '''
    A dataset that batchify images into videos
    In the root directory contains files in the format: prefix_{scene_id}_{frame_id}_{view_id}.png
    we batchify the images with the same scene_id and view_id into a video clip with specified length
    each time when calling __getitem__, we randomly sample a video clip from the dataset
    '''
    def __init__(self, args, action=False, split='train', transform=None):
        self.split_root = args.split_root
        self.data_root = args.data_root
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((args.resolution, args.resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
            ])
        else:
            self.transform = transform
        self.length = args.sequence_length
        self.split = split
        self.action = action
        self.mask_action = args.action_mask
        self.mask_action_ratio = args.action_mask_ratio
        self.filenames = []

        with open(os.path.join(self.split_root, f'{split}.jsonl'), 'r') as f:
            for line in f:
                instance_data = json.loads(line)
                num_frames = instance_data['frame_number']
                if num_frames < self.length:
                    continue
                instance_format = args.data_root + '/outputimage_' + str(instance_data['trajectory_id']) + '_{}_' + str(instance_data['view']) + '.png'
                new_instance = {'image_paths': instance_format, 'frame_number': num_frames, 'image_indices': instance_data['image_indices']}
                if self.action:
                    new_instance['actions'] = instance_data['actions']
                self.filenames.append(new_instance)

        # with open(os.path.join(self.root, f'{split}.txt'), 'r') as f:
        #     img_filenames = f.readlines()
        # img_filenames = [img_filename.strip() for img_filename in img_filenames]

        # for img_filename in img_filenames:
        #     _, scene_id, frame_id, view_id = img_filename.split('/')[-1].split('.')[0].split('_')
        #     key = (scene_id, view_id)
        #     if key not in self.filenames:
        #         self.filenames[key] = []
        #     self.filenames[key].append(img_filename)

        # self.keys = list(self.filenames.keys())

        # remove keys that have less than self.length frames
        # self.keys = [key for key in self.keys if len(self.filenames[key]) >= self.length]

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):

        data = self.filenames[index]
        start = torch.randint(-1, data['frame_number'] - self.length + 1, (1,)).item()
        video = []
        actions = []

        if start == -1: # video will be self.length duplicates of frame 0, and each action entry will be [0] * 7
            img_filename = data['image_paths'].format(data['image_indices'][0])
            img = Image.open(img_filename)
            img = self.transform(img)
            video = [img] * self.length
            if self.action:
                actions = [[0. for _ in range(7)] for _ in range(self.length)]
        else:
            for i in range(start, start + self.length):
                img_filename = data['image_paths'].format(data['image_indices'][i])
                img = Image.open(img_filename)
                img = self.transform(img)
                video.append(img)
                if self.action:
                    actions.append(data['actions'][i-1] if i > 0 else [0. for _ in range(7)])
        
        if self.action and self.mask_action:
            mask_indices = torch.randperm(self.length)[:int(self.length * self.mask_action_ratio)]
            actions_masked = copy.deepcopy(actions)
            for i in mask_indices:
                actions_masked[i] = [0. for _ in range(7)]

        ret = {}
        ret['video'] = torch.stack(video).permute(1, 0, 2, 3) # (C, T, H, W)
        if self.action:
            ret['actions'] = torch.tensor(actions) # (T, 7), 7 is the number of action dimension
            if self.mask_action:
                ret['actions_masked'] = torch.tensor(actions_masked)
        return ret

        # key = self.keys[index]
        # filenames = self.filenames[key]
        # if len(filenames) < self.length:
        #     raise ValueError('Not enough frames for the video clip')
        # start = torch.randint(0, len(filenames) - self.length + 1, (1,)).item()
        # video = []
        # for i in range(start, start + self.length):
        #     img_filename = filenames[i]
        #     img = Image.open(os.path.join(self.root, img_filename))
        #     if self.transform:
        #         img = self.transform(img)
        #     video.append(img)
        # video = torch.stack(video).permute(1, 0, 2, 3) # (C, T, H, W)
        # return {'video': video, 'key': key}

def get_image_action_dataloader(args, split='train', action=False):
    dataset = ImageActionDataset(args, split=split, action=action)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             num_workers=args.num_workers,
                                             shuffle=True if split == 'train' else False)
    return dataloader

# def get_image_dataloader(args, split='train'):
#     transform = transforms.Compose([
#         transforms.Resize((args.resolution, args.resolution)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
#     ])
#     dataset = ImageDataset(args, split=split, transform=transform)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
#                                              num_workers=args.num_workers,
#                                              shuffle=True if split == 'train' else False)
#     return dataloader

# class ImageActionDataset(Dataset):
#     '''
#     A dataset that batchify images into videos
#     In the root directory contains files in the format: prefix_{scene_id}_{frame_id}_{view_id}.png
#     we batchify the images with the same scene_id and view_id into a video clip with specified length
#     each time when calling __getitem__, we randomly sample a video clip from the dataset
#     '''
#     def __init__(self, args, split='train', transform=None, num_shards=None):
#         self.root = args.dataroot
#         self.transform = transform
#         self.length = args.sequence_length
#         self.split = split
#         self.filenames = {}

#         num_shards = len(os.listdir(os.path.join(self.root, split))) if num_shards is None else 1 # for debug propose

#         self.data_full = []
#         for shard_id in range(num_shards):
#             with open(os.path.join(self.root, f'{split}/shard_{shard_id:03d}.jsonl'), 'r') as f:
#                 shard_data = f.readlines()
#                 for line in shard_data:
#                     instance_data = json.loads(line)
#                     if int(instance_data['Frame_number'] >= self.length):
#                         img0 = instance_data['Visual'][0]
#                         prefix = os.path.join(*img0.split('/')[:-1])
#                         filename_prefix, scene_id, frame_id, view_id = img0.split('/')[-1].split('.')[0].split('_')
#                         instance_format = prefix + '/' + filename_prefix + '_' + str(scene_id) + '_{}_' + view_id
#                         self.data_full.append({'img_filename_format': instance_format, 'Action': instance_data['Action'], 'Frame_number': instance_data['Frame_number']})

#     def __len__(self):
#         return len(self.data_full)
    
#     def __getitem__(self, index):
#         data = self.data_full[index]
#         img_filename_format = data['img_filename_format']
#         start = torch.randint(0, data['frame_number'] - self.length + 1, (1,)).item()
#         video = []
#         actions = []
#         for i in range(start, start + self.length):
#             img_filename = img_filename_format.format(i)
#             img = Image.open(os.path.join(self.root, img_filename))
#             if self.transform:
#                 img = self.transform(img)
#             video.append(img)
#             if i != start + self.length - 1:
#                 actions.append(data['Action'])
#         video = torch.stack(video).permute(1, 0, 2, 3) # (C, T, H, W)
#         actions = torch.tensor(actions) # (T-1, 7), 7 is the number of action dimension
#         return {'video': video, 'actions': actions}

        

