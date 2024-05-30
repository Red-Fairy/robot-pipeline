import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import argparse
import json
from torch.utils.data import DataLoader
from datasets import IterableDataset

def image_action_generator(args, split='train', action=False, transform=None):
    json_file = os.path.join(args.root, f'{split}.jsonl')
    with open(json_file, 'r') as f:
        for line in f:
            instance_data = json.loads(line)
            num_frames = instance_data['frame_number']
            if num_frames < args.sequence_length:
                continue
            for start in range(num_frames - args.sequence_length + 1):
                video = []
                actions = []
                for i in range(start, start + args.sequence_length):
                    img = Image.open(instance_data['images'][i])
                    if transform is None:
                        transform = transforms.Compose([
                            transforms.Resize((args.resolution, args.resolution)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.)) # To [-0.5, 0.5]
                        ])
                    img = transform(img)
                    video.append(img)
                    if i != start + args.sequence_length - 1:
                        actions.append(instance_data['actions'][i])

                video = torch.stack(video).permute(1, 0, 2, 3) # (C, T, H, W)
                actions = torch.tensor(actions) # (T-1, 7), 7 is the number of action dimension

                yield {'video': video, 'actions': actions}

def get_image_action_dataloader(args, split='train'):
    iteratable_dataset = IterableDataset.from_generator(image_action_generator, args, split)
    dataloader = DataLoader(iteratable_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True if split == 'train' else False)
    return dataloader