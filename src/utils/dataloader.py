from common.function import pad_frame_sequence
import numpy as np
import torch

def _init_fn(worker_id):
    np.random.seed(2024)

def collate_fn(batch):
    # label
    label = [item['label'] for item in batch]
    # event
    event = [item['event'] for item in batch]
    # audio
    audioframes = [item['audioframes'] for item in batch]
    # frame
    frames = [item['frames'] for item in batch]
    # c3d
    # text
    text = [item['text'] for item in batch]
    # masker
    masker = [item['masker'] for item in batch]


    return {
        'label': torch.stack(label),
        'event': torch.stack(event),
        'audioframes': torch.stack(audioframes),
        #'audioframes_masks': audioframes_masks,
        'frames': torch.stack(frames),
        #'frames_masks': frames_masks,
        'text': torch.stack(text),
        #'c3d_masks': c3d_masks,
        #'title': torch.stack(title),
        #'intro': torch.stack(intro),
        #'comment': torch.stack(comment),
        #'profile': torch.stack(profile),
        'masker': torch.stack(masker)
    }



