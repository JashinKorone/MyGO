"""FakeSV+ data pipeline.

The dataset follows Sec. 3.1 and Sec. 4.1 of the paper:

* every sample carries frame-level features of the three modalities
  (VGG19 for video, VGGish for audio, BERT for text), missing modalities are
  zero-padded;
* every sample carries the 4-digit **missing prompt** of Eq. 2.  The first digit
  is ``1`` when *any* modality is missing, the remaining three digits mark
  whether video / text / audio is missing (``1`` == missing).  A sample whose
  audio modality is missing therefore gets ``1001``;
* ``masker.json`` produced by ``dataset_mask.py`` stores the per-video missing
  state, and FakeSV+ guarantees at least one available modality per sample;
* two split schemes are supported -- event-split (``event``, 5-fold) and
  temporal-split (``time``, 70/15/15).
"""

import os
import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset

MODALITY_ORDER = ('vision', 'text', 'audio')


def build_missing_prompt(vision_missing, text_missing, audio_missing):
    """Build the missing prompt ``P_i`` of Eq. 2.

    Args:
        vision_missing / text_missing / audio_missing: ``1`` when the
            corresponding modality is missing.

    Returns:
        list of four ints, ``[any_missing, video, text, audio]``.
    """
    digits = [int(bool(vision_missing)), int(bool(text_missing)), int(bool(audio_missing))]
    return [int(any(digits))] + digits


class Dataset_load(Dataset):
    """A single split (train / val / test) of FakeSV+."""

    def __init__(self, path, config, data_complete):
        self.vid = []
        with open(path, 'r') as fr:
            for line in fr.readlines():
                vid = line.strip()
                if vid:
                    self.vid.append(vid)

        data = data_complete[data_complete.video_id.isin(self.vid)].copy()
        data['video_id'] = pd.Categorical(data['video_id'], categories=self.vid, ordered=True)
        data = data.sort_values('video_id', ascending=True).reset_index(drop=True)
        self.data = data

        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path'] + self.dataset_name)
        self.emb_path = os.path.join(self.dataset_path + config['emb_path'])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = str(item['video_id'])

        # Missing prompt, Eq. 2 -- 1 marks a missing modality.
        prompt = build_missing_prompt(item['vision'], item['text'], item['audio'])
        prompt = torch.tensor(prompt, dtype=torch.int)

        # Event id, used by the event-level contrastive loss of Eq. 6.
        event = torch.tensor(item['keywords_code'], dtype=torch.long)

        label = 0 if item['annotation'] == '真' else 1
        label = torch.tensor(label, dtype=torch.long)

        with open(os.path.join(self.emb_path, vid + '.pkl'), 'rb') as f:
            emb = pickle.load(f)

        sample = {
            'label': label,
            'event': event,
            'audioframes': emb['audio'],
            'frames': emb['vision'],
            'text': emb['text'],
            'prompt': prompt,
        }
        # Embedded captions (Fig. 1a).  Pre-extracted OCR captions are optional;
        # when absent CKA falls back to the textual features.
        if 'caption' in emb:
            sample['caption'] = emb['caption']
        return sample

    def __str__(self):
        info = [self.dataset_name]
        info.extend(['The number of posts: {}'.format(self.data.shape[0])])
        info.extend(['Modality-complete ratio: {:.2f}%'.format(self.complete_ratio() * 100)])
        return '\n'.join(info)

    def complete_ratio(self):
        if self.data.shape[0] == 0:
            return 0.0
        missing = self.data[list(MODALITY_ORDER)].sum(axis=1)
        return float((missing == 0).mean())


class FVDDataset:
    """Loader of the FakeSV+ annotations, masker and split files."""

    def __init__(self, config):
        self.num_events = None
        self.config = config

        self.dataset_name = config['dataset']
        self.dataset_mode = config['dataset_mode']
        self.dataset_general_path = os.path.abspath(config['data_path'] + self.dataset_name)
        self.dataset_path = os.path.join(self.dataset_general_path + '/data')

        news_data = self.load_data()
        self.debunk_dataset = None

        if self.dataset_mode == 'event':
            # Event-split: 5-fold cross validation, 80% train / 20% test.
            self.dataset_path_split = os.path.join(
                self.dataset_path, self.dataset_mode, str(config['fold'])
            )
            self.train_dataset = Dataset_load(
                os.path.join(self.dataset_path_split, 'train.txt'), config, news_data
            )
            self.val_dataset = None
            self.test_dataset = Dataset_load(
                os.path.join(self.dataset_path_split, 'test.txt'), config, news_data
            )
        else:
            # Temporal-split: earliest 70% train, next 15% val, latest 15% test.
            self.dataset_path_split = os.path.join(self.dataset_path, self.dataset_mode)
            self.train_dataset = Dataset_load(
                os.path.join(self.dataset_path_split, 'train.txt'), config, news_data
            )
            self.val_dataset = Dataset_load(
                os.path.join(self.dataset_path_split, 'val.txt'), config, news_data
            )
            self.test_dataset = Dataset_load(
                os.path.join(self.dataset_path_split, 'test.txt'), config, news_data
            )

    def load_data(self):
        data_complete_path = os.path.join(self.dataset_path, 'data.json')
        data_complete = pd.read_json(data_complete_path, orient='records', dtype=False, lines=True)

        masker_path = os.path.join(self.dataset_general_path, 'masker.json')
        masker = pd.read_json(masker_path).T.reset_index(drop=False)
        masker['video_id'] = masker['index'].apply(lambda x: str(x).split('.')[0])
        masker = masker.drop(columns=['index'])
        data_complete = pd.merge(data_complete, masker, on='video_id', how='inner')

        self.num_events = data_complete['keywords'].nunique()
        data_complete['keywords'] = pd.Categorical(data_complete['keywords'])
        data_complete['keywords_code'] = data_complete['keywords'].cat.codes
        # '辟谣' (debunking) videos are excluded from the detection task.
        return data_complete[data_complete['annotation'] != '辟谣']
