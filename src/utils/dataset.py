"""
Data pre-processing
##########################
"""
from torch.utils.data import Dataset
import os
import pickle
import pandas as pd
import torch
import h5py
import json
from common.function import pad_frame_sequence_dataset



class Dataset_load(Dataset):
    ### 需要修改！！！！ ###
    def __init__(self, path, config, data_complete):
        self.vid = []
        with open(path, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = data_complete[data_complete.video_id.isin(self.vid)]
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)
        self.data.reset_index(inplace=True)

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path'] + self.dataset_name)

        self.emb_path = os.path.join(self.dataset_path + config['emb_path'])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # masker
        masker_vision = item['vision']
        masker_text = item['text']
        masker_audio = item['audio']
        masker = []
        if (masker_vision == 0) and (masker_text == 0) and (masker_audio == 0):
            masker.append(1)
        else:
            masker.append(0)
        # Tag Encoder第一位: 存在数据缺失则置0，数据完整则置1

        if masker_vision == 1:
            masker.append(1)
        else:
            masker.append(0)
        # Tag Encoder第二位: 视觉数据缺失置1，完整则置0

        if masker_audio == 1:
            masker.append(1)
        else:
            masker.append(0)
        # Tag Encoder第三位: 听觉数据缺失置1，完整则置0

        if masker_text == 1:
            masker.append(1)
        else:
            masker.append(0)
        # Tag Encoder第四位: 文本数据缺失置1，完整则置0

        masker = torch.tensor(masker, dtype=torch.int)

        # event
        event = item['keywords_code']
        event = torch.tensor(event, dtype=torch.int)

        # label
        label = 0 if item['annotation'] == '真' else 1
        label = torch.tensor(label)

        # audio
        emb = pickle.load(open(os.path.join(self.emb_path, vid + '.pkl'), 'rb'))
        audioframes = emb['audio']

        # frames
        #frames = pickle.load(open(os.path.join(self.frame_path, vid + '.pkl'), 'rb'))
        frames = emb['vision']

        # video
        #c3d = h5py.File(self.c3d_path + vid + ".hdf5", "r")[vid]['c3d_features']
        #c3d = torch.FloatTensor(c3d)

        # text
        text = emb['text']
        #text = pickle.load(open(os.path.join(self.text_path, vid + '.pkl'), 'rb'))
        #title = text['title']
        #intro = text['intro']
        #comment = text['comment']
        #profile = text['profile']

        return {
            'label': label,
            'event': event,
            'audioframes': audioframes,
            'frames': frames,
            'text': text,
            'masker': masker,
            #'title': title,
            #'intro': intro,
            #'comment': comment,
            #'profile': profile
        }

    def __str__(self):
        info = [self.dataset_name]
        info.extend(['The number of posts: {}'.format(self.data.shape[0])])
        return '\n'.join(info)


class FVDDataset:
    def __init__(self, config):
        self.num_events = None
        self.config = config

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_mode = config['dataset_mode']
        self.dataset_general_path = os.path.abspath(config['data_path'] + self.dataset_name)
        self.dataset_path = os.path.join(self.dataset_general_path + '/data')

        #self.audio_path = os.path.join(self.dataset_general_path + config['audio_path'])
        #self.frame_path = os.path.join(self.dataset_general_path + config['frame_path'])
        #self.c3d_path = os.path.join(self.dataset_general_path + config['c3d_path'])
        #self.text_path = os.path.join(self.dataset_general_path + config['text_path'])

        # load data
        debunk_data, news_data = self.load_data()
        self.debunk_data_file = os.path.join(self.dataset_path, 'debunk_data.pkl')
        self.debunk_dataset = None
        #if os.path.exists(self.debunk_data_file):
            #self.debunk_dataset = pickle.load(open(self.debunk_data_file, 'rb'))
        #else:
            #self.debunk_dataset = self.load_debunk_data(debunk_data)

        if self.dataset_mode == 'event':
            self.dataset_path_split = os.path.join(self.dataset_path + '/' + self.dataset_mode + '/' + str(config['fold']))
            train_path = os.path.join(self.dataset_path_split + '/train.txt')
            test_path = os.path.join(self.dataset_path_split + '/test.txt')
            self.train_dataset = Dataset_load(train_path, config, news_data)
            # Modifying for ablation
            # ablation_dataset_path = '/home/qiumj/Workspace/fvd/dataset/FakeSV-30/data'
            # new_dataset_path_split = os.path.join(
            #     ablation_dataset_path + '/' + self.dataset_mode + '/' + str(config['fold']))
            # test_path = os.path.join(new_dataset_path_split + '/test.txt')
            # config['dataset'] = 'FakeSV-30'
            self.test_dataset = Dataset_load(test_path, config, news_data)
        else:
            self.dataset_path_split = os.path.join(self.dataset_path + '/' + self.dataset_mode)
            train_path = os.path.join(self.dataset_path_split + '/train.txt')
            val_path = os.path.join(self.dataset_path_split + '/val.txt')
            test_path = os.path.join(self.dataset_path_split + '/test.txt')
            self.train_dataset = Dataset_load(train_path, config, news_data)
            self.val_dataset = Dataset_load(val_path, config, news_data)
            # Modifying for ablation
            # ablation_dataset_path = '/home/qiumj/Workspace/fvd/dataset/FakeSV-50/data'
            # new_dataset_path_split = os.path.join(
            #     ablation_dataset_path + '/' + self.dataset_mode)
            # test_path = os.path.join(new_dataset_path_split + '/test.txt')
            # config['dataset'] = 'FakeSV-50'
            self.test_dataset = Dataset_load(test_path, config, news_data)

    def load_data(self):
        data_complete_path = os.path.join(self.dataset_path + '/data.json')
        data_complete = pd.read_json(data_complete_path, orient='records', dtype=False, lines=True)
        masker_path = os.path.join(self.dataset_general_path + '/masker.json')
        masker = pd.read_json(masker_path).T.reset_index(drop=False)
        masker['video_id'] = masker['index'].apply(lambda x: x.split('.')[0])
        data_complete = pd.merge(data_complete, masker, on='video_id', how='inner')
        self.num_events = data_complete['keywords'].nunique()
        data_complete['keywords'] = pd.Categorical(data_complete['keywords'])
        data_complete['keywords_code'] = data_complete['keywords'].cat.codes
        news_data = data_complete[data_complete['annotation'] != '辟谣']
        data_debunk = data_complete[data_complete['annotation'] == '辟谣']
        return data_debunk, news_data

    def load_debunk_data_feature(self, video_id):
        # audio
        audioframes = pickle.load(open(os.path.join(self.audio_path, video_id + '.pkl'), 'rb'))
        audioframes = audioframes.cpu()
        audioframes, _ = pad_frame_sequence_dataset(50, audioframes)
        # frame data
        frames = pickle.load(open(os.path.join(self.frame_path, video_id + '.pkl'), 'rb'))
        frames = torch.FloatTensor(frames)
        frames, _ = pad_frame_sequence_dataset(83, frames)
        # c3d data
        c3d = h5py.File(self.c3d_path + video_id + ".hdf5", "r")[video_id]['c3d_features']
        c3d = torch.FloatTensor(c3d)
        c3d, _ = pad_frame_sequence_dataset(83, c3d)
        # text
        text = pickle.load(open(os.path.join(self.text_path, video_id + '.pkl'), 'rb'))
        title = text['title']
        intro = text['intro']
        comment = text['comment']
        profile = text['profile']
        return {
            'audioframes': audioframes,
            'frames': frames,
            'c3d': c3d,
            'title': title,
            'intro': intro,
            'comment': comment,
            'profile': profile
        }

    def load_debunk_data(self, data_debunk):
        debunk_data = {}
        for data in data_debunk.itertuples():
            if data.keywords_code in debunk_data:
                debunk_data[data.keywords_code][data.video_id] = self.load_debunk_data_feature(data.video_id)
            else:
                debunk_data[data.keywords_code] = {}
                debunk_data[data.keywords_code][data.video_id] = self.load_debunk_data_feature(data.video_id)

        new_debunk_data = {}
        # 遍历原始字典的第一级索引（事件名）
        for event_name, event_data in debunk_data.items():
            # 创建一个空的二级字典，用于存储事件的特征
            event_features = {}
            # 遍历事件中的贴子名和特征名
            for post_name, post_data in event_data.items():
                for feature_name, feature_tensor in post_data.items():
                    # 将特征张量添加到对应的特征名中
                    if feature_name in event_features:
                        event_features[feature_name] = torch.cat(
                            (event_features[feature_name], feature_tensor.unsqueeze(0)), dim=0)
                    else:
                        event_features[feature_name] = feature_tensor.unsqueeze(0)

            # 将二级字典添加到新的二级字典中
            new_debunk_data[event_name] = event_features

        with open(self.debunk_data_file, 'wb') as file:
            pickle.dump(new_debunk_data, file)
        return new_debunk_data




