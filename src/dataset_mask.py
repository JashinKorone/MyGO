import math
import os
import argparse
import random
import pickle
import json

import h5py
import pandas as pd
import torch
import numpy as np


class LegacyMasker:
    def __init__(self, config):
        self.config = config

    def masking(self):
        self.random_walker()

    def masking_pkl_audio(self, ind, osl, masked, it):
        mask_item_fp = osl[ind] + '/' + it
        masked_item_fp = masked[ind] + '/' + it
        with open(mask_item_fp, 'rb') as f:
            self.audio_data = pickle.load(f)
            self.audio_data_shape = self.audio_data.shape
            masked_data = torch.zeros(self.audio_data_shape)
        with open(masked_item_fp, 'wb') as f:
            pickle.dump(masked_data, f)

    def masking_h5py_c3d(self, ind, osl, masked, it):
        it_split_list = it.split('.')
        it_split_list[-1] = 'hdf5'
        it = '.'.join(it_split_list)

        mask_item_fp = osl[ind] + '/' + it
        masked_item_fp = masked[ind] + '/' + it

        f = h5py.File(mask_item_fp, 'r')
        key = list(f.keys())
        assert len(key) == 1
        f_1 = f[key[0]]

        key_1 = list(f_1.keys())
        assert len(key_1) == 1
        f_2 = f_1[key_1[0]][:]

        self.c3d_data = np.zeros_like(f_2)
        with h5py.File(masked_item_fp, 'w') as f_save:
            f_save.create_dataset(key[0] + '/' + key_1[0], data=self.c3d_data)

    def masking_pkl_txt(self, ind, osl, masked, it):
        mask_item_fp = osl[ind] + '/' + it
        masked_item_fp = masked[ind] + '/' + it
        with open(mask_item_fp, 'rb') as f:
            data = pickle.load(f)
            masked_dict = {}
            for key in data.keys():
                value_array = data[key]
                if torch.is_tensor(value_array):
                    value_zero = torch.zeros(size=value_array.shape)
                else:
                    value_zero = np.zeros_like(value_array)
                update_dict = {key: value_zero}
                masked_dict.update(update_dict)
            self.txt_data = masked_dict
        with open(masked_item_fp, 'wb') as f:
            pickle.dump(self.txt_data, f)

    def summarizer(self, sample, ind):
        self.id_fp = config['masked_dataset_fp'] + '/' + 'id.txt'
        with open(self.id_fp, 'w') as f:
            for line in sample:
                f.write(str(line) + '\n')
        self.mask_fp = config['masked_dataset_fp'] + '/' + 'mask.txt'
        with open(self.mask_fp, 'w') as f:
            for line in ind:
                f.write(str(line) + '\n')

    def random_walker(self):
        os_len_list = []
        os_fold_list = []
        masked_fold_list = []
        for fold in config['fea']:
            fea_fold = config['dataset_fp'] + '/' + fold
            masked_fea_fold = config['masked_dataset_fp'] + '/' + fold
            masked_fold_list.append(masked_fea_fold)
            os_len_list.append(len(os.listdir(fea_fold)))
            os_fold_list.append(fea_fold)
        # Confirm all data features have the same number
        assert len(set(os_len_list)) == 1
        print('Loading ' + str(os_len_list[0]) + ' video information. ')

        # Sampling videos to masked
        demo_path = os_fold_list[0]
        demo_list = os.listdir(demo_path)
        sample_number = int(config['mask_ratio'] * len(demo_list))
        sampled_list = random.sample(demo_list, sample_number)

        # Masking sampled videos
        mask_indicator = 0
        mask_ind_list = []
        for item in sampled_list:
            if mask_indicator % 3 == 0:
                # Masking audio features
                self.masking_pkl_audio(0, os_fold_list, masked_fold_list, item)
                mask_ind_list.append(0)
            elif mask_indicator % 3 == 1:
                # Masking c3d features
                self.masking_h5py_c3d(1, os_fold_list, masked_fold_list, item)
                mask_ind_list.append(1)
            elif mask_indicator % 3 == 2:
                # Masking txt features
                self.masking_pkl_txt(2, os_fold_list, masked_fold_list, item)
                mask_ind_list.append(2)
            else:
                raise ValueError
            mask_indicator += 1
        assert len(mask_ind_list) == len(sampled_list)


class GCNetMasker:
    def __init__(self, config):
        self.config = config
        self.mask_ratio = config['mask_ratio']
        self.modality = ['audio', 'video', 'social']
        self.fea_list = ['audio', 'c3d', 'txt', 'ptvgg19_frames']
        self.masked_dataset_fp = config['masked_dataset_fp']
        self.dataset_fp = config['dataset_fp']
        self.preserve_ratio = config['preserve_ratio']
        self.min_modality = config['min_modality']
        self.data_json_fp = config['dataset_fp'] + '/data/data.json'
        print('The Current Masker is GCNetMasker...')

    def masking(self):
        # Make Sure the os list to be masked is consistent
        tmp_list1 = []
        tmp_list2 = []
        for fea in self.fea_list:
            fea_fold = self.dataset_fp + '/' + fea
            video_num = len(os.listdir(fea_fold))
            tmp_list1.append(video_num)
            tmp_list2.append(os.listdir(fea_fold))
        assert len(set(tmp_list1)) == 1
        self.video_num = tmp_list1[0]

        self.video_id_list = [i.split('.')[0] for i in tmp_list2[0]]
        # Choose preserved videos
        json_df = pd.read_json(self.data_json_fp, lines=True)

        kwd_set = set(json_df['keywords'].tolist())
        # At least one kept video for every event
        self.videos_to_keep = []
        for keyword in kwd_set:
            tmp_df1 = json_df[(json_df['keywords'] == keyword)]
            kwd_video_num = tmp_df1.shape[0]
            kwd_keep_video_num = math.ceil(kwd_video_num * self.preserve_ratio)
            tmp_list3 = tmp_df1['video_id'].tolist()
            kwd_videos_to_keep = random.sample(tmp_list3, kwd_keep_video_num)
            assert len(kwd_videos_to_keep) >= self.min_modality
            self.videos_to_keep.extend(kwd_videos_to_keep)
        self.videos_to_mask = list(set(self.video_id_list) - set(self.videos_to_keep))
        self.keeping_dict = {}
        keep_dict_value = {'audio': 0, 'video': 0, 'social': 0}
        for video in self.videos_to_keep:
            self.keeping_dict.update({video: keep_dict_value})
        print('The number of preserved videos: ' + str(len(self.videos_to_keep)) + '. The kept ratio is ' +
                      str(len(self.videos_to_keep) / self.video_num))

        # Transferring target mask ratio to actual mask ratio
        self.num_videos_to_keep = len(self.videos_to_keep)
        self.num_videos_to_mask = self.video_num - self.num_videos_to_keep
        mask_modality_sum = int(self.video_num * 3 * self.mask_ratio)
        self.actual_masking_ratio = mask_modality_sum / (self.num_videos_to_mask * 3)
        print('The required masking ratio for all modalities is ' + str(self.actual_masking_ratio))

        # Masking left videos
        self.masking_dict = {}
        for video in self.videos_to_mask:
            mask_idx_dict = {}
            # Masking with ratio of self.mask_ratio
            for modality in self.modality:
                x = random.uniform(0, 1)
                if x < self.actual_masking_ratio:
                    mask_idx = 1
                else:
                    mask_idx = 0
                mask_idx_dict.update({modality: mask_idx})
            # Make sure there is at least one available modality for videos
            if set(list(mask_idx_dict.values())) == {1}:
                insurance_modality = random.choice(self.modality)
                mask_idx_dict.update({insurance_modality: 0})
            self.masking_dict.update({video: mask_idx_dict})

        # Merge Masking and Keeping dicts
        self.masker = {}
        self.masker.update(self.keeping_dict)
        self.masker.update(self.masking_dict)

        # Checking the masking ratio
        mask_modality_sum = 0
        keep_modality_sum = 0
        modality_list = list(self.masker.values())
        for item in modality_list:
            mask_idx_list = list(item.values())
            mask_num = mask_idx_list.count(1)
            keep_num = mask_idx_list.count(0)
            mask_modality_sum += mask_num
            keep_modality_sum += keep_num
        print('The actual masking ratio of the whole dataset is ' +
              str(mask_modality_sum / (mask_modality_sum + keep_modality_sum)))

        # Saving to target
        for vid in self.video_id_list:
            value_dict = self.masker[vid]

            if value_dict['video'] == 1:
                self.masking_video(vid)
            elif value_dict['video'] == 0:
                self.copying_video(vid)
            else:
                raise ValueError('The video masker in vid ' + vid + ' is missing.')

            if value_dict['audio'] == 1:
                self.masking_audio(vid)
            elif value_dict['audio'] == 0:
                self.copying_audio(vid)
            else:
                raise ValueError('The audio masker in vid' + vid + ' is missing.')

            if value_dict['social'] == 1:
                self.masking_social(vid)
            elif value_dict['social'] == 0:
                self.copying_social(vid)
            else:
                raise ValueError('The social masker in vid' + vid + ' is missing.')
        masker_json_path = self.masked_dataset_fp + '/masker.json'
        with open(masker_json_path, 'w') as f:
            json.dump(self.masker, f)

    def masking_video(self, vid):
        c3d_fp = self.dataset_fp + '/c3d/' + vid + '.hdf5'
        ptvgg19_fp = self.dataset_fp + '/ptvgg19_frames/' + vid + '.pkl'
        c3d_masked_fp = self.masked_dataset_fp + '/c3d/' + vid + '.hdf5'
        ptvgg19_masked_fp = self.masked_dataset_fp + '/ptvgg19_frames/' + vid + '.pkl'

        # Reading and masking c3d features
        f = h5py.File(c3d_fp, 'r')
        key = list(f.keys())
        assert len(key) == 1
        f_1 = f[key[0]]
        key_1 = list(f_1.keys())
        assert len(key_1) == 1
        f_2 = f_1[key_1[0]][:]
        c3d_masked = np.zeros_like(f_2)
        with h5py.File(c3d_masked_fp, 'w') as f_save:
            f_save.create_dataset(key[0] + '/' + key_1[0], data=c3d_masked)

        # Reading and masking ptvgg19 features
        with open(ptvgg19_fp, 'rb') as f:
            ptvgg19_feature = pickle.load(f)
            ptvgg19_masked = np.zeros_like(ptvgg19_feature)
        with open(ptvgg19_masked_fp, 'wb') as f:
            pickle.dump(ptvgg19_masked, f)

    def copying_video(self, vid):
        c3d_fp = self.dataset_fp + '/c3d/' + vid + '.hdf5'
        ptvgg19_fp = self.dataset_fp + '/ptvgg19_frames/' + vid + '.pkl'
        c3d_copied_fp = self.masked_dataset_fp + '/c3d/' + vid + '.hdf5'
        ptvgg19_copied_fp = self.masked_dataset_fp + '/ptvgg19_frames/' + vid + '.pkl'

        # Reading and copying c3d features
        f = h5py.File(c3d_fp, 'r')
        key = list(f.keys())
        assert len(key) == 1
        f_1 = f[key[0]]
        key_1 = list(f_1.keys())
        assert len(key_1) == 1
        f_2 = f_1[key_1[0]][:]
        with h5py.File(c3d_copied_fp, 'w') as f_save:
            f_save.create_dataset(key[0] + '/' + key_1[0], data=f_2)

        # Reading and copying ptvgg19 features
        with open(ptvgg19_fp, 'rb') as f:
            ptvgg19_feature = pickle.load(f)
        with open(ptvgg19_copied_fp, 'wb') as f:
            pickle.dump(ptvgg19_feature, f)

    def masking_audio(self, vid):
        audio_fp = self.dataset_fp + '/audio/' + vid + '.pkl'
        audio_masked_fp = self.masked_dataset_fp + '/audio/' + vid + '.pkl'
        with open(audio_fp, 'rb') as f:
            audio_data = pickle.load(f)
            audio_data_shape = audio_data.shape
            audio_masked_data = torch.zeros(audio_data_shape)
        with open(audio_masked_fp, 'wb') as f:
            pickle.dump(audio_masked_data, f)

    def copying_audio(self, vid):
        audio_fp = self.dataset_fp + '/audio/' + vid + '.pkl'
        audio_copied_fp = self.masked_dataset_fp + '/audio/' + vid + '.pkl'
        with open(audio_fp, 'rb') as f:
            audio_data = pickle.load(f)
        with open(audio_copied_fp, 'wb') as f:
            pickle.dump(audio_data, f)

    def masking_social(self, vid):
        social_fp = self.dataset_fp + '/txt/' + vid + '.pkl'
        social_masked_fp = self.masked_dataset_fp + '/txt/' + vid + '.pkl'
        with open(social_fp, 'rb') as f:
            data = pickle.load(f)
            masked_dict = {}
            for key in data.keys():
                value_array = data[key]
                if torch.is_tensor(value_array):
                    value_zero = torch.zeros(size=value_array.shape)
                else:
                    value_zero = np.zeros_like(value_array)
                update_dict = {key: value_zero}
                masked_dict.update(update_dict)
            txt_masked_data = masked_dict
        with open(social_masked_fp, 'wb') as f:
            pickle.dump(txt_masked_data, f)

    def copying_social(self, vid):
        social_fp = self.dataset_fp + '/txt/' + vid + '.pkl'
        social_masked_fp = self.masked_dataset_fp + '/txt/' + vid + '.pkl'
        with open(social_fp, 'rb') as f:
            data = pickle.load(f)
        with open(social_masked_fp, 'wb') as f:
            pickle.dump(data, f)


class GCNetMasker_v2:
    def __init__(self, config):
        self.config = config
        self.mask_ratio = config['mask_ratio']
        self.modality = ['audio', 'vision', 'text']
        self.masked_dataset_fp = config['masked_dataset_fp']
        self.dataset_fp = config['dataset_fp']
        self.data_json_fp = config['dataset_fp'] + '/data/data.json'
        self.masker_fp = config['masker_fp']
        print('The Current Masker is GCNetMasker_v2...')

    def masking(self):
        # Obtaining pkl list in target dir
        file_list = os.listdir(self.dataset_fp)
        self.masking_dict = {}
        mask_count, keep_count = 0, 0
        for file in file_list:
            mask_idx_dict = {}
            # Masking video modality with pre-defined masking ratio
            # 0: kept, 1: masked
            for modality in self.modality:
                x = random.uniform(0, 1)
                if x <= self.mask_ratio:
                    mask_idx = 1
                else:
                    mask_idx = 0
                mask_idx_dict.update({modality: mask_idx})
            # Make sure there is at least one available modality for videos
            if set(list(mask_idx_dict.values())) == {1}:
                insurance_modality = random.choice(self.modality)
                mask_idx_dict.update({insurance_modality: 0})
            self.masking_dict.update({file: mask_idx_dict})
            for idx in list(mask_idx_dict.values()):
                if idx == 1:
                    mask_count += 1
                elif idx == 0:
                    keep_count += 1

        print('The total number of masked modality is ' + str(mask_count))
        print('The total number of kept modality is ' + str(keep_count))
        print('The masking ratio is ' + str(mask_count / (mask_count + keep_count)))

        print('The masker json file path is ' + self.masker_fp)

        for vid in file_list:
            value_dict = self.masking_dict[vid]
            emb_fp = self.dataset_fp + '/' + vid
            masked_emb_fp = self.masked_dataset_fp + '/embs/' + vid

            with open(emb_fp, 'rb') as f:
                data = pickle.load(f)
                for modality in self.modality:
                    if value_dict[modality] == 1:
                        zero_tensor = torch.zeros_like(data[modality])
                        data[modality] = zero_tensor
            with open(masked_emb_fp, 'wb') as f:
                pickle.dump(data, f)
        with open(self.masker_fp, 'w') as f:
            json.dump(self.masking_dict, f)


def read_args():
    config_dict = {}
    # Confirm target dataset
    if args.dataset == 'FakeSV':
        config_dict.update({'dataset_fp': '../dataset/FakeSV/embs'})
    else:
        raise NotImplementedError
    config_dict.update({'mask_ratio': args.mask_ratio})
    config_dict.update({'mask_type': args.mask_type})
    config_dict.update({'fea': ['audio', 'c3d', 'txt', 'ptvgg19_frames']})
    config_dict.update({'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')})
    masked_fp = create_masked_dataset_fp(config_dict)
    config_dict.update({'masked_dataset_fp': masked_fp})
    config_dict.update({'preserve_ratio': args.preserve_ratio})
    config_dict.update({'min_modality': args.min_modality})
    config_dict.update({'masker_fp': masked_fp + '/' + args.masker_fp})

    return config_dict


def create_masked_dataset_fp(conf):
    ratio = conf['mask_ratio']
    dataset = args.dataset
    masked_dataset_fp = '../dataset/' + dataset + '-' + str(int(ratio * 100))
    if os.path.exists(masked_dataset_fp):
        exit(114514)
    if not os.path.exists(masked_dataset_fp):
        os.makedirs(masked_dataset_fp)
        for fea in conf['fea']:
            os.makedirs(masked_dataset_fp + '/' + fea)
        os.makedirs(masked_dataset_fp + '/' + 'embs')
    return masked_dataset_fp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_ratio', type=float, default=0.1, help='ratio of masked features in FakeSV dataset')
    parser.add_argument('--mask_type', type=str, default='GCNet-v2', help='legacy or GCNet')
    parser.add_argument('--dataset', type=str, default='FakeSV', help='Name of the dataset')
    parser.add_argument('--min_modality', type=int, default=1, help='The minimum number of modality in a video')
    parser.add_argument('--preserve_ratio', type=float, default=0, help='The ratio of videos with complete modality')
    parser.add_argument('--masker_fp', type=str, default='masker.json')

    args, _ = parser.parse_known_args()
    config = read_args()

    if config['mask_type'] == 'Legacy':
        legacy_masker = LegacyMasker(config)
        legacy_masker.masking()
    elif config['mask_type'] == 'GCNet':
        gcnet_masker = GCNetMasker(config)
        gcnet_masker.masking()
    elif config['mask_type'] == 'GCNet-v2':
        gcnet_masker_v2 = GCNetMasker_v2(config)
        gcnet_masker_v2.masking()
    else:
        raise NotImplementedError
