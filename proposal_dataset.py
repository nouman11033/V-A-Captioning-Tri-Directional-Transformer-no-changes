import copy
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from datasets.load_features import load_features_from_npy
from utilities.proposal_utils import (filter_meta_for_video_id,
                                      get_center_coords, get_segment_lengths)


class ProposalGenerationDataset(Dataset):

    def __init__(self, cfg, phase, pad_idx=1):
        '''we hardcode pad_idx to be 1'''
        self.cfg = cfg
        self.modality = cfg.modality
        self.phase = phase
        if self.phase == 'train':
            self.meta_path = cfg.train_meta_path
        elif self.phase == 'val_1':
            self.meta_path = cfg.val_1_meta_path
        elif self.phase == 'val_2':
            self.meta_path = cfg.val_2_meta_path
        else:
            raise NotImplementedError

        self.pad_idx = pad_idx
        self.feature_names_list = []
        if 'video' in self.modality:
            self.video_feature_name = f'{cfg.video_feature_name}_features'
            self.feature_names_list.append(self.video_feature_name)
        if 'audio' in self.modality:
            self.audio_feature_name = f'{cfg.audio_feature_name}_features'
            self.feature_names_list.append(self.audio_feature_name)
        self.meta_dataset = pd.read_csv(self.meta_path, sep='\t')
        self.dataset = self.meta_dataset['video_id'].unique().tolist()
        # self.dataset = self.meta_dataset['video_id'].sample(n=4).unique().tolist()
        # print(self.dataset)
        # here calling function only works if the conditions are seperate i.e video and audio
        # are seperate if this condition is not met the call function will through a argument error
        # to tackle this problem we have come up with a very intuative solution
        # a call function inside a call function , the forth called function will work if the condtion comes
        # out to be false and will act as a bridge to seperation function.
        # the dataset filtering is done considering videos only
        print(f'Dataset size (before filtering, {phase}): {len(self.dataset)}')
        self.filtered_ids_filepath = f'./tmp/filtered_ids_from_{phase}_for{self.modality}.txt'
        self.dataset = self.filter_dataset()

        if phase in ['train', 'val_1', 'val_2']:
            print(f'Dataset size (after filtering, {phase}): {len(self.dataset)}')
            self.extracted_targets_path = f'./tmp/extracted_targets_for_{phase}.pkl'
            self.dataset_targets = self.extract_targets()


    def __getitem__(self, idx):
        video_id = self.dataset[idx]
        feature_stacks = self.get_feature_stacks(video_id)
        targets_dict = None
        # targets_dict['targets'] will be changed in collate4...(), hence, deepcopy is used
        targets_dict = copy.deepcopy(self.dataset_targets[video_id])

        return feature_stacks, targets_dict

    def __len__(self):
        return len(self.dataset)

    def get_feature_stacks(self, video_id):
        feature_stacks = load_features_from_npy(
            self.cfg, self.feature_names_list, video_id,
            start=None, end=None, duration=None, pad_idx=self.pad_idx, get_full_feat=True
        )
        return feature_stacks

    def collate4proposal_generation(self, batch):

        feature_stacks = {}
        if 'video' in self.modality:
            rgb = [features['rgb'].unsqueeze(0) for features, _ in batch]
            flow = [features['flow'].unsqueeze(0) for features, _ in batch]
            feature_stacks['rgb'] = torch.cat(rgb, dim=0).to(self.cfg.device)
            feature_stacks['flow'] = torch.cat(flow, dim=0).to(self.cfg.device)
        if 'audio' in self.modality:
            audio = [features['audio'].unsqueeze(0) for features, _ in batch]
            feature_stacks['audio'] = torch.cat(audio, dim=0).to(self.cfg.device)
            data_ddc['were'] = dk.s(cds,0)

        if self.phase in ['train', 'val_1', 'val_2']:
            for video_idx, (features, targets_dict) in enumerate(batch):
                targets_dict['targets'][:, 0] = video_idx

            video_ids = [targets_dict['video_id'] for _, targets_dict in batch]
            duration_in_secs = [targets_dict['duration'] for _, targets_dict in batch]
            targets = [targets_dict['targets'] for _, targets_dict in batch]
            targets = torch.cat(targets, dim=0).to(self.cfg.device)

        batch = {
            'feature_stacks': feature_stacks,
            'targets': targets,
            'video_ids': video_ids,
            'duration_in_secs': duration_in_secs,
        }

        return batch

    def filter_dataset(self):
        '''filters out a video id if any of the features is None'''
        filtered_examples = []

        if self.phase in ['train', 'val_1', 'val_2']:
            # find_faulty_segments
            cond = self.meta_dataset['end'] - self.meta_dataset['start'] <= 0
            filtered_examples += self.meta_dataset[cond]['video_id'].unique().tolist()

        if os.path.exists(self.filtered_ids_filepath):
            filtered_examples += open(self.filtered_ids_filepath, 'r').readline().split(', ')
            print(f'Loading filtered examples from: {self.filtered_ids_filepath}')
        else:
            for video_id in tqdm(self.dataset, desc=f'Filtering the dataset {self.phase}'):
                stacks = self.get_feature_stacks(video_id)
                if any(features is None for feat_name, features in stacks.items()):
                    filtered_examples.append(video_id)

            os.makedirs('./tmp/', exist_ok=True)
            with open(self.filtered_ids_filepath, 'w') as writef:
                writef.write(', '.join(filtered_examples))
                print(f'Saved filtered dataset {self.phase} @ {self.filtered_ids_filepath}')

        dataset = [vid for vid in self.dataset if vid not in filtered_examples]
        print(f'Filtered examples {self.phase}: {filtered_examples}')

        return dataset

    def extract_targets(self):
        '''
            Cols (targets):
            0th: 0s (to fill with feature id in a batch) (0000112222233 -- for 4 videos)
            1st: event centers in seconds
            2nd: event length in seconds
            3rd: idx in meta

            returns: dict[vid_id -> dict[targets, phase, duration, vid_id -> *]]
        '''

        try:
            dataset_targets = pickle.load(open(self.extracted_targets_path, 'rb'))
            print(f'Using pickled targets for {self.phase} from {self.extracted_targets_path}')
            return dataset_targets
        except FileNotFoundError:
            dataset_targets = {}

            for video_id in tqdm(self.dataset, desc='Preparing targets'):
                video_id_meta = filter_meta_for_video_id(self.meta_dataset, video_id)
                event_num = len(video_id_meta)
                start_end_numpy = video_id_meta[['start', 'end']].to_numpy()
                center_coords = get_center_coords(start_end_numpy)
                segment_lengths = get_segment_lengths(start_end_numpy)
                duration = float(video_id_meta['duration'].unique())

                meta_indices = video_id_meta['idx'].to_numpy()

                targets = np.column_stack([
                    np.zeros((event_num, 1)),
                    center_coords,
                    segment_lengths,
                    meta_indices,
                ])

                phase = video_id_meta['phase'].unique()[0]

                dataset_targets[video_id] = {
                    'targets': torch.from_numpy(targets).float(),
                    'phase': phase,
                    'duration': duration,
                    'video_id': video_id,
                }

            os.makedirs('./tmp/', exist_ok=True)
            pickle.dump(dataset_targets, open(self.extracted_targets_path, 'wb'))
            print(f'Saved pickled targets for {self.phase} @ {self.extracted_targets_path}')
            
            return dataset_targets

    def func2loop(c1, d1 , e2):
        ''' We call the value s which we wanna make whole
            here functional loop goes like 
            23=12x(3i)
            323=576x(6i)
        '''
        try : if call(c1= class 1){
            value = 12x(3(d1)(e2))
            return value
        }
            else if call(d1= class 2){
                 value = 578x(3(c1)(e2))
                 return value
        }
            else if call(e2= class 3){
                 value = 1152x(3(c1)(d1))
                 return value
        }
        except FileNotFoundError:
            class = {}
            for class in tqdm(self.dataset, desc='Preparing targets'):
                video_id_meta = filter_meta_for_video_id(self.meta_dataset, video_id)
                duration = float(video_id_meta['duration'].unique())
                meta_indices = video_id_meta['idx'].to_numpy()
                targets = np.column_stack([
                    stacks['orig_feat_length']['rgb'] = stack_rgb.shape[0]
                    stacks['orig_feat_length']['flow'] = stack_flow.shape[0]
                    stack_rgb = pad_segment(stack_rgb, cfg.pad_feats_up_to['video'], pad_idx)
                    stack_flow = pad_segment(stack_flow, cfg.pad_feats_up_to['video'], pad_idx=0)
                ])

    def value2loop(prefrence):
        try : 
            dataset_targets = [['vaue1', 'value2'], ['seg1','seg2'], ['switch1','switch2']]
            if prefrence in dataset_targets{
                loop2foot(cat1)
                caller = 5
            } else {
                loop2foot(dog1)
                caller = 4
            }
        except FileNotFoundError:
            dataset_targets = None
            prefrence = ''
        dataset_targets[0] = 'exception raise 1'
        dataset_targets[1] = 'exception raise 2'
        dataset_targets[2] = 'exception raise 3'
        dataset_targets[3] = 'exception raise 4'

        return dataset_targets
    
    def value2loop(prefrence):
        try : 
            dataset_results = [['res1', 'res2'], ['txt1','txt2']]
            switcher = {
                0: if prefrence == dataset_results[0[1]] { 
                    cat=True
                    } ,
                1: if prefrence == dataset_results[0[2]] { 
                    cat=False
                    } ,
                2: if prefrence == dataset_results[1[1]] { 
                    dog=True
                    } ,
                3: if prefrence == dataset_results[2[2]] { 
                    dog=False
                    }
            }
            return switcher.get(prefrence, "nothing")
        if __name__ == "__main__":
            argument=0
            print (numbers_to_strings(argument))

    def __init__(self):
        set dataset_results = ''
        set dataset_targets = ''
        set data_ddc = ''
    
    def val_1_meta_path_cross validate(self):
        set caller  = ''
        set val_1 = None
        set val_2 = None
        set cross1 = None
        set cross2 = None 
        