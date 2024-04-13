import torch.utils.data as data
import cv2

#processes
#set_current_working_space
from os import chdir
from os.path import dirname,abspath 
chdir(dirname(abspath(__file__)))
#set_current_working_space
from make_data import uniform_data
from make_map import makeShrinkMap , MakeBorderMap , MakeGrabCut , make_padding , get_punish
import math

import json
import numpy as np
from save_colllector import data_collection

#data_loader
from torch.utils.data import DataLoader


class ImageDataset(data.Dataset):
    r'''Dataset reading from images.
    Args:
        Processes: A series of Callable object, which accept as parameter and return the data dict,
            typically inherrited the `DataProcess`(data/processes/data_process.py) class.
    '''

    def __init__(self, data_dir=None, data_list=None, cmd={},):
        self.data_dir = data_dir or self.data_dir
        self.data_list = data_list or self.data_list
        if 'train' in self.data_list:
            self.is_training = True
        else:
            self.is_training = False
        self.debug = cmd.get('debug', False)
        self.image_paths = []
        self.gt_paths = []
        self.img_dir = 'train_images\\'
        self.gt_dir  = 'train_gts\\'
        self.p_M_H = self.p_M_W = 0
        self.get_all_samples()
        self.uni_data = uniform_data()
        self.make_shrink = makeShrinkMap()
        self.make_boarder = MakeBorderMap()
        self.make_grab = MakeGrabCut()
        self.load_padding = make_padding(maxH=self.p_M_H , maxW=self.p_M_W)
        
    def get_all_samples(self):
        image_paths = self.image_paths
        gt_paths = self.gt_paths
        with open(self.data_list, mode='r',encoding='utf-8') as fid:
            if self.is_training:
                if 'icdar2019' in self.data_list:
                    file_json_list = json.load(fid)
                    maxH = file_json_list['maxH']
                    maxW = file_json_list['maxW']
                    self.p_M_H = math.ceil(maxH / 32) * 32
                    self.p_M_W = math.ceil(maxW / 32) * 32
                    for file_str in file_json_list['img_list']:
                        image_paths.append(file_str.strip())
                        gt_paths.append(file_str.strip().split('.')[0]+'.json')
                elif 'icdar2015' in self.data_list: #for icdar 2015 datasets
                    file_json_list = json.load(fid)
                    maxH = file_json_list['maxH']
                    maxW = file_json_list['maxW']
                    self.p_M_H = math.ceil(maxH / 32) * 32
                    self.p_M_W = math.ceil(maxW / 32) * 32
                    for file_str in file_json_list['img_list']:
                        image_paths.append(file_str.strip())
                        gt_paths.append(file_str.strip()+'.txt')
            else:
                if 'icdar2019' in self.data_list:
                    file_json_list = json.load(fid)
                    maxH = file_json_list['maxH']
                    maxW = file_json_list['maxW']
                    self.p_M_H = math.ceil(maxH / 32) * 32
                    self.p_M_W = math.ceil(maxW / 32) * 32
                    for file_str in file_json_list['img_list']:
                        image_paths.append(file_str.strip())
                        gt_paths.append(file_str.strip().split('.')[0]+'.json')
                elif 'icdar2015' in self.data_list:
                    file_json_list = json.load(fid)
                    maxH = file_json_list['maxH']
                    maxW = file_json_list['maxW']
                    self.p_M_H = math.ceil(maxH / 32) * 32 
                    self.p_M_W = math.ceil(maxW / 32) * 32
                    for file_str in file_json_list['img_list']:
                        image_paths.append(file_str.strip())
                        gt_paths.append(file_str.strip()+'.txt')
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def load_ann(self):
            res = []
            if 'icdar2019' in self.data_dir: #icdar2019
                for gt in self.gt_paths:
                    lines = []
                    with open(self.data_dir+self.gt_dir+gt,mode='r',encoding='utf-8') as raw_str:
                        ann_dict = json.load(raw_str)
                        ann_lines = ann_dict['lines']
                        for ann_line in ann_lines:
                            item = {}
                            points = ann_line['points']
                            temp_points = []
                            for p_i in range(0,len(points),2):
                                temp_points.append([int(points[p_i]),int(points[p_i+1])]) #[x,y]
                            item['poly'] = temp_points
                            if ann_line['ignore']: item['text'] = '###' #ignore
                            else: item['text'] = ann_line['transcription']
                            lines.append(item)
                    res.append(lines)
                return res

            elif 'icdar2015' in self.data_dir:
                    for gt in self.gt_paths:
                        lines = []
                        with open(self.data_dir+self.gt_dir+gt,mode='r',encoding='utf-8') as raw_str:
                            ann_lines = raw_str.readlines()
                            for ann_line in ann_lines:
                                item = {}
                                parts = ann_line.strip().strip('\ufeff\xef\xbb\xbf').split(',')
                                label = parts[-1]
                                poly = []
                                for p_i in range(0,len(parts[:8]),2):
                                    poly.append([int(parts[p_i]),int(parts[p_i+1])]) #[x,y]
                                item['poly'] = poly
                                item['text'] = label
                                lines.append(item)
                        res.append(lines)
                    return res     


    def __getitem__(self, index):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(self.data_dir+''+self.img_dir+image_path, cv2.IMREAD_COLOR)#.astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['is_trainning'] = True
        else:
            data['filename'] = image_path.split('\\')[-1]
            data['is_trainning'] = False
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target  


        data = self.uni_data(data=data)
        data = self.make_shrink(data=data)
        data = self.make_boarder(data=data)
        data = self.make_grab(data=data)
        data = self.load_padding(data = data)       



        return data

    def __len__(self):
        # return 20 #for debug samples
       return len(self.image_paths)


    # -1) filename data_id image(origin) lines(list of array which contains ploy and text).
    # 0) augment_data.AugmentDetectionData is rotate with annotation. 
    # 1) random_crop_data.RandomCropData with crop scale with annotation. 
    # 2) polygons,ignore_tags (makeICDARData)
    # 3) shrink(MakeSegDetectionData )
    # 4) padding(makeBoarderMap)
    # 5) grabMap(MakeGrabCut)
    # 6) normalize(normalize_image.NormalizeImage)
    # 7) FilterKeys


if __name__ == '__main__':
    image_dataset = ImageDataset(data_dir='icdar2019\\',data_list='icdar2019\\train_loader.json')
    c_min,c_mid,c_max = image_dataset.load_padding.getCanvasSizes()
    data_collector = data_collection(r_dir='icdar2019\\',c_min=c_min,c_mid=c_mid,c_max=c_max)
    # image_dataset = ImageDataset(data_dir='icdar2015\\',data_list='icdar2015\\train_loader.json')
    # c_min,c_mid,c_max = image_dataset.load_padding.getCanvasSizes()
    # data_collector = data_collection(r_dir='icdar2015\\',maxH = image_dataset.p_M_H ,maxW = image_dataset.p_M_W)
    dataloader = DataLoader(image_dataset,batch_size=1)
    for idx,batch in enumerate(dataloader):
        cur_filename = batch['filename'][0]
        concat_mask = np.concatenate((batch['gt'] , batch['mask'] , batch['thresh'] , batch['punish']) , axis = 0)
        data_collector.save_to_image(image_array = concat_mask , filename = cur_filename.strip() , mode='numpy',size_group = batch['s_group'][0].item())  # after removing image suffix
        if idx >0 and idx % 200==0:
            p_c , unp_c = get_punish()
            data_collector.save_total_info(dir_name = 'icdar2019\\',punish_c = p_c,unpunish_c = unp_c)


        