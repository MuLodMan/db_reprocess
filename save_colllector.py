from PIL import Image
import numpy as np
import json

class data_collection():
    def __init__(self,r_dir:str,c_min,c_mid,c_max) -> None:
        self.r_dir = r_dir
        self.shared_data = [{'c_size':c_min,'imgl':[]},{'c_size':c_mid,'imgl':[]},{'c_size':c_max,'imgl':[]}]
        self.file_list = 'train_list.json'


    def save_total_info(self , dir_name,punish_c,unpunish_c):
        self.shared_data = sorted(self.shared_data,key=lambda se:len(se['imgl']))
        try:
            with open(dir_name+self.file_list,'w') as file:
                json.dump({'img_list':self.shared_data,'punish_c':punish_c,'unpunish_c':unpunish_c},file)
        except Exception as e:
            print("An error occurred while saving the dictionary as JSON:", str(e))


    def save_to_image(self,image_array,filename:str,mode:str,size_group:int = -1):
        if mode == 'RGB':
            np_array = image_array.numpy()
            canvas = Image.fromarray(np_array[:,:,::-1] , mode = mode) #bgr to rgb
            canvas.save(self.r_dir + 'padded_train_images\\'+ filename)
        elif mode == 'numpy':
            np.savez_compressed(self.r_dir +'train_mask\\'+filename.split('.')[0],masks = image_array)
        else:raise Exception("bad mode")
        if type(filename) == str and size_group > -1:
            self.shared_data[size_group]['imgl'].append(filename)