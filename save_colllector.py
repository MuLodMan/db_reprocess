from PIL import Image
import numpy as np
import json

class data_collection():
    def __init__(self,r_dir:str,maxH:int,maxW:int) -> None:
        self.r_dir = r_dir
        maxSize =  maxH if maxH > maxW else maxW
        self.c_size = [maxSize / 4 , maxSize / 2 , maxSize]
        self.shared_data = [[],[],[]]
        self.file_list = 'train_list.json'


    def save_total_info(self , dir_name):
        try:
            with open(dir_name+self.file_list,'w') as file:
                json.dump({'Csize':self.c_size,'img_list':self.shared_data},file)
        except Exception as e:
            print("An error occurred while saving the dictionary as JSON:", str(e))


    def save_to_image(self,image_array,filename:str,mode:str,size_group:int = -1):
        if mode == 'RGB':
            np_array = image_array.numpy()
            canvas = Image.fromarray(np_array[:,:,::-1] , mode = mode) #bgr to rgb
            canvas.save(self.r_dir + 'padded_train_images\\'+ filename)
        elif mode == 'numpy':
            np.savez_compressed(self.r_dir +'concatMask\\'+filename,masks = image_array)
        else:raise Exception("bad mode")
        if type(filename) == str and size_group > -1:
            self.shared_data[size_group].append(filename)