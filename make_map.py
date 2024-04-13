import imgaug
import numpy as np

from data_process import DataProcess
import cv2 as cv
import math

from shapely.geometry import Polygon
import pyclipper
#important
unpunished = 2#0.875
punished_weak = 1#0.75
punished_aug = 3#1.125
punished_count = 0
unpunished_count = 0

def get_punish():
    return (punished_count , unpunished_count)
class make_padding(DataProcess):
      size_dict = {}
      def __init__(self,maxH,maxW):
         maxSize =  maxH if maxH > maxW else maxW
         min_size = maxSize / 4
         mid_size = maxSize / 2 
         self.min_c_size = min_size + (32 - min_size % 32)
         self.mid_c_size = mid_size + (32 - mid_size % 32)
         self.max_c_size = maxSize
        
      def process(self, data:dict):
          mask = data.get('mask')
          (Height,Width) = mask.shape
          maxS = Height if Height > Width else Width
          cur_group = 0
          if maxS <= self.min_c_size: #4 *32
             cur_group = 0
          elif maxS <= self.mid_c_size:
             cur_group = 1
          elif maxS <= self.max_c_size:
             cur_group = 2
          else:
              raise Exception("bad maxSize")
          
          data['s_group'] = cur_group
          return data
      
      def getCanvasSizes(self):
          return (self.min_c_size,self.mid_c_size,self.max_c_size)

        
class makeShrinkMap(DataProcess):
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    min_text_size = 8
    shrink_ratio = 0.4

    def __init__(self):
       pass

    def process(self, data):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']

        h, w = image.shape[:2]
        gt = np.zeros((h, w), dtype=np.uint8)
        mask = np.ones((h, w), dtype=np.uint8)
        for i in range(len(polygons)):
            polygon = polygons[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv.fillPoly(mask, [polygon.astype(
                    np.int32)], 0)
                ignore_tags[i] = True
            else:
                shrink_polygon = Polygon(polygon)
                distance = shrink_polygon.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / shrink_polygon.length
                
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(polygon, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:#目的使不符合shrink条件的的polygon失效
                    cv.fillPoly(mask, [polygon.astype(
                        np.int32)], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv.fillPoly(gt, [shrinked.astype(np.int32)], 1)

        if filename is None:
            assert False
        data['polygons'] = polygons
        data['gt'] = gt
        data['mask'] = mask
        data['filename'] = filename
        return data

    
class MakeBorderMap(DataProcess):
    r'''
    Making the border map from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    shrink_ratio = 0.625
    # thresh_min = State(default=0.3)
    # thresh_max = State(default=0.7)

    def __init__(self):
      pass

    def process(self, data):
        r'''
        required keys:
            image, polygons, ignore_tags
        adding keys:
            thresh, thresh_mask
        '''
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        #set new data["padded_polygons"] to padded polygon array 

        canvas = np.zeros(image.shape[:2], dtype=np.uint8)
        padded_rects = []
        padded_polygons = []

        for i in range(len(polygons)):
            if ignore_tags[i]:
                padded_polygons.append([])
                padded_rects.append([])
                continue
            p_polygon = self.make_border_map(polygons[i], canvas=canvas)
            padded_polygons.append(p_polygon)
            padded_rects.append(
                self.make_rect_points(p_polygon)
            )
        data['thresh'] = canvas
        data['padded_rects'] = padded_rects
        data['padded_polygons'] = padded_polygons
        return data

    def make_border_map(self, polygon, canvas):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(polygon, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = padding.Execute(distance)[0]
        (ori_height , ori_width) = canvas.shape
        padded_polygon = np.array(self.__limit_boundary(padded_polygon=padded_polygon,origin_height = ori_height ,origin_width = ori_width))
        cv.fillPoly(img = canvas,pts = [padded_polygon.astype(np.int32)],color=1)
        return padded_polygon
    

    def __limit_boundary(self,padded_polygon,origin_height,origin_width):
       polygon_list = []
       curX = curY =0
       for point in padded_polygon:
         #prevent out of boundary in with x axies
           if point[0] < 0: 
              curX = 0
           elif point[0] >= origin_width: 
              curX = origin_width - 1
           else:curX = point[0]

         #prevent out of boundary in with y axies
           if point[1] < 0: 
              curY = 0
           elif point[1] >= origin_height: 
              curY = origin_height - 1
           else:curY = point[1]
           
           polygon_list.append([curX,curY]) 

       return polygon_list 
    
    def make_rect_points(self,points:np.ndarray):
        minX,minY = maxX,maxY = points[0]
        for idx in range(1,len(points)):
            if points[idx][0] < minX:
                minX = points[idx][0]
            if points[idx][0] > maxX:
                maxX = points[idx][0]

            if points[idx][1] < minY:
                minY = points[idx][1]
            if points[idx][1] > maxY:
                maxY = points[idx][1]
        return [minX,minY,maxX,maxY]



class MakeGrabCut(DataProcess):
    def __init__(self):

        pass
    
    def process(self, data:dict):
          padded_rects =  data.get('padded_rects',[])
          padded_polygons = data.get('padded_polygons',[])
          polygons = data.get('polygons',[])
          global unpunished_count
          global punished_count

          padding_Len = len(padded_rects)
          assert padding_Len == len(polygons) == len(padded_polygons)
          ori_image = data.get('image',None)
          grab_mask = np.zeros(shape=ori_image.shape[:2],dtype=np.uint8)
          for idx in range(0,padding_Len):
                padded_rect = padded_rects[idx]
                padded_polygon = padded_polygons[idx]
                if len(padded_rect)>0:
                    (minX,minY,maxX,maxY) = padded_rect

                    paddedCropedImage = ori_image[minY:maxY,minX:maxX]

                    (paddedCropHeight,paddedCropWidth) = paddedCropedImage.shape[:2]

                    gt_relative_p = polygons[idx] - np.array([minX,minY])

                    padded_relative_p = padded_polygon - np.array([minX,minY])

                    grab_sub_mask = self.make_grab_cutMap(paddedHeight = paddedCropHeight , paddedWidth = paddedCropWidth , gr_points = gt_relative_p , image = paddedCropedImage,pr_polygon = padded_relative_p)

                    if grab_sub_mask is None:
                        unpunished_count = unpunished_count + 1
                        self.fillconflictFreeing( grab_mask , minY=minY , maxY=maxY , minX=minX , maxX=maxX ,
                                                  submask = np.full(shape=(paddedCropHeight,paddedCropWidth), 
                                                                    fill_value = unpunished,dtype=np.uint8))
                    else:
                        # grab_mask[minY : maxY,minX : maxX] = grab_sub_mask
                        punished_count = punished_count + 1
                        self.fillconflictFreeing(grab_mask ,  minY = minY , maxY = maxY , minX = minX , maxX = maxX ,submask = grab_sub_mask)

          self.holing_padded_map(data = data,grab_map = grab_mask)
          return data
    
    def fillconflictFreeing(self, mask:np.ndarray , minY , maxY , minX , maxX , submask:np.ndarray):
        for r in range(minY,maxY):
            for c in range(minX,maxX):
                if mask[ r ][ c ] < submask[r - minY][c - minX]: 
                   mask[ r ][ c ] = submask[r - minY][c - minX]


                   


    def holing_padded_map(self,data:dict,grab_map:np.ndarray):
        padded_map =  data.get('thresh')
        shrink_map = data.get('gt')
        assert np.array_equal(padded_map.shape,shrink_map.shape) and np.array_equal(padded_map.shape,grab_map.shape)
        data['thresh'] = (padded_map - shrink_map) #* (1 - grab_map)
        data['punish'] = grab_map #shrink_map * grab_map
        

    def make_grab_cutMap(self,paddedHeight:int,paddedWidth:int,gr_points:np.ndarray,image:np.ndarray,pr_polygon:np.ndarray):
         bgdModel = np.zeros((1,65),np.float64)
         fgdModel = np.zeros((1,65),np.float64)
      #  result_mask,result_bgdMask,reslut_fgdModel = cv.grabCut(img=image,mask=probility_mask,rect=None,bgdModel=bgdModel,fgdModel=fgdModel,iterCount=5,mode=cv.GC_INIT_WITH_MASK)
      #  return np.where(((result_mask==2)|(result_mask==0)),0,1).astype(np.uint8)
      #  cv.GC_BGD = 0,cv.GC_PR_FGD = 3,cv.GC_FGD = 1,cv.GC_PR_BGD=2
            
         line_mask = np.full(shape = (paddedHeight,paddedWidth) , fill_value = cv.GC_PR_BGD , dtype = np.uint8)#np.full(shape=(paddedHeight,paddedWidth),fill_value=0,dtype = np.uint8) 

         cv.fillPoly(img = line_mask , pts = [np.round(pr_polygon).astype(np.int32)] , color = cv.GC_BGD)#

         cv.fillPoly(img = line_mask , pts = [np.round(gr_points).astype(np.int32)] , color = cv.GC_PR_FGD) #

         polygon_area = (line_mask == cv.GC_PR_FGD).sum()

         if polygon_area <= 0:
             return None  

         result_mask,result_bgdMoedel,reslut_fgdModel = cv.grabCut(img=image,mask=line_mask,rect=None,bgdModel=bgdModel,fgdModel=fgdModel,iterCount=12,mode=cv.GC_INIT_WITH_MASK)
         
         new_mask = np.where((result_mask == cv.GC_BGD)|(result_mask == cv.GC_PR_BGD),0,1)

        #  pr_fgd_mask = np.where(result_mask == cv.GC_PR_FGD, 1, 0)
        #  fgd_mask = np.where(result_mask == cv.GC_FGD, 1, 0)
        #  grab_area = fgd_mask.sum()  
         
         grab_area = new_mask.sum()
         grab_ratio = grab_area/polygon_area
 

        #  print("The grab map ratio is "+ str(grab_ratio))

         if grab_ratio > 0.25 and grab_ratio < 0.9375:
            temp_mask = np.full(shape=(paddedHeight,paddedWidth),fill_value = punished_weak,dtype=np.uint8)
            for r in range(paddedHeight):
                for c in range(paddedWidth):
                    if new_mask[ r ][ c ] == 1:
                        temp_mask[ r ][ c ] = punished_aug 
            return temp_mask
         else : return None 