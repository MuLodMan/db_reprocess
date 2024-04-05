from collections import OrderedDict
from data_process import DataProcess
import numpy as np
class uniform_data(DataProcess):
    def __init__(self):
        # self.load_all(**kwargs)
       pass
    def process(self, data):
        polygons  = []
        ignore_tags = []
        assert type(data.get('image',None)) == np.ndarray
        for line in data['lines']:
            assert type(line['poly']) == list
            polygons.append(np.array(line['poly']))
            ignore_tags.append(line['text'] == '###')
        # data.update(shape=image.shape[:2])
        return OrderedDict(image = data['image'],
                           polygons = polygons,
                           ignore_tags = ignore_tags,
                           filename = data.get('filename',''))