from asyncio import new_event_loop
from pathlib import Path
import numpy as np

import json
from copy import deepcopy

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

my_anno = "/opt/ml/input/data/dataset/ufo/annotation.json"  ## <Custom> : Annotation json파일 지정    
anno = read_json(my_anno) 
annotations = deepcopy(anno)

new_dict = {'images':{}}


for img_name in annotations['images']:
    if not "0" in annotations['images'][img_name]['words']:
        ## 'words'에 아무런 데이터도 없는 경우 제외
        continue
    for idx in annotations['images'][img_name]['words']:
        box = annotations['images'][img_name]['words'][idx]['points']
        if len(box) == 4:
            continue
        elif len(box) > 4 and len(box) % 2 == 0:
            npl = np.array(box)
            min_x = npl[:,0].min()
            max_x = npl[:,0].max()
            min_y = npl[:,1].min()
            max_y = npl[:,1].max()
            annotations['images'][img_name]['words'][idx]['points'] = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            ## 사각형으로 만드는 부분
        else:
            print("WTF?")

    new_dict['images'][img_name] = annotations['images'][img_name]


with open('train_v3.json', 'w') as f:
    json_string = json.dump(new_dict, f, indent=2)