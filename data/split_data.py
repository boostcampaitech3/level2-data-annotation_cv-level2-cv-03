"""
같은 version의 train-validation split을 경로 설정만 달리해서 복수의 데이터셋에 적용하기 위해 작성하였습니다.
시각화를 통한 데이터 및 작업 구조 파악은 split_data.ipynb 이용을 권장합니다.

데이터 경로는 하드코딩되어있으며 수정해서 사용하시면 됩니다.

작성자: 김대근 (daegunkim0425@gmail.com)
"""

# Load modules
import os
import pandas as pd
import json
import numpy as np
import random
import seaborn as sns
import matplotlib as mpl

from pathlib import Path
from sklearn.model_selection import train_test_split
from copy import deepcopy
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold 
from tqdm import tqdm

## Fix random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# custom functions
def save_json(data: dict, file_nm: str, dir_path: str):
    with open(os.path.join(dir_path, file_nm), 'w') as outfile:
        json.dump(data, outfile)


def get_box_size(quads):
    """ 단어 영역의 사각형 좌표가 주어졌을 때 가로, 세로길이를 계산해주는 함수.
    TODO: 각 변의 길이를 단순히 max로 처리하기때문에 직사각형에 가까운 형태가 아니면 약간 왜곡이 있다.
    Args:
        quads: np.ndarray(n, 4, 2) n개 단어 bounding-box의 4개 점 좌표 (단위 pixel)
    Return:
        sizes: np.ndarray(n, 2) n개 box의 (height, width)쌍
    """
    dists = []
    for i, j in [(1, 2), (3, 0), (0, 1), (2, 3)]: # [right(height), left(height), upper(width), lower(width)] sides
        dists.append(np.linalg.norm(quads[:, i] - quads[:, j], ord=2, axis=1))

    dists = np.stack(dists, axis=-1).reshape(-1, 2, 2) # shape (n, 2, 2) widths, heights into separate dim
    return np.rint(dists.mean(axis=-1)).astype(int)

def rectify_poly(poly, direction, img_w, img_h):
    """일반 polygon형태인 라벨을 크롭하고 rectify해주는 함수.
    Args:
        poly: np.ndarray(2n+4, 2) (where n>0), 4, 6, 8
        image: np.ndarray opencv 포멧의 이미지
        direction: 글자의 읽는 방향과 진행 방향의 수평(Horizontal) 혹은 수직(Vertical) 여부
    Return:
        rectified: np.ndarray(2, ?) rectify된 단어 bbox의 사이즈.
    """
    
    n_pts = poly.shape[0]
    assert n_pts % 2 == 0
    if n_pts == 4:
        size = get_box_size(poly[None])
        h = size[:, 0] / img_h
        w = size[:, 1] / img_w
        return np.stack((h,w))

    def unroll(indices):
        return list(zip(indices[:-1], indices[1:]))

    # polygon하나를 인접한 사각형 여러개로 쪼갠다.
    indices = list(range(n_pts))
    if direction == 'Horizontal':
        upper_pts = unroll(indices[:n_pts // 2]) # (0, 1), (1, 2), ... (4, 5)
        lower_pts = unroll(indices[n_pts // 2:])[::-1] # (8, 9), (7, 8), ... (6, 7)

        quads = np.stack([poly[[i, j, k, l]] for (i, j), (k, l) in zip(upper_pts, lower_pts)])
    else:
        right_pts = unroll(indices[1:n_pts // 2 + 1]) # (1, 2), (2, 3), ... (4, 5)
        left_pts = unroll([0] + indices[:n_pts // 2:-1]) # (0, 9), (9, 8), ... (7, 6)

        quads = np.stack([poly[[i, j, k, l]] for (j, k), (i, l) in zip(right_pts, left_pts)])

    sizes = get_box_size(quads)
    if direction == 'Horizontal':
        h = sizes[:, 0].max() / img_h
        widths = sizes[:, 1]
        w = np.sum(widths) / img_w
        return np.stack((h,w)).reshape(2,-1)
        #return np.stack((h,w))
    elif direction == 'Vertical':
        heights = sizes[:, 0]
        w = sizes[:, 1].max() / img_w
        h = np.sum(heights) / img_h
        return np.stack((h,w)).reshape(2,-1)
    else:
        h = sizes[:, 0] / img_h
        w = sizes[:, 1] / img_w
        return np.stack((h,w),-1)
    

def get_image_dfs(data):
    df = {}
    df['image'] = []
    df['word_counts'] = []
    df['image_width'] = []
    df['image_height'] = []
    df['image_tags'] = []
    img_tags = []

    quads = []
    polys = []
    seq_length = []
    hor_sizes = []
    ver_sizes = []
    irr_sizes = []
    languages = []
    orientation = []
    word_tags = []
    aspect_ratio = []
    ver_string = []

    bbox_properties = []
    
    for image_key, image_value in data["images"].items():
        df['image'].append(image_key)
        img_w = image_value['img_w']
        img_h = image_value['img_h']
        df['image_width'].append(img_w)
        df['image_height'].append(img_h)
        df['image_tags'].append(image_value['tags'])
        df['image_tags']= [['None'] if v is None else v for v in df['image_tags']] # our data does not inlcude multi-tag images 
        word_ann = image_value['words']
        count_ill = 0 
        for word in word_ann.values():
            if word['illegibility']== False:
                orientation.append(word['orientation'])
                orientation = [v for v in orientation]
                seq_length.append(len(word['transcription']))
                languages.append(word['language'])
                languages = [['None'] if v is None else v for v in languages] # our data does not inlcude multi-language words
                if word['word_tags'] != None:
                    word_tags.extend(word['word_tags'][:])
                elif word['word_tags']== None:
                    word_tags.append('None')
                poly = np.int32(word['points'])
                size = rectify_poly(poly, word['orientation'], img_w, img_h)
                if word['orientation'] == 'Horizontal':
                    hor_sizes.append(size)
                    bbox_properties.append([image_key, size, 'Horizontal'])
                    # print(image_key, size, 'Horizontal')
                elif word['orientation'] == 'Vertical':
                    ver_sizes.append(size)
                    bbox_properties.append([image_key, size, 'Vertical'])
                    # print(image_key, size, 'Vertical')
                else:
                    irr_sizes.append(size)
                    bbox_properties.append([image_key, size, 'Irregular'])
            else:
                count_ill += 1

        df['word_counts'].append(len(word_ann)-count_ill)


    all_sizes = hor_sizes + ver_sizes + irr_sizes
    quad_area = [all_sizes[i][0]*all_sizes[i][1] for i in range(len(all_sizes))]
    total_area = []
    for s in quad_area:
        if s.shape[0] == 1:
            total_area.append(np.sum(s[0])) 
        else:
            total_area.append(np.sum(s))

    hor_aspect_ratio = [hor_sizes[i][1]/hor_sizes[i][0] for i in range(len(hor_sizes))]
    ver_aspect_ratio = [ver_sizes[i][1]/ver_sizes[i][0] for i in range(len(ver_sizes))]

    image_df = pd.DataFrame.from_dict(df)
    bbox_df = pd.DataFrame(data=bbox_properties,
                          columns=['image', 'size', 'orientation'])
    
    bbox_df['aspect_ratio'] = bbox_df.apply(lambda x: (x['size'][1]/x['size'][0])[0], axis=1)
    
    return image_df, bbox_df


def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann    

def auto_binning(data:pd.Series, min_elements=10, log_transform=True, print_outlier_percent=False):
    # The goal of this function is binning a continuous-valued 1-d sequence for stratified splitting
    """
    Approximate the distribution of continuous 1-d sequence as a discrete distribution by pandas cut function

    Args:
        data: Input 1-d sequence as a pandas format.
        min_elements: The number of miminum elements for a bin.
                      It is used for finding moderate number of bins.
        log_transform: Whether using natural log transformation before binning or not.
        print_outlier_percent: Whether printing outlier ratio among whole dataset or not.

    Returns:
        1-d pandas series with discrete class
    """   
    assert type(data) is pd.Series
    
    def bind_outlier(data: pd.Series, print_outlier_percent=False):
        """
        Outlier binding function for 1-d array.
        Outlier is determined by statistical convention using 25 & 75 percentile and IQR 

        Args:
            data: Input 1-d sequence as a pandas format.
            print_outlier_percent: Whether printing outlier ratio among whole dataset or not.

        Returns:
            It returns the outlier binded 1-d array.
        """    
        assert type(data) is pd.Series

        cnt_outlier = 0

        q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
        lower = q1 - 1.5*(q3-q1)
        upper = q3 + 1.5*(q3-q1)

        new_data = deepcopy(data)

        cnt_outlier += len(new_data[new_data <= lower])
        new_data[new_data <= lower] = lower

        cnt_outlier += len(new_data[new_data >= upper])
        new_data[new_data >= upper] = upper

        if print_outlier_percent:
            print(f'Outliers account for [{100*cnt_outlier/len(data):.3f}]% of total data')

        return new_data
    
    # binding outlier
    if log_transform:
        binded_data = bind_outlier(np.log(data), print_outlier_percent)
    else:
        binded_data = bind_outlier(data, print_outlier_percent)
        
    num_bins = 10
    max_iter_limit = 9999
    iter_cnt = 0
    while True:
        dist_approx = pd.cut(binded_data, bins=num_bins, labels=np.arange(num_bins))
        min_elem_cnt = dist_approx.value_counts().min() # Miminum number of elements for a bin
        
        if min_elem_cnt < min_elements:
            print(f'Minimum # of elements for a bin is [{min_elem_cnt}]')
            print(f'Number of bins [{num_bins}]')
            break
        
        if iter_cnt > max_iter_limit:
            print('Iteration exceeded limit')
            raise RuntimeError
        
        num_bins += 1
        iter_cnt += 1
    
    return dist_approx


def split_dataset(root_dir: str, path_data: str, data_nm: str):

    data_org = read_json(os.path.join(root_dir, path_data, data_nm))

    # Get image properties
    image_df, bbox_df = get_image_dfs(data_org)

    # Get validation version 1
    # Split images at random
    X_train_v1, X_valid_v1, y_train_v1, y_valid_v1 = \
    train_test_split(
        image_df.image,
        image_df.image,
         test_size=0.2,
          shuffle=True,
           random_state=seed)

    train_v1 = {'images': {k: v for k, v in data_org['images'].items() if k in X_train_v1.values}}
    valid_v1 = {'images': {k: v for k, v in data_org['images'].items() if k in X_valid_v1.values}}

    # Save validation version 1
    data_list = [train_v1,
             valid_v1
            ]
    file_nm_list = ['train_v1.json',
                    'valid_v1.json',
                ]

    for data, file_nm in zip(data_list, file_nm_list):
        save_json(data, file_nm, dir_path=os.path.join(root_dir, path_data))

    # Get validation version 2
    # Remove bbox with aspect ratio 0 or nan if it exists
    cnt_aspect_ratio_zero = bbox_df[bbox_df.aspect_ratio==0]
    print(f'# of BBox with aspect zero [{len(cnt_aspect_ratio_zero)}]')

    cnt_aspect_ratio_null = bbox_df[bbox_df.aspect_ratio.isnull()]
    print(f'# of BBox with aspect zero [{len(cnt_aspect_ratio_null)}]')

    bbox_df = bbox_df[bbox_df.aspect_ratio != 0]
    bbox_df = bbox_df[~bbox_df.aspect_ratio.isnull()]

    # Reindexing
    bbox_df = bbox_df.reset_index().drop(columns='index')

    bbox_df.aspect_ratio.describe()
    bbox_df.aspect_ratio.isnull().sum()

    # Auto binning
    bbox_df['aspect_class'] = auto_binning(bbox_df.aspect_ratio, print_outlier_percent=False)

    cv_val_v2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) 

    for train_idx_v2, valid_idx_v2 in cv_val_v2.split(bbox_df.aspect_class, bbox_df.aspect_class, bbox_df.image): 
        pass
        
    print(f'Length train idx [{len(train_idx_v2)}]')
    print(f'Length valid idx [{len(valid_idx_v2)}]')

    # Devide images by aspect ratio class
    train_image_v2 = set(bbox_df.image[train_idx_v2])
    valid_image_v2 = set(bbox_df.image[valid_idx_v2])

    # Check v2 exclusivity
    print(f'Intersection between train images v2 & valid images v2 : [{train_image_v2.intersection(valid_image_v2)}]')

    train_image_v2 = list(train_image_v2)
    valid_image_v2 = list(valid_image_v2)

    train_v2 = {'images': {k: v for k, v in data_org['images'].items() if k in train_image_v2}}
    valid_v2 = {'images': {k: v for k, v in data_org['images'].items() if k in valid_image_v2}}

    print(f"Length train v2 [{len(train_v2['images'])}] || valid v2 [{len(valid_v2['images'])}]")

    # Save validation version 2
    data_list_v2 = [train_v2,
             valid_v2
            ]
    file_nm_list_v2 = ['train_v2.json',
                    'valid_v2.json',
                ]

    for data, file_nm in zip(data_list_v2, file_nm_list_v2):
        save_json(data, file_nm, dir_path=os.path.join(root_dir, path_data))




#%% Main script
root_dir = '../input/data/'

path_ICDAR17_Korean = 'ICDAR17_Korean/ufo/'
path_ICDAR17_New  = 'ICDAR17_New/ufo/'

data_nm_ICDAR17_Korean = 'train.json'
data_nm_ICDAR17_New = 'merged.json'

list_root_dir = [
    root_dir,
    root_dir,
    ]

list_data_path = [
    path_ICDAR17_Korean,
    path_ICDAR17_New,
    ]

list_data_nm = [
    data_nm_ICDAR17_Korean,
    data_nm_ICDAR17_New
    ]

path_data = path_ICDAR17_Korean
data_nm = data_nm_ICDAR17_Korean

for root, data_dir, name in tqdm(zip(list_root_dir, list_data_path, list_data_nm)):
    split_dataset(root_dir=root_dir, path_data=data_dir, data_nm=name)