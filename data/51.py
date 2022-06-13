import glob
import fiftyone as fo

from pathlib import Path

import json

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

my_port = 3001     ## <Custom> : 포트번호 지정
my_anno = "/opt/ml/input/data/ICDAR17_Korean/ufo/train.json"  ## <Custom> : Annotation json파일 지정
my_img = "/opt/ml/input/data/ICDAR17_Korean/images/*.jpg" ## <Custom> : 이미지 파일 지정

## 데이터 셋 지정하는 부분
annotations = read_json(my_anno) 
annotations = annotations['images']

# Create samples for your data
samples = []
for filepath in glob.glob(my_img):
    sample = fo.Sample(filepath=filepath)

    
    k = filepath.split('/')
    k = k[-1].split("\\")
    img_path = k[-1]

    info = annotations[img_path]
    if info['tags'] is not None:
        sample['tags'] = info['tags']

    # Convert detections to FiftyOne format
    detections = []
    num_boxes = len(info['words'])
    img_h = info['img_h']
    img_w = info['img_w']
    

    # A closed, filled polygon with a label
    polylines = []
    for idx in info['words']:
        box_info = info['words'][idx]
        points = box_info['points']
        etc = box_info
        del(etc['points'])

        converted_points = []
        for i in points:
            converted_points.append((i[0]/img_w, i[1]/img_h))

        tags = []
        tags.append(str(box_info['illegibility']))
        tags.append(box_info['orientation'])
        for i in box_info['language']:
            tags.append(i)
        if box_info['word_tags'] is not None:
        # for i in box_info['word_tags']:
            tags.append(i)
            

        polyline = fo.Polyline(
            label=box_info['transcription'],
            tags=tags,
            points = [converted_points],
            closed=True,
            filled=True,
        )
        polylines.append(polyline)
    sample["polylines"] = fo.Polylines(polylines=polylines)
    samples.append(sample)
    


# # # Create dataset
dataset = fo.Dataset("my-detection-dataset")
dataset.add_samples(samples)

# session = fo.launch_app()
session = fo.launch_app(dataset, remote=True, port=my_port)
session.wait()
