import glob
import fiftyone as fo

from pathlib import Path

import json

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

my_port = 30001     ## <Custom> : 포트번호 지정
my_anno = "/opt/ml/input/data/ICDAR17_Korean/ufo/valid_v2.json"  ## <Custom> : Annotation json파일 지정
# my_img = "/opt/ml/input/data/ICDAR17_Korean/images/*.jpg" ## <Custom> : 이미지 파일 지정

# my_anno = "/opt/ml/input/data/dataset/annotation.json"  ## <Custom> : Annotation json파일 지정
# my_anno = "/opt/ml/input/data/dataset/squared_practice.json"
my_img = "/opt/ml/input/data/ICDAR17_Korean/images/*.jpg" ## <Custom> : 이미지 파일 지정
# my_img2 = "/opt/ml/input/data/dataset/*.jpeg" ## <Custom> : 이미지 파일 지정

## 데이터 셋 지정하는 부분
annotations = read_json(my_anno) 
annotations = annotations['images']


count = 0
# Create samples for your data
samples = []
my_glob = glob.glob(my_img)
print(my_glob)
# my_glob = glob.glob(my_img) + glob.glob(my_img2)
for filepath in my_glob:
    metadata = fo.ImageMetadata.build_for(filepath)
    sample = fo.Sample(filepath=filepath, metadata=metadata)


    # sample = fo.Sample(filepath=filepath)

    
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
    sample.metadata.width = img_w
    sample.metadata.height = img_h


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
        if box_info['orientation'] is not None:
            tags.append(box_info['orientation'])
        if box_info['language'] is not None and len(box_info['language']) != 0:
            for i in box_info['language']:
                tags.append(i)
        if box_info['word_tags'] is not None:
            for i in box_info['word_tags']:
                tags.append(i)
        tags.append(str(len(converted_points)))
        # print(len(converted_points))
        input_label = box_info['transcription']
        if input_label is None:
            input_label = "None"
            
        print(tags)
        print(box_info)
        polyline = fo.Polyline(
            label=input_label,
            tags=tags,
            points = [converted_points],
            closed=True,
            filled=True
        )
        polylines.append(polyline)
    sample["ground_truth"] = fo.Polylines(polylines=polylines)
    samples.append(sample)
    count += 1
    # if count == 10:
    #     break
    


# # # Create dataset
dataset = fo.Dataset("squared_practice12345")
dataset.add_samples(samples)

# session = fo.launch_app()
session = fo.launch_app(dataset, remote=True, port=my_port)
session.wait()
