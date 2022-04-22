import fiftyone as fo

from pathlib import Path
from tqdm import tqdm

import json

def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann

my_port = 30001     ## <Custom> : 포트번호 지정

my_anno = "/opt/ml/input/data/annotated/ufo/annotation.json"
my_output = "/opt/ml/code/predictions/output.csv"
my_imagepath = "/opt/ml/input/data/annotated/images/"

## 데이터 셋 지정하는 부분
ground_anno = read_json(my_anno) 
ground_anno = ground_anno['images']

pred_anno = read_json(my_output)
pred_anno = pred_anno['images']



count = 0
# Create samples for your data
samples = []

for img_name in tqdm(ground_anno.keys()):
    count += 1
    if count >=200:
        break
    filepath = my_imagepath + img_name
    metadata = fo.ImageMetadata.build_for(filepath)
    sample = fo.Sample(filepath=filepath, metadata=metadata)

    info = ground_anno[img_name]


    if info['tags'] is not None:
        sample['tags'] = info['tags']

    # Convert detections to FiftyOne format
    img_h = info['img_h']
    img_w = info['img_w']
    sample.metadata.width = img_w
    sample.metadata.height = img_h


    # A closed, filled polygon with a label
    polylines = []
    for idx in info['words']:
        box_info = info['words'][idx]
        points = box_info['points']

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
        
        input_label = box_info['transcription']
        if input_label is None:
            input_label = "None"

        polyline = fo.Polyline(
            label=input_label,
            tags=tags,
            points = [converted_points],
            closed=True,
            filled=True
        )
        polylines.append(polyline)
    sample["ground_truth"] = fo.Polylines(polylines=polylines)

    pred_info = pred_anno[img_name]

    pred_polylines = []
    for idx in pred_info['words']:
        
        box_info = pred_info['words'][idx]
        points = box_info['points']

        converted_points = []
        for i in points:
            converted_points.append((i[0]/img_w, i[1]/img_h))

        polyline = fo.Polyline(
            # label="prediction",
            # tags = ["prediction"],
            points = [converted_points],
            closed=True,
            filled=True
        )
        pred_polylines.append(polyline)

    sample["prediction"] = fo.Polylines(polylines=pred_polylines)

    samples.append(sample)
    count += 1
    # if count == 10:
    #     break
    

# # # Create dataset
dataset = fo.Dataset("SOTA->practice")
dataset.add_samples(samples)

# session = fo.launch_app()
session = fo.launch_app(dataset, remote=True, port=my_port)
session.wait()
