import json

ann_path = 'ball_dataset_2L_3600/all/_annotations.coco.json'

with open(ann_path) as f:
    data = json.load(f)

# Drop category id=0 (ball-OK00), shift ball 1->0, ball_out 2->1
remap = {1: 0, 2: 1}

data['categories'] = [
    {'id': 0, 'name': 'ball', 'supercategory': 'none'},
    {'id': 1, 'name': 'ball_out', 'supercategory': 'none'},
]

for ann in data['annotations']:
    ann['category_id'] = remap[ann['category_id']]

with open(ann_path, 'w') as f:
    json.dump(data, f)

counts = {0: 0, 1: 0}
for ann in data['annotations']:
    counts[ann['category_id']] += 1
print(f"Done. ball (0): {counts[0]}, ball_out (1): {counts[1]}")
