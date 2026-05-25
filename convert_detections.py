import json
import sys

def convert_detections(input_file, output_file, ball_size=20.0):
    """Convert detection JSON to the format expected by ball_detection_metrics.py"""

    with open(input_file, 'r') as f:
        records = json.load(f)

    detections = []
    for record in records:
        # Skip if no detection
        if record.get('x') is None or record.get('y') is None:
            continue

        # Convert from center (x, y) to bbox [x1, y1, x2, y2]
        x, y = record['x'], record['y']
        half_size = ball_size / 2.0
        bbox_xyxy = [x - half_size, y - half_size, x + half_size, y + half_size]

        # Convert bbox_xyxy to bbox_xywh (x, y, width, height)
        x1, y1, x2, y2 = bbox_xyxy
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

        # Use confidence if available, else use 0.5
        score = record.get('confidence') if record.get('confidence') is not None else 0.5
        if score is None:
            score = 0.5

        detection = {
            "image_id": record['frame_idx'] - 1,  # Convert from 1-indexed to 0-indexed
            "bbox_xywh": bbox_xywh,
            "score": float(score),
            "stage": "final",
        }
        detections.append(detection)

    output = {"detections": detections}

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Converted {len(detections)} detections from {input_file} to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.json', '_converted.json')
        convert_detections(input_file, output_file)
    else:
        convert_detections('clip1_baseline.json', 'clip1_baseline_converted.json')
        convert_detections('clip1_v6_11.json', 'clip1_v6_11_converted.json')
