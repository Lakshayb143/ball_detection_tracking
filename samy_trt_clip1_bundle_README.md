# Samy TRT / PyTorch Clip1 Bundle

This bundle contains:

- TensorRT engine
- TensorRT standalone inference script
- PyTorch checkpoint
- PyTorch standalone inference script
- clip1 ground-truth dataset

## Layout

- `tensorrt/ball1120fp16.engine`
- `tensorrt/inference_trt_ball_standalone.py`
- `pytorch/ball_samy_1120.pth`
- `pytorch/inference_pytorch_ball_standalone.py`
- `data/train/`

The `data/train/` folder contains:

- all clip1 images
- `_annotations.coco.json`

## Example TensorRT Command

```bash
python tensorrt/inference_trt_ball_standalone.py \
  --model_path tensorrt/ball1120fp16.engine \
  --input_path data/train \
  --prediction_path outputs/prediction_trt.txt \
  --output_path outputs/output_trt.mp4 \
  --confidence 0.01 \
  --input_size 1120 \
  --fps 30
```

## Example PyTorch Command

```bash
python pytorch/inference_pytorch_ball_standalone.py \
  --model_path pytorch/ball_samy_1120.pth \
  --input_path data/train \
  --prediction_path outputs/prediction_pytorch.txt \
  --output_path outputs/output_pytorch.mp4 \
  --confidence 0.01 \
  --input_size 1120 \
  --fps 30 \
  --optimize_for_inference
```


