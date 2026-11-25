# FloodVision AI — Model Card

## Model Information

- Model: MobileNetV3 U-Net
- Version: 1.0.0
- Experiment: EXP007
- Parameters: 1,034,057
- Best checkpoint: Epoch 18
- Framework: PyTorch

## Intended Use

Research-oriented water segmentation using
dual-polarization Sentinel-1 SAR imagery.

## Input Requirements

- VV and VH SAR channels
- Float32 tensors
- 512 x 512 tiles
- Training-compatible preprocessing

## Evaluation Results

| Metric | Score |
|---|---:|
| Validation IoU | 83.46% |
| Validation Dice | 90.98% |
| Independent Test IoU | 51.76% |
| Independent Test Dice | 68.21% |
| Independent Test Precision | 56.62% |
| Independent Test Recall | 85.78% |
| Pakistan IoU | 27.28% |
| Spain IoU | 69.76% |

## Computational Efficiency

- CPU inference latency: 71.99 ms
- GPU inference latency: 9.35 ms
- Production artifact: approximately 4.13 MiB

Benchmarks are hardware-specific and do not
represent end-to-end application latency.

## Known Limitations

- Significant geographical domain shift.
- Reduced performance on Pakistan test samples.
- False positives in water-free regions.
- Reduced accuracy on sparse water coverage.
- Single-seed experimental evaluation.
- Limited independent geographical test coverage.
- Water segmentation does not necessarily
  distinguish permanent water from new flooding.

## Responsible Use

This model is a research prototype.

Predictions require independent verification
and must not be used as the sole basis for
emergency response, evacuation, or other
safety-critical decisions.

## Model Selection

MobileNetV3 U-Net achieved the highest validation
and independent test IoU among the four evaluated
architectures, while providing the lowest measured
CPU and GPU inference latency.
