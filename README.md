# vllm-ascend-quant-hust

## Description 

A repository for post-training quantization on Ascend NPUs, supporting 8-bit, 4-bit, and mixed-precision quantization for large language models.

## Usage

### Environment setup

```bash
conda activate vllm-hust-dev
pip install -r requiremnets.txt
```

### W8A8 quantization

W8A8 quantization run command：

```bash
msmodelslim quant \
  --model_path /root/models/Qwen2.5-7B-Instruct \
  --save_path /root/models/Qwen2.5-7B-Instruct-w8a8-test \
  --device npu \
  --model_type Qwen2.5 \
  --config_path /root/workspace1/vllm-ascend-quant-hust/qwen_w8a8.yaml \
  --trust_remote_code True
```

Note: replace root with real root.

### PPL evaluation

PPL evaluation run command：

```bash
python test_ppl.py --model_path /root/models/Qwen2.5-7B-w8a8-Instruct-smooth
```

Note: run this command in a free npu.
