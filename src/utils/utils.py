"""Common utils that can be shared across different tasks.
"""
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Any, Callable, T, Dict, Text

OUTPUT_CACHE_DIR = "YOUR_OUTPUT_CACHE_DIR"
DATABASE_DIR = "YOUR_DATABASE_DOWNLOAD_DIR"
MODEL_CACHE_DIR = "YOUR_MODEL_CACHE_DIR_FOR_HUGGINGFACE"

def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)

def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

def paginate_func(
    items: List[Any],
    page_size: int,
    func: Callable[..., T],
    combination: Callable[[List[T]], T],
    silent: bool = True
) -> T:
    
    results = []
    
    iterator = range(0, len(items), page_size)
    if not silent:
        iterator = tqdm(iterator, desc="Paginating")
        
    for i in iterator:
        results.append(
            func(
                items[i:i+page_size]
            )
        )
        
    return combination(results)

def save_cache(model_name, verifier, retrieval):
    if verifier:
        verifier.save_cache()
    if "retrieval" in model_name:
        for k, v in retrieval.items():
            v.save_cache()

class QuantizedLinearInt8(torch.nn.Module):
    '''
    A simple but effictive implmenetion of Int8 quantization for linear layers.
    The weights are quantized and stored as Int8, which saves ~50% of the gpu memory.
    During the forwared pass, the weights are de-quantized back to fp16 to do multiplication.
    Pros:
        - saves ~50% of the gpu memory
        - accurate quantization because only the weights are quantized, and the weights don't suffer
            from the "outliers" issue mentioned in the LLM.int8 paper; only the activations do.
        - high precision results beacuse the multiplication is done in fp16
        - much faster than LLM.int8
    Cons:
        - a bit slower because of the added computation of dequantization in each forward pass. In practice, the slowdown
            is not large because in the generation application, gpu utilization is not very high.
    '''
    def __init__(self, linear_layer):
        super().__init__()
        self.bias = linear_layer.bias

        weight_bit_width = 8
        weight = linear_layer.weight

        self.weight_scale = torch.nn.Parameter(
            (weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half(),
        )
        # print(self.weight_scale.max().item(), self.weight_scale.min().item(), self.weight_scale.mean().item())
        # if self.weight_scale.max().item() > 0.002:
            # print(self.weight_scale.max().item())
        self.weight = torch.nn.Parameter(
            torch.round(weight.float() / self.weight_scale[:, None]).char(),
            requires_grad=False
            )

    def forward(self, x):
        weight = self.weight.half() * self.weight_scale[:, None]
        return torch.nn.functional.linear(x, weight, self.bias)

def get_memory_footprint(model, return_buffers=True):
    """
    Get the memory footprint of a model. This will return the memory footprint of the current model in bytes.
    Useful to benchmark the memory footprint of the current model and design some tests. Solution inspired from the
    PyTorch discussions: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2
    Arguments:
        return_buffers (`bool`, *optional*, defaults to `True`):
            Whether to return the size of the buffer tensors in the computation of the memory footprint. Buffers
            are tensors that do not require gradients and not registered as parameters. E.g. mean and std in batch
            norm layers. Please see: https://discuss.pytorch.org/t/what-pytorch-means-by-buffers/120266/2
    """
    mem = sum([param.nelement() * param.element_size() for param in model.parameters()])
    if return_buffers:
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
        mem = mem + mem_bufs
    return mem


def ـreplace_linear_with_int8linear(model, modules_to_not_convert="lm_head"):
    for name, module in model.named_children():
        ـreplace_linear_with_int8linear(module, modules_to_not_convert)

        if isinstance(module, torch.nn.Linear) and name != modules_to_not_convert:
            model._modules[name] = QuantizedLinearInt8(linear_layer=module)
    return


def convert_model_to_int8_on_gpu(model, device):
    """
    Quantize a model to int8 and move it to GPU using a simple method.
    """
    if 'cuda' not in device:
        raise ValueError(f"Target device should be a gpu. Device {device} is not supported")

    model.half()

    memory_before_quantization = get_memory_footprint(model)  # without lm_head

    ـreplace_linear_with_int8linear(model)  # replace `Linear` with `QuantizedLinearInt8`

    model.to(device=device)
    memory_after_quantization = get_memory_footprint(model)  # without lm_head

    saving = round(100 * memory_after_quantization/memory_before_quantization)
    memory_before_quantization = round(memory_before_quantization / 2**30, 2)  # rounding for printing
    memory_after_quantization = round(memory_after_quantization / 2**30, 2)  # rounding for printing

    print(f'Quantization memory - before: {memory_before_quantization} GB, after: {memory_after_quantization} GB ({saving}% of the size before)')
    return model

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg