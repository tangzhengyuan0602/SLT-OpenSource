"""
This file is modified from:
https://github.com/facebookresearch/deit/blob/main/utils.py
"""

# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import io
import os
import time,random
import numpy as np
from collections import defaultdict, deque
import datetime


def _ensure_local_caches():
    """Avoid DeepSpeed/Triton/HF caches on NFS/home (often small or slow).

    Only sets env vars when user hasn't provided them.
    """

    base_cache = os.environ.get("UNI_SIGN_CACHE_DIR", "/tmp/uni_sign_cache")

    def _mkdir(p: str):
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            pass

    def _setdefault_dir(k: str, p: str):
        if not os.environ.get(k):
            os.environ[k] = p
            _mkdir(p)

    _mkdir(base_cache)
    _setdefault_dir("TRITON_CACHE_DIR", os.path.join(base_cache, "triton"))
    _setdefault_dir("HF_HOME", os.path.join(base_cache, "hf"))
    _setdefault_dir("TRANSFORMERS_CACHE", os.path.join(base_cache, "hf", "transformers"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_ensure_local_caches()

import torch
import torch.distributed as dist
import torch.nn.functional as F

import pickle
import gzip
import re


def normalize_zh_text(s: str) -> str:
    """Normalize Chinese sentence text for CE-CSL/CSL_Daily style SLT.

    - Removes all whitespace.
    - Normalizes common full-width punctuations to ASCII, to reduce metric noise.
    """
    if not s:
        return ''
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    s = (
        s.replace("，", ",")
         .replace("。", ".")
         .replace("？", "?")
         .replace("！", "!")
         .replace("：", ":")
         .replace("；", ";")
         .replace("（", "(")
         .replace("）", ")")
    )
    return s


# global definition
import deepspeed

import torch
import torch.nn.functional as F
from torch import Tensor
import argparse
import torch.backends.cudnn as cudnn

import deepspeed.comm as dist
from deepspeed.accelerator import get_accelerator


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def count_parameters_in_MB(model):
    # sum(p.numel() for p in model.parameters() if p.requires_grad)
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters())/1e6

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        print("save ckpt begin")
        torch.save(*args, **kwargs)
        print("save ckpt finish")

def init_distributed_mode(args):
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ.get('LOCAL_RANK', '0')) if use_cuda else -1
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = (args.rank % torch.cuda.device_count()) if use_cuda else -1
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
    else:
        args.dist_backend = 'gloo'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

def init_distributed_mode_ds(args):
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0

    def _nvml_available() -> bool:
        """Return True if NVML shared library is available.

        Some GPU worker images don't ship libnvidia-ml.so.1 inside the container.
        NCCL may fail hard during init/barrier when NVML is missing. In that case,
        we fall back to gloo backend (world_size==1 still uses GPU for compute).
        """
        try:
            import ctypes

            ctypes.CDLL("libnvidia-ml.so.1")
            return True
        except Exception:
            return False
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ.get('LOCAL_RANK', '0')) if use_cuda else -1
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = (args.rank % torch.cuda.device_count()) if use_cuda else -1
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        # Prefer nccl for GPU, but fall back to gloo if NVML is unavailable.
        args.dist_backend = 'nccl' if _nvml_available() else 'gloo'
        if args.dist_backend != 'nccl':
            print('[warn] NVML (libnvidia-ml.so.1) not found; fallback dist backend to gloo', flush=True)
    else:
        args.dist_backend = 'gloo'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    # torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)
    # On CPU-only machines, use gloo backend; otherwise use nccl.
    # Some GPU containers may miss NVML, in which case nccl init/barrier can fail.
    try:
        deepspeed.init_distributed(dist_backend=args.dist_backend)
        torch.distributed.barrier()
    except Exception as e:
        if args.dist_backend != 'gloo':
            print(f'[warn] dist init failed with backend={args.dist_backend}: {e}; retry with gloo', flush=True)
            args.dist_backend = 'gloo'
            deepspeed.init_distributed(dist_backend='gloo')
            torch.distributed.barrier()
        else:
            raise
    setup_for_distributed(args.rank == 0)

def sampler_func(clip, sn, random_choice=True):
    if random_choice:
        f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                range(int(n * i / sn),
                                                                                        max(int(n * i / sn) + 1,
                                                                                            int(n * (
                                                                                                    i + 1) / sn))))
                        for i in range(sn)]
    else:
        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                max(int(
                                                                                                    n * i / sn) + 1,
                                                                                                    int(n * (
                                                                                                            i + 1) / sn))))
                        for i in range(sn)]
    return f(clip)

def cosine_scheduler(base_value, final_value, epochs):
    iters = np.arange(epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return schedule

def cosine_scheduler_func(base_value, final_value, iters, epochs):
    schedule = lambda x: final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * x / epochs))
    return schedule(iters)

def load_dataset_file(filename):
    # Uni-Sign default: gzip'ed pickle dict (e.g. CSL_Daily labels.*)
    # Also support plain CSV for CE-CSL (Number,Translator,Chinese Sentences,Gloss,Note)
    if filename.endswith('.csv'):
        import csv
        data = {}
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                number = (row.get('Number') or '').strip()
                translator = (row.get('Translator') or '').strip()
                text = normalize_zh_text((row.get('Chinese Sentences') or '').strip())
                gloss_raw = (row.get('Gloss') or '').strip()

                # Example:
                # Number=train-00001, Translator=A
                # video_path=train/A/train-00001.mp4
                split = number.split('-', 1)[0] if '-' in number else 'train'
                video_path = f"{split}/{translator}/{number}.mp4"

                gloss = [g for g in gloss_raw.split('/') if g]
                key = f"{number}_{translator}" if translator else number
                data[key] = {
                    'name': key,
                    'gloss': gloss,
                    'text': text,
                    'video_path': video_path,
                }
        return data

    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

def yield_tokens(file_path):
    with io.open(file_path, encoding = 'utf-8') as f:
        for line in f:
            yield line.strip().split()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    
    output = torch.cat(tensors_gather,dim=0)
    return output

def get_train_ds_config(offload,
                        dtype,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name="",
                        args=''):

    device = "cpu" if offload else "none"

    # CPU-only environments: gloo backend does not support bf16 broadcast.
    # Disable mixed precision to keep everything in fp32.
    cpu_only = (not torch.cuda.is_available())
    # Default: fp32 training (mixed precision disabled)
    data_type = "fp16"
    dtype_config = {"enabled": False}
    if not cpu_only:
        if dtype == "fp32":
            # Keep everything in fp32 on GPU.
            # DeepSpeed uses fp32 when fp16/bf16 is disabled.
            data_type = "fp16"
            dtype_config = {"enabled": False}
        elif dtype == "fp16":
            data_type = "fp16"
            dtype_config = {"enabled": True, "loss_scale_window": 100}
        elif dtype == "bf16":
            data_type = "bfloat16"
            dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != get_accelerator().device_count():
            zero_opt_dict["zero_hpz_partition_size"] = get_accelerator(
            ).device_count()
    return {
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        },
    }

def init_deepspeed(args, model, optimizer, lr_scheduler):

    ds_config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
        args=args
    )

    # Allow using a client-provided optimizer (e.g. AdamW) with ZeRO-Offload.
    # Otherwise DeepSpeed requires DeepSpeedCPUAdam when offloading.
    if args.offload:
        ds_config["zero_force_ds_cpu_optimizer"] = False

    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['gradient_clipping'] = args.gradient_clipping

    use_deepspeed = True
    if use_deepspeed:
        print("Using deepspeed to train...")
        print("Initializing deepspeed...")
        # For single-process training (world_size==1), avoid forcing distributed
        # initialization. Some environments trigger MPI discovery (mpi4py import)
        # even when running with a single GPU.
        dist_init_required = bool(getattr(args, "world_size", 1) and getattr(args, "world_size", 1) > 1)
        _wrapped_model, _optimizer, _, _lr_sched = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            config=ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=dist_init_required)
    
    return _wrapped_model, _optimizer, _lr_sched

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    
    cudnn.deterministic = True # Since the input dim is dynamic.
    cudnn.benchmark = False # Since the input dim is dynamic.

def get_args_parser():
    parser = argparse.ArgumentParser('Uni-Sign scripts', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--gradient-accumulation-steps', default=8, type=int)
    parser.add_argument('--gradient-clipping', default=1., type=float)
    parser.add_argument('--epochs', default=20, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # * Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')
    
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=float, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

     # * Baise params
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='bf16',
                        choices=['fp32', 'fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument('--zero_stage',
                        type=int,
                        default=2,
                        help='ZeRO optimization stage for Actor model (and clones).')
    ## low precision
    parser.add_argument('--compute_fp32_loss',
                        action='store_true',
                        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
                        'If specified, loss is calculated in fp32.')
    
    parser.add_argument('--quick_break',
                        type=int,
                        default=0,
                        help='save ckpt per quick_break step')
    
    # RGB branch
    parser.add_argument('--rgb_support', action='store_true',)

    # RGB-pose interaction: score-aware sampling probability (paper default: 0.1)
    parser.add_argument(
        "--rgb_psamp",
        default=0.1,
        type=float,
        help="When --rgb_support is enabled, randomly sample this ratio of low-confidence frames for RGB hand crops (0~1).",
    )
    
    # Pose length
    parser.add_argument("--max_length", default=256, type=int)

    # Text generation / target text length
    parser.add_argument(
        "--tgt_max_length",
        default=50,
        type=int,
        help="Max token length for target sentence tokenization (labels).",
    )
    parser.add_argument(
        "--gen_max_new_tokens",
        default=100,
        type=int,
        help="Generation max_new_tokens for evaluation/inference.",
    )
    parser.add_argument(
        "--gen_num_beams",
        default=4,
        type=int,
        help="Generation num_beams for evaluation/inference.",
    )
    parser.add_argument(
        "--gen_length_penalty",
        default=1.0,
        type=float,
        help="Generation length_penalty for evaluation/inference.",
    )
    parser.add_argument(
        "--gen_no_repeat_ngram_size",
        default=0,
        type=int,
        help="Generation no_repeat_ngram_size for evaluation/inference (0 disables).",
    )
    parser.add_argument(
        "--gen_repetition_penalty",
        default=1.0,
        type=float,
        help="Generation repetition_penalty for evaluation/inference.",
    )
    
    # select dataset
    # NOTE: allow comma-separated datasets for unified training, e.g. "CE-CSL,CSL_News".
    # Keep backward compatibility for single dataset names.
    parser.add_argument(
        "--dataset",
        default="CSL_Daily",
        type=str,
        help="Dataset name (e.g. CSL_Daily) or comma-separated list for combined training (e.g. 'CE-CSL,CSL_News').",
    )

    # combined dataset sampling strategy (only used when --dataset contains ',')
    parser.add_argument(
        "--combined_sampling",
        default="balanced",
        choices=["balanced", "concat"],
        help="Sampling strategy for combined datasets: balanced (interleave) or concat.",
    )

    parser.add_argument(
        "--combined_allow_empty",
        action="store_true",
        help="Allow combined dataset to drop empty sub-datasets (useful while downloading).",
    )

    # CSL-News data availability control
    parser.add_argument(
        "--news_existing_only",
        action="store_true",
        help="For CSL_News, only keep samples whose pose/rgb files already exist on disk.",
    )
    
    # select task
    parser.add_argument("--task", default="SLT", choices=['SLT', "ISLR", "CSLR"])
    
    # select label smooth
    parser.add_argument("--label_smoothing", default=0.2, type=float)

    # online inference
    parser.add_argument("--online_video", default="", type=str)

    # CE-CSL specifics
    parser.add_argument(
        "--ce_csl_existing_pose_only",
        action="store_true",
        help="For CE-CSL, only keep samples whose pose .pkl already exists under pose_format. "
             "This avoids extremely slow on-the-fly pose extraction during training.")

    parser.add_argument(
        "--ce_csl_pose_max_frames",
        default=128,
        type=int,
        help="For CE-CSL on-demand pose extraction, uniformly sample up to N frames per video (0 means all frames).",
    )
    parser.add_argument(
        "--ce_csl_pose_max_workers",
        default=16,
        type=int,
        help="For CE-CSL on-demand pose extraction, number of frame workers per video.",
    )

    # safety valves for quick experiments
    parser.add_argument("--max_train_steps", default=0, type=int,
                        help="Stop training early after N optimizer steps in each epoch (0 means no limit).")
    parser.add_argument("--max_eval_samples", default=0, type=int,
                        help="Evaluate only on first N samples (0 means full eval set).")

    # checkpoint saving
    parser.add_argument(
        "--save_each_epoch",
        action="store_true",
        help="Also save checkpoint_<epoch>.pth every epoch. If not set, only save best_checkpoint.pth (and DeepSpeed ZeRO-3 ckpt dirs).",
    )

    return parser
