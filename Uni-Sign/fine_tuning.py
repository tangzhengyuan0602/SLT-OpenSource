import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset, S2T_Dataset_news, S2T_Dataset_combined
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from SLRT_metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
from config import *

def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)

    print(f"Creating dataset:")
        
    # Support combined training by passing comma-separated datasets, e.g. "CE-CSL,CSL_News".
    if isinstance(args.dataset, str) and (',' in args.dataset):
        train_data = S2T_Dataset_combined(
            args=args,
            phase='train',
            datasets=args.dataset,
            sampling=getattr(args, 'combined_sampling', 'balanced'),
        )
    elif args.dataset == 'CSL_News':
        train_data = S2T_Dataset_news(path=train_label_paths[args.dataset], args=args, phase='train')
    else:
        train_data = S2T_Dataset(path=train_label_paths[args.dataset], args=args, phase='train')
    print(train_data)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=True)
        
    if isinstance(args.dataset, str) and (',' in args.dataset):
        test_data = S2T_Dataset_combined(
            args=args,
            phase='test',
            datasets=args.dataset,
            sampling=getattr(args, 'combined_sampling', 'balanced'),
        )
    elif args.dataset == 'CSL_News':
        test_data = S2T_Dataset_news(path=test_label_paths[args.dataset], args=args, phase='test')
    else:
        test_data = S2T_Dataset(path=test_label_paths[args.dataset], args=args, phase='test')
    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)

    if "How2Sign" not in args.dataset:
        if isinstance(args.dataset, str) and (',' in args.dataset):
            dev_data = S2T_Dataset_combined(
                args=args,
                phase='dev',
                datasets=args.dataset,
                sampling=getattr(args, 'combined_sampling', 'balanced'),
            )
        elif args.dataset == 'CSL_News':
            dev_data = S2T_Dataset_news(path=dev_label_paths[args.dataset], args=args, phase='dev')
        else:
            dev_data = S2T_Dataset(path=dev_label_paths[args.dataset], args=args, phase='dev')
        print(dev_data)
        # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
        dev_sampler = torch.utils.data.SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    collate_fn=dev_data.collate_fn,
                                    sampler=dev_sampler,
                                    pin_memory=args.pin_mem)
    else:
        dev_dataloader = test_dataloader

    print(f"Creating model:")
    model = Uni_Sign(args=args)

    # Support CPU-only environments (e.g., devbox without NVIDIA driver).
    # Keep GPU behavior unchanged when CUDA is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # Deepspeed broadcast requires contiguous tensors
    for param in model.parameters():
        if isinstance(param, torch.Tensor):
            param.data = param.data.contiguous()

    def _maybe_resolve_deepspeed_checkpoint(finetune_path: str):
        """Resolve DeepSpeed checkpoint base_dir + tag from a path.

        Supported inputs:
        - <output_dir>/checkpoint_0 (tag dir)
        - <output_dir> (contains a 'latest' file)
        """
        if not finetune_path:
            return None
        try:
            p = Path(finetune_path)
        except Exception:
            return None
        if not p.exists() or not p.is_dir():
            return None

        # Case 1: user passes the tag directory itself.
        # e.g. .../checkpoint_0/zero_pp_rank_0_mp_rank_00_model_states.pt
        if any(p.glob("*model_states.pt")):
            return str(p.parent), p.name

        # Case 2: user passes the output_dir which contains 'latest'.
        latest = p / "latest"
        if latest.exists() and latest.is_file():
            try:
                tag = latest.read_text(encoding="utf-8").strip()
            except Exception:
                tag = latest.read_text().strip()
            if tag:
                return str(p), tag

        # Case 3: output_dir without 'latest' but has exactly one tag subdir.
        candidates = []
        for sub in p.iterdir():
            if sub.is_dir() and any(sub.glob("*model_states.pt")):
                candidates.append(sub.name)
        if len(candidates) == 1:
            return str(p), candidates[0]

        return None

    def _normalize_state_dict_keys(sd):
        # DeepSpeed conversion utilities may produce keys prefixed with "module.".
        if isinstance(sd, dict) and len(sd) > 0:
            keys = list(sd.keys())
            if all(isinstance(k, str) and k.startswith('module.') for k in keys):
                return {k[len('module.'):]: v for k, v in sd.items()}
        return sd

    ds_ckpt = _maybe_resolve_deepspeed_checkpoint(args.finetune)

    # If finetune points to a DeepSpeed checkpoint dir, we must load it via
    # engine.load_checkpoint() AFTER deepspeed.initialize().
    if args.finetune != '' and ds_ckpt is None:
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        ckpt = torch.load(args.finetune, map_location='cpu')
        state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        state_dict = _normalize_state_dict_keys(state_dict)

        # Some pretrained weights (e.g. stage2 with RGB branch) may contain extra
        # parameters not used in the current setting. Prefer strict load, but
        # fallback to non-strict to reuse the shared backbone.
        try:
            ret = model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"[warn] strict load_state_dict failed, fallback to strict=False: {e}")
            ret = model.load_state_dict(state_dict, strict=False)

        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    
    model_without_ddp = model
    if args.distributed:
        # CPU-only / no-CUDA environments: DDP should not be constructed with device_ids.
        if torch.cuda.is_available() and getattr(args, 'gpu', -1) >= 0:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.gpu],
                find_unused_parameters=True,
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=True,
            )
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    # In eval-only mode, avoid initializing DeepSpeed (may require mpi4py/NCCL init).
    # Use plain PyTorch model for single-process evaluation.
    if args.eval:
        if utils.is_main_process():
            if args.task != "ISLR" and "How2Sign" not in args.dataset:
                print("📄 dev result")
                dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
                print("dev_stats:", json.dumps(dev_stats, ensure_ascii=False))
            print("📄 test result")
            test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')
            print("test_stats:", json.dumps(test_stats, ensure_ascii=False))
        return

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )
    
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module.module
    # print(model_without_ddp)
    print(optimizer)

    if ds_ckpt is not None:
        base_dir, tag = ds_ckpt
        print('***********************************')
        print(f'Load DeepSpeed Checkpoint... base_dir={base_dir}, tag={tag}')
        print('***********************************')
        try:
            load_path, client_state = model.load_checkpoint(base_dir, tag=tag)
            print(f"DeepSpeed checkpoint loaded: {load_path}")
            if client_state is not None and isinstance(client_state, dict) and len(client_state) > 0:
                print(f"DeepSpeed client_state keys: {list(client_state.keys())}")
        except TypeError:
            # Compatibility with older DS versions.
            load_path = model.load_checkpoint(base_dir, tag)
            print(f"DeepSpeed checkpoint loaded: {load_path}")

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    if args.task == "CSLR":
        max_accuracy = 1000
    
    # (eval-only handled above)
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        # Optional per-epoch checkpointing (can consume lots of disk on large models).
        # Best checkpoint is saved separately below.
        if args.output_dir and getattr(args, 'save_each_epoch', False):
            # For ZeRO-3, normal state_dict saving will be sharded/empty.
            # Save DeepSpeed checkpoint directory instead, and convert later via
            # `python -m deepspeed.utils.zero_to_fp32 <ckpt_dir> <fp32.pth>`.
            if getattr(args, 'zero_stage', 2) == 3 and hasattr(model, 'save_checkpoint'):
                tag = f'checkpoint_{epoch}'
                model.save_checkpoint(str(output_dir), tag=tag)
            else:
                checkpoint_path = output_dir / f'checkpoint_{epoch}.pth'
                utils.save_on_master({'model': get_requires_grad_dict(model_without_ddp)}, checkpoint_path)

        # single gpu inference
        if utils.is_main_process():
            dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev')
            test_stats = evaluate(args, test_dataloader, model, model_without_ddp, phase='test')

            if args.task == "SLT":
                # Select best checkpoint by dev BLEU-4.
                if max_accuracy < dev_stats["bleu4"]:
                    max_accuracy = dev_stats["bleu4"]
                    if args.output_dir and utils.is_main_process():
                        if getattr(args, 'zero_stage', 2) == 3 and hasattr(model, 'save_checkpoint'):
                            model.save_checkpoint(str(output_dir), tag='best_checkpoint')
                        else:
                            checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                            for checkpoint_path in checkpoint_paths:
                                utils.save_on_master({
                                    'model': get_requires_grad_dict(model_without_ddp),
                                }, checkpoint_path)

                print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {dev_stats['bleu4']:.2f}")
                print(f'Max BLEU-4: {max_accuracy:.2f}%')
            
            elif args.task == "ISLR":
                if max_accuracy < test_stats["top1_acc_pi"]:
                    max_accuracy = test_stats["top1_acc_pi"]
                    if args.output_dir and utils.is_main_process():
                        if getattr(args, 'zero_stage', 2) == 3 and hasattr(model, 'save_checkpoint'):
                            model.save_checkpoint(str(output_dir), tag='best_checkpoint')
                        else:
                            checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                            for checkpoint_path in checkpoint_paths:
                                utils.save_on_master({
                                    'model': get_requires_grad_dict(model_without_ddp),
                                }, checkpoint_path)

                print(f"PI accuracy of the network on the {len(dev_dataloader)} dev videos: {test_stats['top1_acc_pi']:.2f}")
                print(f'Max PI accuracy: {max_accuracy:.2f}%')
            
            elif args.task == "CSLR":
                if max_accuracy > test_stats["wer"]:
                    max_accuracy = test_stats["wer"]
                    if args.output_dir and utils.is_main_process():
                        if getattr(args, 'zero_stage', 2) == 3 and hasattr(model, 'save_checkpoint'):
                            model.save_checkpoint(str(output_dir), tag='best_checkpoint')
                        else:
                            checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                            for checkpoint_path in checkpoint_paths:
                                utils.save_on_master({
                                    'model': get_requires_grad_dict(model_without_ddp),
                                }, checkpoint_path)
                            
                print(f"WER of the network on the {len(dev_dataloader)} dev videos: {test_stats['wer']:.2f}")
                print(f'Min WER: {max_accuracy:.2f}%')
        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'dev_{k}': v for k, v in dev_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    target_dtype = None
    if hasattr(model, "bfloat16_enabled") and model.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif hasattr(model, "fp16_enabled") and model.fp16_enabled():
        target_dtype = torch.float16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Move model inputs to device.
        for key in list(src_input.keys()):
            if isinstance(src_input[key], torch.Tensor):
                t = src_input[key]
                if t.is_floating_point() and t.dtype == torch.float64:
                    t = t.float()
                if target_dtype is not None:
                    t = t.to(target_dtype)
                src_input[key] = t.to(device)

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']
        stack_out = model(src_input, tgt_input)
        
        total_loss = stack_out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if args.max_train_steps and (step + 1) >= args.max_train_steps:
            print(f"Reached max_train_steps={args.max_train_steps}, stop epoch early")
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = None
    if hasattr(model, "bfloat16_enabled") and model.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif hasattr(model, "fp16_enabled") and model.fp16_enabled():
        target_dtype = torch.float16
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
        tgt_name = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            # Move model inputs to device.
            for key in list(src_input.keys()):
                if isinstance(src_input[key], torch.Tensor):
                    t = src_input[key]
                    if t.is_floating_point() and t.dtype == torch.float64:
                        t = t.float()
                    if target_dtype is not None:
                        t = t.to(target_dtype)
                    src_input[key] = t.to(device)
            
            if args.task == "CSLR":
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            stack_out = model(src_input, tgt_input)
            
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())
        
            # Make generation configurable for better BLEU and easier debugging.
            gen_max_new_tokens = getattr(args, "gen_max_new_tokens", 100)
            gen_num_beams = getattr(args, "gen_num_beams", 4)
            gen_kwargs = {
                "length_penalty": getattr(args, "gen_length_penalty", 1.0),
                "no_repeat_ngram_size": getattr(args, "gen_no_repeat_ngram_size", 0),
                "repetition_penalty": getattr(args, "gen_repetition_penalty", 1.0),
            }
            # Avoid passing disabled options.
            if not gen_kwargs.get("no_repeat_ngram_size"):
                gen_kwargs.pop("no_repeat_ngram_size", None)

            output = model_without_ddp.generate(
                stack_out,
                max_new_tokens=gen_max_new_tokens,
                num_beams=gen_num_beams,
                **gen_kwargs,
            )

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])
                tgt_name.append(src_input['name_batch'][i])

            if args.max_eval_samples and len(tgt_refs) >= args.max_eval_samples:
                break

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id
    
    # `pad_sequence` supports variable-length 1D tensors directly.
    # The old implementation padded the first sample to a fixed length (150),
    # which can crash when sequences are longer than 150.
    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    if args.task == "SLT":
        # Chinese SLT datasets (CE-CSL/CSL_Daily/CSL_News):
        # Use character-level tokenization to make BLEU/ROUGE meaningful for Chinese.
        if args.dataset in {'CSL_Daily', 'CE-CSL', 'CSL_News'}:
            tgt_pres_tok = [' '.join(list(utils.normalize_zh_text(r))) for r in tgt_pres]
            tgt_refs_tok = [' '.join(list(utils.normalize_zh_text(r))) for r in tgt_refs]
            bleu_dict, rouge_score = translation_performance(tgt_refs_tok, tgt_pres_tok, tokenizer_args='none')
        else:
            bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres, tokenizer_args='13a')
        for k,v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)
        if args.eval and (args.dataset == 'How2Sign' or args.dataset == 'OpenASL'):
            # BLEURT # follow GloFE
            # Due to the long processing time, only --eval will be executed.
            from bleurt import score
            checkpoint = "./BLEURT-20"
            scorer = score.BleurtScorer(checkpoint)
            scores_bleurt = scorer.score(references=tgt_refs[:], candidates=tgt_pres[:])
            # assert isinstance(scores, list) and len(scores) == 1
            print('BLEURT:', sum(scores_bleurt)/len(scores_bleurt))

    elif args.task == "ISLR":
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
        
    elif args.task == "CSLR":
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        print(wer_results)
        for k,v in wer_results.items():
            metric_logger.meters[k].update(v)

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir+f'/{phase}_tmp_pres.txt','w') as f:
            for i in range(len(tgt_pres)):
                f.write(f"sample: {tgt_name[i]}, prediction: " + tgt_pres[i]+'\n')
        with open(args.output_dir+f'/{phase}_tmp_refs.txt','w') as f:
            for i in range(len(tgt_refs)):
                f.write(f"sample: {tgt_name[i]}, ground-truth: " + tgt_refs[i]+'\n')
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
