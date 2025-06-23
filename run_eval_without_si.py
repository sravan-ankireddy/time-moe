#!/usr/bin/env python
# -*- coding:utf-8 _*-
import json
import os
import argparse
import numpy as np
import logging
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM

from time_moe.datasets.benchmark_dataset import BenchmarkEvalDataset, GeneralEvalDataset


def setup_nccl(rank, world_size, master_addr='127.0.0.1', master_port=9899):
    dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
                            world_size=world_size)


def count_num_tensor_elements(tensor):
    n = 1
    for s in tensor.shape:
        n = n * s
    return n


# ------------------ Metrics ------------------
class SumEvalMetric:
    def __init__(self, name, init_val: float = 0.0):
        self.name = name
        self.value = init_val

    def push(self, preds, labels, **kwargs):
        self.value += self._calculate(preds, labels, **kwargs)

    def _calculate(self, preds, labels, **kwargs):
        pass


class MSEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum((preds - labels) ** 2)


class MAEMetric(SumEvalMetric):
    def _calculate(self, preds, labels, **kwargs):
        return torch.sum(torch.abs(preds - labels))


class TimeMoE:
    def __init__(self, model_path, device, context_length, prediction_length, downsample_rate=1, **kwargs):
        self.downsample_rate = downsample_rate
        self.original_prediction_length = prediction_length
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        try:
            from time_moe.models.modeling_time_moe import TimeMoeForPrediction
            model = TimeMoeForPrediction.from_pretrained(
                model_path,
                device_map=device,
                attn_implementation='flash_attention_2',
                torch_dtype='auto',
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                attn_implementation='flash_attention_2',
                torch_dtype='auto',
                trust_remote_code=True,
            )

        logging.info(f'>>> Model dtype: {model.dtype}; Attention:{model.config._attn_implementation}')

        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, batch):
        model = self.model
        device = self.device
        prediction_length = self.prediction_length
        downsample_rate = self.downsample_rate

        inputs = batch['inputs'].to(device).to(model.dtype)
        labels = batch['labels'].to(device)
        
        # Apply downsampling
        if downsample_rate > 1:
            inputs = inputs[:, ::downsample_rate]
            labels = labels[:, ::downsample_rate]

        outputs = model.generate(
            inputs=inputs,
            max_new_tokens=prediction_length,
        )
        preds = outputs[:, -prediction_length:]

        if len(preds.shape) > len(labels.shape):
            labels = labels[..., None]
        return preds, labels


def evaluate(args):
    batch_size = args.batch_size
    context_length = args.context_length
    prediction_length = args.prediction_length
    downsample_rates = args.downsample_rates

    master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', 9899)
    world_size = int(os.getenv('WORLD_SIZE') or 1)
    rank = int(os.getenv('RANK') or 0)
    local_rank = int(os.getenv('LOCAL_RANK') or 0)
    if torch.cuda.is_available():
        world_size = int(os.getenv("WORLD_SIZE") or 1)
        if world_size > 1:
            # only initialize NCCL if we're truly doing >1 process
            setup_nccl(rank, world_size, master_addr, master_port)
            device = f"cuda:{local_rank}"
            is_dist = True
        else:
            # single-GPU / CPU fallback
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
            is_dist = False
    else:
        device = 'cpu'
        is_dist = False

    # Store results for all downsampling rates
    all_results = {}
    
    for downsample_rate in downsample_rates:
        print(f"Evaluating with downsample_rate: {downsample_rate}")
        
        # Apply downsampling to context and prediction lengths
        downsampled_context_length = context_length // downsample_rate
        downsampled_prediction_length = prediction_length // downsample_rate
        
        # evaluation
        metric_list = [
            MSEMetric(name='mse'),
            MAEMetric(name='mae'),
        ]

        model = TimeMoE(
            args.model,
            device,
            context_length=downsampled_context_length,
            prediction_length=downsampled_prediction_length,
            downsample_rate=downsample_rate
        )
        
        if args.data.endswith('.csv'):
            dataset = BenchmarkEvalDataset(
                args.data,
                context_length=context_length,
                prediction_length=prediction_length,
            )
        else:
            dataset = GeneralEvalDataset(
                args.data,
                context_length=context_length,
                prediction_length=prediction_length,
            )

        if torch.cuda.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset=dataset, shuffle=False)
        else:
            sampler = None
        test_dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=2,
            prefetch_factor=2,
            drop_last=False,
        )

        acc_count = 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(test_dl, desc=f"Evaluating downsample_rate={downsample_rate}")):
                preds, labels = model.predict(batch)

                for metric in metric_list:
                    metric.push(preds, labels)

                acc_count += count_num_tensor_elements(preds)

                if args.max_batches is not None and idx >= args.max_batches - 1:
                    break

        ret_metric = {}
        for metric in metric_list:
            ret_metric[metric.name] = metric.value / acc_count
        print(f'{rank} - {ret_metric}')

        metric_tensors = [metric.value for metric in metric_list] + [acc_count]
        if is_dist:
            stat_tensor = torch.tensor(metric_tensors).to(model.device)
            gathered_results = [torch.zeros_like(stat_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_results, stat_tensor)
            all_stat = torch.stack(gathered_results, dim=0).sum(dim=0)
        else:
            all_stat = metric_tensors

        if rank == 0:
            item = {
                'model': args.model,
                'data': args.data,
                'context_length': args.context_length,
                'prediction_length': args.prediction_length,
                'downsample_rate': downsample_rate,
            }

            count = all_stat[-1]
            for i, metric in enumerate(metric_list):
                val = all_stat[i] / count
                item[metric.name] = float(val.cpu().numpy())
            logging.info(item)
            
            # Store results for plotting
            all_results[downsample_rate] = {
                'mse': item['mse'],
                'mae': item['mae']
            }
    
    # Generate plot if we have multiple rates and we're on rank 0
    if rank == 0 and len(downsample_rates) > 1:
        plot_results(all_results, args)


def plot_results(results, args):
    """Generate bar plot comparing MSE and MAE across downsampling rates."""
    rates = sorted(results.keys())
    mse_values = [results[rate]['mse'] for rate in rates]
    mae_values = [results[rate]['mae'] for rate in rates]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE bar plot
    bars1 = ax1.bar([str(rate) for rate in rates], mse_values, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('MSE vs Downsampling Rate')
    ax1.set_xlabel('Downsampling Rate')
    ax1.set_ylabel('MSE')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.4f}', ha='center', va='bottom')
    
    # MAE bar plot
    bars2 = ax2.bar([str(rate) for rate in rates], mae_values, color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_title('MAE vs Downsampling Rate')
    ax2.set_xlabel('Downsampling Rate')
    ax2.set_ylabel('MAE')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create structured folder and save plot
    dataset_name = os.path.basename(args.data).split('.')[0]
    results_dir = f"results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    plot_filename = f"{results_dir}/p{args.prediction_length}_c{args.context_length}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TimeMoE Evaluate')
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='/home/sa53869/time-series/model_weights/time-moe-200m',
        help='Model path'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Benchmark data path'
    )

    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=32,
        help='Batch size of evaluation'
    )
    parser.add_argument(
        '--context_length', '-c',
        type=int,
        help='Context length'
    )
    parser.add_argument(
        '--prediction_length', '-p',
        type=int,
        default=96,
        help='Prediction length'
    )
    parser.add_argument(
        '--downsample_rates',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8],
        help='List of downsampling rates to evaluate (e.g., 1 2 4 8)'
    )
    parser.add_argument(
        '--max_batches',
        type=int,
        default=None,
        help='Maximum number of batches to evaluate (default: all batches)'
    )
    args = parser.parse_args()
    if args.context_length is None:
        if args.prediction_length == 96:
            args.context_length = 512
        elif args.prediction_length == 192:
            args.context_length = 1024
        elif args.prediction_length == 336:
            args.context_length = 2048
        elif args.prediction_length == 720:
            args.context_length = 3072
        else:
            args.context_length = args.prediction_length * 4
    evaluate(args)
