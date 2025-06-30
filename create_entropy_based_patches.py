#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import os
import glob
try:
    from datasets import load_from_disk
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. HuggingFace dataset support disabled.")

def compute_entropy_and_scaled(x, B, pooled_scaled_data=None, context_length=None):
    """Compute scaled signal and per-sample entropy."""
    s = np.mean(np.abs(x))
    if s == 0:
        raise ValueError("All-zero signal")
    scaled = x / s
    
    if context_length is not None:
        # Contextual entropy
        per_sample_entropy = np.zeros(len(scaled))
        for i in range(len(scaled)):
            start = max(0, i - context_length // 2)
            end = min(len(scaled), start + context_length)
            if end - start < context_length and start > 0:
                start = max(0, end - context_length)
            
            context = scaled[start:end]
            c_min, c_max = context.min(), context.max()
            if c_min == c_max:
                per_sample_entropy[i] = 0.0
                continue
                
            edges = np.linspace(c_min, c_max, B + 1)[1:-1]
            sample_bin = np.clip(np.digitize([scaled[i]], edges)[0], 0, B-1)
            context_bins = np.clip(np.digitize(context, edges), 0, B-1)
            counts = np.bincount(context_bins, minlength=B)
            probs = counts / len(context)
            per_sample_entropy[i] = -np.log2(probs[sample_bin] + np.finfo(float).eps)
        return scaled, per_sample_entropy
    
    # Local/global entropy
    prob_data = pooled_scaled_data if pooled_scaled_data is not None else scaled
    c_min, c_max = prob_data.min(), prob_data.max()
    edges = np.linspace(c_min, c_max, B + 1)[1:-1]
    
    bins = np.clip(np.digitize(scaled, edges), 0, B-1)
    prob_bins = np.clip(np.digitize(prob_data, edges), 0, B-1)
    counts = np.bincount(prob_bins, minlength=B)
    probs = counts / len(prob_data)
    per_sample_entropy = -np.log2(probs[bins] + np.finfo(float).eps)
    
    return scaled, per_sample_entropy

def load_huggingface_dataset(dataset_path: str, max_samples_per_split: int = None):
    """Load numeric data from multiple HuggingFace dataset directories."""
    if not HF_DATASETS_AVAILABLE:
        raise ImportError("datasets library is required for HuggingFace dataset support. Install with: pip install datasets")
    
    print(f"Loading HuggingFace datasets from {dataset_path}")
    
    all_scaled_data = []
    total_samples = 0
    
    # Find all subdirectories that contain dataset files
    dataset_dirs = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            # Check if this directory contains dataset files
            has_arrow = any(f.endswith('.arrow') for f in os.listdir(item_path))
            has_state = os.path.exists(os.path.join(item_path, 'state.json'))
            if has_arrow or has_state:
                dataset_dirs.append(item_path)
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    
    for dataset_dir in dataset_dirs:
        try:
            print(f"Loading dataset from {dataset_dir}")
            dataset = load_from_disk(dataset_dir)
            
            # Handle both single Dataset and DatasetDict
            if hasattr(dataset, 'keys'):
                # DatasetDict - process each split
                for split_name in dataset.keys():
                    split_data = dataset[split_name]
                    processed_samples = process_dataset_split(split_data, max_samples_per_split)
                    all_scaled_data.extend(processed_samples)
                    total_samples += sum(len(data) for data in processed_samples)
            else:
                # Single Dataset
                processed_samples = process_dataset_split(dataset, max_samples_per_split)
                all_scaled_data.extend(processed_samples)
                total_samples += sum(len(data) for data in processed_samples)
                
        except Exception as e:
            print(f"Warning: Failed to load dataset from {dataset_dir}: {e}")
            continue
    
    if not all_scaled_data:
        raise ValueError("No valid numeric data found in any HuggingFace datasets")
    
    pooled_scaled = np.concatenate(all_scaled_data)
    print(f"Total pooled scaled samples from HuggingFace datasets: {len(pooled_scaled)}")
    return pooled_scaled

def process_dataset_split(split_data, max_samples_per_split):
    """Process a single dataset split and extract numeric data."""
    processed_data = []
    
    split_limit = len(split_data)
    if max_samples_per_split is not None:
        split_limit = min(split_limit, max_samples_per_split)
    
    for i, row in enumerate(split_data.select(range(split_limit))):
        if i >= split_limit:
            break
            
        # Extract numeric values from all fields in the row
        for field_name, field_value in row.items():
            try:
                if isinstance(field_value, (list, tuple)):
                    numeric_data = np.array(field_value, dtype=float)
                    if numeric_data.size > 0 and not np.all(np.isnan(numeric_data)):
                        clean_data = numeric_data[~np.isnan(numeric_data)]
                        if len(clean_data) > 0:
                            s = np.mean(np.abs(clean_data))
                            if s > 0:
                                processed_data.append(clean_data / s)
                elif isinstance(field_value, (int, float)):
                    if not np.isnan(float(field_value)):
                        val = float(field_value)
                        if val != 0:
                            processed_data.append(np.array([val / abs(val)]))
                elif isinstance(field_value, str):
                    try:
                        val = float(field_value)
                        if not np.isnan(val) and val != 0:
                            processed_data.append(np.array([val / abs(val)]))
                    except ValueError:
                        continue
            except (ValueError, TypeError, OverflowError):
                continue
    
    return processed_data

def load_and_scale_all_data(datasets_dir: str, use_huggingface: bool = False, max_samples_per_split: int = None):
    """Load and scale all columns from all CSV files or HuggingFace dataset."""
    if use_huggingface:
        return load_huggingface_dataset(datasets_dir, max_samples_per_split)
    
    # Original CSV loading logic
    all_scaled_data = []
    csv_files = glob.glob(os.path.join(datasets_dir, "**", "*.csv"), recursive=True)
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            for col_idx in range(df.shape[1]):
                try:
                    col_data = df.iloc[:, col_idx].to_numpy(dtype=float)
                    col_data = col_data[~np.isnan(col_data)]
                    
                    if len(col_data) > 0:
                        s = np.mean(np.abs(col_data))
                        if s > 0:
                            all_scaled_data.append(col_data / s)
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_scaled_data:
        raise ValueError("No valid numeric data found")
    
    pooled_scaled = np.concatenate(all_scaled_data)
    print(f"Total pooled scaled samples: {len(pooled_scaled)}")
    return pooled_scaled

def create_variable_patches(scaled, per_sample_entropy, min_size=2, max_size=64, 
                          stability_threshold=0.1, growth_factor=1.2, spike_threshold=1.0):
    """Create variable-size patches based on entropy stability."""
    patches = []
    i = 0
    
    while i < len(scaled):
        current_size = min_size
        best_size = min_size
        end_idx = min(i + min_size, len(scaled))
        
        if end_idx <= i:
            break
            
        baseline_entropy = per_sample_entropy[i:end_idx]
        baseline_mean = np.mean(baseline_entropy)
        baseline_std = np.std(baseline_entropy)
        
        # Grow patch while stable
        while current_size < max_size:
            next_size = max(current_size + 1, int(current_size * growth_factor))
            next_size = min(next_size, max_size)
            next_end = min(i + next_size, len(scaled))
            
            if next_end - i < next_size and next_end >= len(scaled):
                best_size = next_end - i
                break
                
            extended_entropy = per_sample_entropy[i:next_end]
            extended_mean = np.mean(extended_entropy)
            extended_std = np.std(extended_entropy)
            
            mean_change = abs(extended_mean - baseline_mean) / (baseline_mean + 1e-8)
            std_increase = extended_std / (baseline_std + 1e-8)
            
            # Check for spikes in new region
            new_region = per_sample_entropy[end_idx:next_end]
            spike_absolute = np.max(np.abs(new_region - baseline_mean)) if len(new_region) > 0 else 0
            
            is_stable = (mean_change < stability_threshold and 
                        std_increase < 2.0 and 
                        spike_absolute < spike_threshold)
            
            if is_stable:
                best_size = next_size
                current_size = next_size
                end_idx = next_end
                # Update baseline with exponential moving average
                alpha = 0.7
                baseline_mean = alpha * baseline_mean + (1 - alpha) * extended_mean
                baseline_std = alpha * baseline_std + (1 - alpha) * extended_std
            else:
                break
                
            if next_size >= max_size:
                break
        
        final_end = min(i + best_size, len(scaled))
        actual_size = final_end - i
        
        if actual_size <= 0:
            break
            
        patch_entropy = per_sample_entropy[i:final_end]
        patches.append({
            'start': i,
            'end': final_end,
            'size': actual_size,
            'scaled': scaled[i:final_end],
            'entropy': patch_entropy,
            'mean_entropy': np.mean(patch_entropy),
            'entropy_std': np.std(patch_entropy),
            'entropy_range': np.ptp(patch_entropy)
        })
        
        i = final_end
    
    return patches

def print_patch_stats(patches):
    """Print patch statistics."""
    sizes = [p['size'] for p in patches]
    entropies = [p['mean_entropy'] for p in patches]
    
    print(f"\n=== Patch Statistics ===")
    print(f"Total patches: {len(patches)}")
    print(f"Size - mean: {np.mean(sizes):.2f}, median: {np.median(sizes):.1f}, range: {np.min(sizes)}-{np.max(sizes)}")
    print(f"Entropy - mean: {np.mean(entropies):.4f}, range: {np.min(entropies):.4f}-{np.max(entropies):.4f}")
    
    # Size distribution
    unique_sizes = sorted(set(sizes))[:10]
    print(f"Top sizes: {', '.join(f'{s}({sum(1 for x in sizes if x == s)})' for s in unique_sizes)}")

def plot_signal_overview(scaled, per_sample_entropy, patches, portion_length, title_suffix, output_path):
    """Plot 3 random signal portions with patch boundaries as markers."""
    if len(scaled) < portion_length:
        raise ValueError(f"Signal too short ({len(scaled)}) for portion length {portion_length}")
    
    np.random.seed(42)
    max_start = len(scaled) - portion_length
    starts = np.sort(np.random.choice(max_start + 1, size=3, replace=False))
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Calculate global y-limits for all portions
    all_scaled_portions = []
    all_entropy_portions = []
    for start in starts:
        end = start + portion_length
        all_scaled_portions.append(scaled[start:end])
        all_entropy_portions.append(per_sample_entropy[start:end])
    
    scaled_ylim = (np.min(np.concatenate(all_scaled_portions)), 
                   np.max(np.concatenate(all_scaled_portions)))
    entropy_ylim = (np.min(np.concatenate(all_entropy_portions)), 
                    np.max(np.concatenate(all_entropy_portions)))
    
    for i, start in enumerate(starts):
        end = start + portion_length
        indices = np.arange(start, end)
        
        ax1 = axes[i]
        # Plot data with markers and dash-dot line style
        ax1.plot(indices, scaled[start:end], '-.o', color='C0', alpha=0.8, markersize=4)
        ax1.set_ylabel("Scaled value", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax1.set_ylim(scaled_ylim)
        
        # Add patch boundaries as dotted vertical lines
        relevant_patches = [p for p in patches if p['start'] < end and p['end'] > start]
        for patch in relevant_patches:
            patch_start = max(patch['start'], start)
            patch_end = min(patch['end'], end)
            ax1.axvline(patch_start, color='k', linestyle=':', alpha=0.7)
            if patch_end < end:
                ax1.axvline(patch_end, color='k', linestyle=':', alpha=0.7)
        
        ax2 = ax1.twinx()
        ax2.plot(indices, per_sample_entropy[start:end], 'C1', alpha=0.8)
        ax2.set_ylabel("Entropy (bits)", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax2.set_ylim(entropy_ylim)
        
        ax1.set_title(f"Signal portion {i+1}: samples {start}-{end-1}")
        
        if i == 2:
            ax1.set_xlabel("Sample index")
    
    plt.suptitle(f"Signal Overview with Variable-Size Patches {title_suffix}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Overview plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Entropy analysis with variable-size patches")
    parser.add_argument("--csv_path", default="./datasets/ETT-small/ETTm2.csv", help="CSV file path")
    parser.add_argument("--column", type=int, default=1, help="Column index (0-based)")
    parser.add_argument("--bins", type=int, default=4096, help="Quantization bins")
    parser.add_argument("--use_global", action="store_true", help="Use global vocabulary")
    parser.add_argument("--use_contextual", action="store_true", help="Use contextual vocabulary")
    parser.add_argument("--context_length", type=int, default=1024, help="Context window length")
    
    # Global vocabulary options
    parser.add_argument("--global_data_path", default="./datasets/ETT-small", 
                       help="Path to directory containing CSV files or HuggingFace dataset")
    parser.add_argument("--use_huggingface", action="store_true", 
                       help="Use HuggingFace dataset format instead of CSV files for global vocabulary")
    parser.add_argument("--max_samples_per_split", type=int, default=None,
                       help="Maximum samples to load per dataset split (for large HuggingFace datasets)")
    
    # Patch parameters
    parser.add_argument("--min_patch_size", type=int, default=2, help="Minimum patch size")
    parser.add_argument("--max_patch_size", type=int, default=64, help="Maximum patch size")
    parser.add_argument("--entropy_stability_threshold", type=float, default=0.3, help="Stability threshold")
    parser.add_argument("--growth_factor", type=float, default=1.2, help="Patch growth factor")
    parser.add_argument("--spike_absolute_threshold", type=float, default=1.0, help="Spike threshold")
    parser.add_argument("--plot_portion_length", type=int, default=256, help="Plot portion length")
    args = parser.parse_args()

    # Load data
    print(f"Loading {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    if args.column >= df.shape[1]:
        raise IndexError(f"Column {args.column} not found (file has {df.shape[1]} columns)")
    
    x = df.iloc[:, args.column].to_numpy(dtype=float)
    print(f"Loaded {len(x)} samples")

    # Determine vocabulary mode
    if args.use_global and args.use_contextual:
        raise ValueError("Cannot use both global and contextual modes")
    
    if args.use_global:
        if args.use_huggingface:
            print(f"Using global vocabulary from HuggingFace dataset at {args.global_data_path}...")
            if args.max_samples_per_split:
                print(f"Limiting to {args.max_samples_per_split} samples per split")
        else:
            print(f"Using global vocabulary from CSV files in {args.global_data_path}...")
        
        pooled_data = load_and_scale_all_data(
            args.global_data_path, 
            use_huggingface=args.use_huggingface,
            max_samples_per_split=args.max_samples_per_split
        )
        scaled, entropy = compute_entropy_and_scaled(x, args.bins, pooled_scaled_data=pooled_data)
        vocab_type = "global_hf" if args.use_huggingface else "global"
    elif args.use_contextual:
        print(f"Using contextual vocabulary (length {args.context_length})...")
        scaled, entropy = compute_entropy_and_scaled(x, args.bins, context_length=args.context_length)
        vocab_type = f"contextual_ctx{args.context_length}"
    else:
        print("Using local vocabulary...")
        scaled, entropy = compute_entropy_and_scaled(x, args.bins)
        vocab_type = "local"

    # Create patches
    patches = create_variable_patches(
        scaled, entropy, 
        min_size=args.min_patch_size,
        max_size=args.max_patch_size,
        stability_threshold=args.entropy_stability_threshold,
        growth_factor=args.growth_factor,
        spike_threshold=args.spike_absolute_threshold
    )
    
    print_patch_stats(patches)

    # Setup output
    dataset_name = os.path.splitext(os.path.basename(args.csv_path))[0]
    output_dir = f"entropy_patches/{dataset_name}/col_{args.column}"
    os.makedirs(output_dir, exist_ok=True)
    
    title_suffix = f"({vocab_type}) - col {args.column}, B={args.bins}"
    filename_suffix = f"{vocab_type}_col{args.column}_bins{args.bins}_var"
    
    # Generate only overview plot with patch markers
    overview_plot_path = os.path.join(output_dir, f"sample_{filename_suffix}.png")
    plot_signal_overview(scaled, entropy, patches, args.plot_portion_length, title_suffix, overview_plot_path)
    
    print(f"Mean entropy: {np.mean(entropy):.4f} bits")

if __name__ == "__main__":
    main()