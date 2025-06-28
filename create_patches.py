#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import tempfile
import os
import glob

def load_and_scale_all_data(datasets_dir: str):
    """Load and individually scale ALL columns from all CSV files for vocabulary creation."""
    all_scaled_data = []
    csv_files = glob.glob(os.path.join(datasets_dir, "**", "*.csv"), recursive=True)
    
    print(f"Found {len(csv_files)} CSV files in {datasets_dir}")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            for col_idx in range(df.shape[1]):
                try:
                    col_data = df.iloc[:, col_idx].to_numpy(dtype=float)
                    col_data = col_data[~np.isnan(col_data)]  # remove NaN
                    
                    if len(col_data) > 0:
                        # Scale each time series individually (mean-absolute scaling)
                        s = np.mean(np.abs(col_data))
                        if s > 0:
                            scaled_col = col_data / s
                            all_scaled_data.append(scaled_col)
                            print(f"  Scaled {len(col_data)} samples from {csv_file} col {col_idx}")
                
                except (ValueError, TypeError):
                    continue  # Skip non-numeric columns
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
    
    if not all_scaled_data:
        raise ValueError("No valid numeric data found")
    
    # Pool all individually-scaled time series for vocabulary
    pooled_scaled = np.concatenate(all_scaled_data)
    print(f"Total pooled scaled samples: {len(pooled_scaled)}")
    return pooled_scaled

def compute_scaled_and_entropy(x: np.ndarray, B: int, pooled_scaled_data: np.ndarray = None):
    """Compute scaled signal and per-sample entropy using global vocabulary or local statistics."""
    
    # Scale the input signal
    s = np.mean(np.abs(x))
    if s == 0:
        raise ValueError("All-zero signal")
    scaled = x / s

    # Determine quantization bounds
    if pooled_scaled_data is not None:
        # Use full range from pooled scaled data
        c_min, c_max = pooled_scaled_data.min(), pooled_scaled_data.max()
        print(f"Global vocab range: [{c_min:.4f}, {c_max:.4f}]")
        prob_data = pooled_scaled_data
    else:
        # Fallback to local statistics
        c_min, c_max = scaled.min(), scaled.max()
        prob_data = scaled

    # Create uniform quantization bins
    edges = np.linspace(c_min, c_max, B + 1)[1:-1]  # B-1 edges for B bins
    bins = np.digitize(scaled, edges, right=False)
    bins = np.clip(bins, 0, B-1)  # Ensure valid range

    # Compute probabilities from appropriate data
    prob_bins = np.digitize(prob_data, edges, right=False)
    prob_bins = np.clip(prob_bins, 0, B-1)
    counts = np.bincount(prob_bins, minlength=B)
    probs = counts / len(prob_data)
    
    # Per-sample entropy (self-information in bits)
    eps = np.finfo(float).eps
    per_sample_entropy = -np.log2(probs[bins] + eps)
    
    # Overall entropy of the distribution
    overall_entropy = -np.sum(probs * np.log2(probs + eps))
    print(f"Overall entropy: {overall_entropy:.4f} bits")

    return scaled, per_sample_entropy

def create_adaptive_patches(scaled: np.ndarray, per_sample_entropy: np.ndarray, patch_sizes=[4, 8, 16, 32]):
    """Create adaptive patches based on entropy levels."""
    
    # Define entropy thresholds for patch size assignment
    entropy_percentiles = np.percentile(per_sample_entropy, [25, 50, 75])
    
    patches = []
    patch_info = []
    i = 0
    
    while i < len(scaled):
        # Get current entropy level
        current_entropy = per_sample_entropy[i]
        
        # Assign patch size based on entropy level (higher entropy = smaller patches)
        if current_entropy >= entropy_percentiles[2]:  # Top 25% entropy
            patch_size = patch_sizes[0]  # Smallest patch (4)
        elif current_entropy >= entropy_percentiles[1]:  # 50-75% entropy
            patch_size = patch_sizes[1]  # Small patch (8)
        elif current_entropy >= entropy_percentiles[0]:  # 25-50% entropy
            patch_size = patch_sizes[2]  # Medium patch (16)
        else:  # Bottom 25% entropy
            patch_size = patch_sizes[3]  # Largest patch (32)
        
        # Ensure we don't exceed signal length
        end_idx = min(i + patch_size, len(scaled))
        actual_patch_size = end_idx - i
        
        # Create patch
        patch_scaled = scaled[i:end_idx]
        patch_entropy = per_sample_entropy[i:end_idx]
        
        patches.append({
            'start': i,
            'end': end_idx,
            'size': actual_patch_size,
            'scaled': patch_scaled,
            'entropy': patch_entropy,
            'mean_entropy': np.mean(patch_entropy)
        })
        
        patch_info.append({
            'start': i,
            'size': actual_patch_size,
            'mean_entropy': np.mean(patch_entropy)
        })
        
        i = end_idx
    
    return patches, patch_info

def print_patch_statistics(patch_info):
    """Print statistics about the patches."""
    sizes = [p['size'] for p in patch_info]
    entropies = [p['mean_entropy'] for p in patch_info]
    
    print(f"\n=== Patch Statistics ===")
    print(f"Total patches: {len(patch_info)}")
    print(f"Average patch size: {np.mean(sizes):.2f}")
    print(f"Patch size distribution:")
    for size in [4, 8, 16, 32]:
        count = sum(1 for s in sizes if s == size)
        percentage = (count / len(sizes)) * 100
        print(f"  Size {size:2d}: {count:4d} patches ({percentage:5.1f}%)")
    
    print(f"\nEntropy statistics:")
    print(f"  Mean entropy: {np.mean(entropies):.4f} bits")
    print(f"  Min entropy:  {np.min(entropies):.4f} bits")
    print(f"  Max entropy:  {np.max(entropies):.4f} bits")

def main():
    parser = argparse.ArgumentParser(
        description="Compute mean-scaled signal and entropy from global vocabulary."
    )
    parser.add_argument("--csv_path", type=str, default="./datasets/ETT-small/ETTm2.csv", help="path to your CSV file")
    parser.add_argument("--column",   type=int, default=1,
                        help="which column index (0-based) to load")
    parser.add_argument("--bins",   type=int, default=4096,
                        help="number of quantization bins (default: 4096)")
    parser.add_argument("--use_global", action="store_true", default=False,
                        help="use global statistics from all columns/CSVs (default: use only current column)")
    args = parser.parse_args()

    # Load specific file for analysis
    print(f"Loading file for analysis: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    if args.column < 0 or args.column >= df.shape[1]:
        raise IndexError(f"Column {args.column} not found (file has {df.shape[1]} columns)")
    
    x = df.iloc[:, args.column].to_numpy(dtype=float)
    print(f"Loaded {len(x)} samples for analysis")

    # Determine vocabulary source based on argument
    if args.use_global:
        print("Loading and scaling all datasets for global vocabulary...")
        pooled_scaled_data = load_and_scale_all_data("./datasets/ETT-small")
        vocab_type = "global"
    else:
        print("Using local column statistics for vocabulary...")
        pooled_scaled_data = None
        vocab_type = "local"

    # Compute entropy using chosen vocabulary
    scaled, per_sample_entropy = compute_scaled_and_entropy(x, args.bins, pooled_scaled_data)

    # Create adaptive patches based on entropy
    patches, patch_info = create_adaptive_patches(scaled, per_sample_entropy)
    
    # Print patch statistics
    print_patch_statistics(patch_info)

    # Select 3 interesting patches for visualization (high, medium, low entropy)
    sorted_patches = sorted(patches, key=lambda p: p['mean_entropy'])
    
    # Select patches: lowest entropy, median entropy, highest entropy
    selected_indices = [0, len(sorted_patches)//2, -1]
    selected_patches = [sorted_patches[i] for i in selected_indices]
    
    # Plot the selected patches
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for i, patch in enumerate(selected_patches):
        patch_indices = np.arange(patch['start'], patch['end'])
        
        ax1 = axes[i]
        ax1.plot(patch_indices, patch['scaled'], linewidth=1.2, color="C0", label='Scaled signal')
        ax1.set_ylabel("Scaled value", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        
        # Add patch boundary visualization
        ax1.axvspan(patch['start'], patch['end'], alpha=0.1, color='gray', label=f'Patch (size {patch["size"]})')
        
        ax2 = ax1.twinx()
        ax2.plot(patch_indices, patch['entropy'], linewidth=1.2, color="C1", label='Entropy')
        ax2.set_ylabel("Entropy (bits)", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        
        entropy_type = ['Low entropy', 'Medium entropy', 'High entropy'][i]
        ax1.set_title(f"{entropy_type} patch (size {patch['size']}, mean entropy: {patch['mean_entropy']:.3f} bits)")
        
        if i == 2:  # only add x-label to bottom subplot
            ax1.set_xlabel("Sample index")

    plt.suptitle(f"Adaptive Patches by Entropy ({vocab_type} vocab) - col {args.column}, B={args.bins}")
    fig.tight_layout()

    # Create a second figure showing 3 random signal portions with patches
    fig2, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Select 3 random portions of length 256
    portion_length = 256
    if len(scaled) < portion_length:
        raise ValueError(f"Signal too short ({len(scaled)} samples) for portion length {portion_length}")
    
    np.random.seed(42)  # for reproducibility
    max_start = len(scaled) - portion_length
    portion_starts = np.random.choice(max_start + 1, size=3, replace=False)
    portion_starts = np.sort(portion_starts)
    
    colors = {4: 'red', 8: 'orange', 16: 'yellow', 32: 'green'}
    
    for i, start_idx in enumerate(portion_starts):
        end_idx = start_idx + portion_length
        portion_indices = np.arange(start_idx, end_idx)
        portion_scaled = scaled[start_idx:end_idx]
        portion_entropy = per_sample_entropy[start_idx:end_idx]
        
        ax1 = axes[i]
        
        # Plot scaled signal
        ax1.plot(portion_indices, portion_scaled, linewidth=1.0, color="C0", alpha=0.8, label='Scaled signal')
        ax1.set_ylabel("Scaled value", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        
        # Add patch boundaries that intersect with this portion
        relevant_patches = [p for p in patches if p['start'] < end_idx and p['end'] > start_idx]
        for patch in relevant_patches:
            patch_start = max(patch['start'], start_idx)
            patch_end = min(patch['end'], end_idx)
            color = colors.get(patch['size'], 'gray')
            ax1.axvspan(patch_start, patch_end, alpha=0.3, color=color)
        
        # Plot entropy on second y-axis
        ax2 = ax1.twinx()
        ax2.plot(portion_indices, portion_entropy, linewidth=1.0, color="C1", alpha=0.8, label='Entropy')
        ax2.set_ylabel("Entropy (bits)", color="C1")
        ax2.tick_params(axis="y", labelcolor="C1")
        
        ax1.set_title(f"Signal portion {i+1}: samples {start_idx}-{end_idx-1}")
        
        if i == 2:  # only add x-label to bottom subplot
            ax1.set_xlabel("Sample index")
        
        # Add legend only to first subplot
        if i == 0:
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[size], alpha=0.3, label=f'Patch size {size}') 
                              for size in [4, 8, 16, 32]]
            ax1.legend(handles=legend_elements, loc='upper left')

    plt.suptitle(f"Signal Overview with Adaptive Patches ({vocab_type} vocab) - col {args.column}, B={args.bins}")

    plt.tight_layout()

    # save both figures to temp/ folder
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save adaptive patches plot
    plot_filename1 = f"adaptive_patches_{vocab_type}_col{args.column}_bins{args.bins}.png"
    plot_path1 = os.path.join(temp_dir, plot_filename1)
    fig.savefig(plot_path1)
    print(f"Adaptive patches plot saved to {plot_path1}")
    
    # Save full signal overview
    plot_filename2 = f"signal_overview_{vocab_type}_col{args.column}_bins{args.bins}.png"
    plot_path2 = os.path.join(temp_dir, plot_filename2)
    fig2.savefig(plot_path2)
    print(f"Signal overview plot saved to {plot_path2}")
    
    print(f"Mean per-sample entropy ({vocab_type} vocab): {np.mean(per_sample_entropy):.4f} bits")

if __name__ == "__main__":
    main()
