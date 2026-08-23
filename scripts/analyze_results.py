#!/usr/bin/env python3
"""Parse training logs and generate experiment summary tables."""
import os
import re
import json
import sys

LOG_DIR = "/data/lab/IJCV_KD/CIARD_Expansion_mobilenetv2_cifar10_v1/logs"
RESULTS_DIR = "/data/lab/IJCV_KD/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_log(log_file):
    """Parse a CIARD training log and extract evaluation metrics per epoch."""
    results = []
    current_epoch = 0
    
    with open(log_file, 'r', errors='replace') as f:
        for line in f:
            # Extract epoch number
            m = re.search(r'the (\d+)th epoch', line)
            if m:
                current_epoch = int(m.group(1))
            
            # Extract robust accuracy
            m = re.search(r'student robust acc ([\d.]+)', line)
            if m:
                robust_acc = float(m.group(1))
            
            # Extract natural (clean) accuracy
            m = re.search(r'student natural acc ([\d.]+)', line)
            if m:
                nat_acc = float(m.group(1))
                results.append({
                    'epoch': current_epoch,
                    'clean_acc': nat_acc,
                    'robust_acc': robust_acc,
                    'best_acc': (nat_acc + robust_acc) / 2,
                })
            
            # Extract SARD metrics
            m = re.search(r'sard_eps:([\d.]+)', line)
            if m and results:
                results[-1]['sard_eps'] = float(m.group(1))
            
            m = re.search(r'sard_trs_mean:([\d.]+)', line)
            if m and results:
                results[-1]['sard_trs'] = float(m.group(1))
    
    return results

def generate_summary(experiments):
    """Generate a summary table from all experiments."""
    print("=" * 80)
    print("EXPERIMENT SUMMARY: SARD Ablation Study (60 epochs)")
    print("=" * 80)
    
    for name, results in experiments.items():
        if not results:
            print(f"\n{name}: No results")
            continue
        
        last = results[-1]
        best = max(results, key=lambda x: x['best_acc'])
        
        print(f"\n--- {name} ---")
        print(f"  Total epochs evaluated: {len(results)}")
        print(f"  Final (epoch {last['epoch']}): clean={last['clean_acc']:.4f}, robust={last['robust_acc']:.4f}, best={last['best_acc']:.4f}")
        print(f"  Best (epoch {best['epoch']}): clean={best['clean_acc']:.4f}, robust={best['robust_acc']:.4f}, best={best['best_acc']:.4f}")
        
        if 'sard_eps' in last:
            print(f"  SARD: eps={last.get('sard_eps', 'N/A'):.4f}, trs={last.get('sard_trs', 'N/A'):.4f}")
    
    # Comparison table
    if len(experiments) >= 2:
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Config':<25} {'Clean Acc':>10} {'Robust Acc':>10} {'Best Acc':>10}")
        print("-" * 55)
        
        for name, results in experiments.items():
            if results:
                last = results[-1]
                print(f"{name:<25} {last['clean_acc']:>10.4f} {last['robust_acc']:>10.4f} {last['best_acc']:>10.4f}")
    
    # Save JSON
    output = {name: results for name, results in experiments.items()}
    output_file = os.path.join(RESULTS_DIR, "ablation_summary.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    experiments = {}
    
    # Parse all ablation logs
    log_files = {
        "baseline": "ablation_baseline.log",
        "sard_full": "ablation_sard_full.log",
        "saa_only": "ablation_saa_only.log",
        "rcd_only": "ablation_rcd_only.log",
    }
    
    for name, filename in log_files.items():
        filepath = os.path.join(LOG_DIR, filename)
        if os.path.exists(filepath):
            experiments[name] = parse_log(filepath)
    
    if experiments:
        generate_summary(experiments)
    else:
        print("No experiment logs found!")
