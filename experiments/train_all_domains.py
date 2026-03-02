"""
Master training script to train all domain-specific GNNs.

Trains Math, Code, and Medical GNNs sequentially with 1000 samples each.
"""

import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_training(script_name, domain_name):
    """Run a training script and capture results."""
    print("\n" + "="*80)
    print(f"STARTING {domain_name.upper()} TRAINING")
    print("="*80 + "\n")
    
    script_path = Path(__file__).parent / script_name
    venv_python = Path(__file__).parent.parent / "venv" / "bin" / "python"
    
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            [str(venv_python), str(script_path)],
            capture_output=False,
            text=True,
            check=True,
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n✅ {domain_name} training completed in {duration:.1f}s")
        return {
            "domain": domain_name,
            "success": True,
            "duration_seconds": duration,
            "error": None,
        }
        
    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n❌ {domain_name} training failed after {duration:.1f}s")
        print(f"Error: {e}")
        return {
            "domain": domain_name,
            "success": False,
            "duration_seconds": duration,
            "error": str(e),
        }


def main():
    print("="*80)
    print("MULTI-DOMAIN GNN TRAINING PIPELINE")
    print("="*80)
    print("\nThis script will train 3 domain-specific GNNs:")
    print("  1. Math GNN (PRM800K) - 1000 samples")
    print("  2. Code GNN (HumanEval) - 1000 samples")
    print("  3. Medical GNN (MedHallu) - 1000 samples")
    print("\nEach training will take approximately 5-10 minutes.")
    print("\n" + "="*80)
    
    overall_start = datetime.now()
    results = []
    
    # Train each domain
    training_tasks = [
        ("quick_test_train.py", "Math"),
        ("train_code_test.py", "Code"),
        ("train_medical_test.py", "Medical"),
    ]
    
    for script, domain in training_tasks:
        result = run_training(script, domain)
        results.append(result)
        
        # Short break between trainings
        import time
        time.sleep(2)
    
    overall_end = datetime.now()
    total_duration = (overall_end - overall_start).total_seconds()
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING PIPELINE SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\nTotal Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    
    print("\nDomain Results:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        domain = result["domain"]
        duration = result["duration_seconds"]
        print(f"  {status} {domain:10s} - {duration:.1f}s")
        if result["error"]:
            print(f"     Error: {result['error']}")
    
    # Load and compare metrics from each domain
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    project_root = Path(__file__).parent.parent
    checkpoint_dirs = [
        ("Math", project_root / "models/checkpoints/test_run"),
        ("Code", project_root / "models/checkpoints/code_test_run"),
        ("Medical", project_root / "models/checkpoints/medical_test_run"),
    ]
    
    print(f"\n{'Domain':<12} {'Accuracy':<12} {'Origin Acc':<12} {'Test Loss':<12}")
    print("-" * 50)
    
    for domain, checkpoint_dir in checkpoint_dirs:
        result_file = checkpoint_dir / f"{domain.lower()}_training_results.json"
        if domain == "Math":
            result_file = checkpoint_dir / "test_training_results.json"
        
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
                metrics = data.get("test_metrics", {})
                acc = metrics.get("accuracy", 0) * 100
                origin_acc = metrics.get("origin_accuracy", 0) * 100
                loss = metrics.get("loss", 0)
                print(f"{domain:<12} {acc:>10.2f}%  {origin_acc:>10.2f}%  {loss:>10.4f}")
        else:
            print(f"{domain:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print("\n" + "="*80)
    print("TARGET METRICS (for reference)")
    print("="*80)
    print("Math:    Origin Detection >85%, Node F1 >80%")
    print("Code:    Origin Detection >75%, Node F1 >70%")
    print("Medical: Origin Detection >70%, Node F1 >65%")
    
    if len(successful) == len(results):
        print("\n" + "="*80)
        print("🎉 ALL DOMAIN TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)
        print("\nNext Steps:")
        print("  1. Review individual training results in models/checkpoints/")
        print("  2. Run cross-domain evaluation")
        print("  3. Scale up to full datasets if results look good")
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TRAINING RUNS FAILED")
        print("="*80)
        print("\nCheck error messages above for details.")
    
    # Save summary
    summary_path = project_root / "experiments" / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": overall_start.isoformat(),
            "total_duration_seconds": total_duration,
            "results": results,
        }, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

