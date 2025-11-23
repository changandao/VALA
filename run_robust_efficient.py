#!/usr/bin/env python3

"""
Simplified script to run memory-efficient robust Weiszfeld algorithm
"""

import os
import sys
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Run memory-efficient robust Weiszfeld feature extraction")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument("--source_path", required=True, help="Path to source data")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration number")
    parser.add_argument("--feature_level", type=int, default=3, help="Feature level")
    parser.add_argument("--weight_threshold", type=float, default=1e-5, help="Weight threshold")
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size")
    
    args = parser.parse_args()
    
    # Check GPU memory
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"Total GPU memory: {total_memory:.2f} GB")
        
        # Set memory allocation strategy for better fragmentation handling
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("Set PyTorch CUDA allocation to expandable_segments")
    else:
        print("CUDA not available!")
        return
    
    # Import and run the feature extractor
    try:
        from gaussian_feature_extractor import process_scene_language_features_robust_efficient
        from arguments import ModelParams, PipelineParams, OptimizationParams
        
        # Create parameter objects (simplified)
        class SimpleArgs:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        # Set up arguments
        simple_args = SimpleArgs(
            model_path=args.model_path,
            source_path=args.source_path,
            sh_degree=3,
            resolution=-1,
            white_background=False,
            data_device="cuda",
            eval=False,
            quiet=True,
            test_iterations=[],
            save_iterations=[],
            checkpoint_iterations=[],
            start_checkpoint=None
        )
        
        # Create parameter extractors
        model_params = ModelParams(argparse.ArgumentParser(), sentinel=True)
        pipeline_params = PipelineParams(argparse.ArgumentParser())
        opt_params = OptimizationParams(argparse.ArgumentParser())
        
        # Run the robust efficient extraction
        print("Starting memory-efficient robust feature extraction...")
        process_scene_language_features_robust_efficient(
            model_params.extract(simple_args),
            opt_params.extract(simple_args), 
            args.iteration,
            pipeline_params.extract(simple_args),
            args.feature_level,
            args.weight_threshold,
            args.batch_size
        )
        
        print("Feature extraction completed successfully!")
        
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 