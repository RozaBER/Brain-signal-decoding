"""
Main script to train and evaluate the Brain-to-Text decoding system.
"""
import os
import argparse
import torch
import config
from train import train
from meg_encoder import get_meg_encoder
from alignment_module import AlignmentModule
from llava_decoder import BrainLLaVA
from data_loading import create_data_loaders
from utils import set_seed, load_checkpoint, evaluate_bleu

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Brain-to-Text Decoding System")
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "inference"],
                        help="Mode of operation")
    
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file for evaluation or inference")
    
    parser.add_argument("--output_dir", type=str, default=config.RESULTS_PATH,
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    return args

def evaluate(model, test_loader, device=config.DEVICE):
    """Evaluate model performance on test set."""
    model.eval()
    
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for batch in test_loader:
            meg_data = batch['meg_data'].to(device)
            raw_text = batch['raw_text']
            
            # Generate text from MEG data
            generated_text = model(meg_data, max_length=100)
            
            # Collect references and hypotheses
            all_references.extend(raw_text)
            all_hypotheses.extend(generated_text)
    
    # Calculate BLEU score
    bleu_score = evaluate_bleu(all_references, all_hypotheses)
    
    print(f"BLEU Score: {bleu_score:.4f}")
    
    # Print some examples
    num_examples = min(5, len(all_references))
    print("\nExample Generations:")
    for i in range(num_examples):
        print(f"Reference: {all_references[i]}")
        print(f"Generated: {all_hypotheses[i]}")
        print()
    
    return bleu_score, all_references, all_hypotheses

def inference(model, test_loader, device=config.DEVICE):
    """Run inference and save results."""
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            meg_data = batch['meg_data'].to(device)
            raw_text = batch['raw_text']
            subject = batch['subject']
            task = batch['task']
            
            # Generate text from MEG data
            generated_text = model(meg_data, max_length=100)
            
            # Collect results
            for j in range(len(raw_text)):
                results.append({
                    'sample_id': i * config.BATCH_SIZE + j,
                    'subject': subject[j],
                    'task': task[j],
                    'reference': raw_text[j],
                    'generated': generated_text[j]
                })
    
    # Save results to file
    import json
    with open(os.path.join(config.RESULTS_PATH, "inference_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Inference results saved to {os.path.join(config.RESULTS_PATH, 'inference_results.json')}")
    
    return results

def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(config.SEED)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "train":
        # Train model
        model = train()
    else:
        # Load model for evaluation or inference
        if args.checkpoint is None:
            args.checkpoint = os.path.join(config.CHECKPOINTS_PATH, "brain_llava_final.pt")
        
        # Initialize model architecture
        meg_encoder = get_meg_encoder().to(config.DEVICE)
        alignment_module = AlignmentModule().to(config.DEVICE)
        model = BrainLLaVA(meg_encoder, alignment_module).to(config.DEVICE)
        
        # Load checkpoint
        model = load_checkpoint(model, args.checkpoint)
        
        # Create data loaders
        _, _, test_loader = create_data_loaders()
        
        if args.mode == "evaluate":
            # Evaluate model
            evaluate(model, test_loader)
        elif args.mode == "inference":
            # Run inference
            inference(model, test_loader)

if __name__ == "__main__":
    main()