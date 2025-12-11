
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize(path):
    save_path = f'exp_results/{path}/'
    
    # Load data
    try:
        sii_history = np.load(save_path + 'sii_history.npy', allow_pickle=True)
        # sii_history is a list of dicts.
        
        # Parse list of dicts
        gens = [x['generation'] for x in sii_history]
        sii_color = [x['sii_color'] for x in sii_history]
        sii_shape = [x['sii_shape'] for x in sii_history]
        ortho = [x['orthogonality'] for x in sii_history]
        acc_c_base = [x['acc_color_base'] for x in sii_history]
        acc_s_base = [x['acc_shape_base'] for x in sii_history]
        
    except Exception as e:
        print(f"Error loading sii_history: {e}")
        return

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Accuracy
    axes[0].plot(gens, acc_c_base, label='Color Accuracy (Base)', color='blue')
    axes[0].plot(gens, acc_s_base, label='Shape Accuracy (Base)', color='green')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Baseline Accuracy over Generations')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. SII
    axes[1].plot(gens, sii_color, label='SII Color', color='blue', linestyle='--')
    axes[1].plot(gens, sii_shape, label='SII Shape', color='green', linestyle='--')
    axes[1].axhline(0, color='gray', linewidth=0.5)
    axes[1].set_ylabel('SII Score')
    axes[1].set_title('Selective Impairment Index (Higher is Better)')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. Orthogonality
    axes[2].plot(gens, ortho, label='Cosine Sim (Color vs Shape)', color='red')
    axes[2].set_ylabel('Cosine Similarity')
    axes[2].set_xlabel('Generation')
    axes[2].set_title('Orthogonality of Attribute Directions (Lower Magnitude is More Orthogonal)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.1) # Extra margin
    plt.savefig(save_path + 'analysis_v3.png', bbox_inches='tight')
    print(f"Saved plot to {save_path}analysis_v3.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Experiment path name')
    args = parser.parse_args()
    
    visualize(args.path)
