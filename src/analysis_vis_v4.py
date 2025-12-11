
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import SpeakingAgent, ListeningAgent
from utils.conf import DEVICE, ATTRI_SIZE, NUM_SYSTEM
from analysis_v3 import estimate_direction, get_embeddings, make_minimal_pairs, project_out, AblatedEncoder

def load_model(path):
    speaker = SpeakingAgent().to(DEVICE)
    listener = ListeningAgent().to(DEVICE)
    
    checkpoint = torch.load(f'exp_results/{path}/model.pth', map_location=DEVICE)
    speaker.load_state_dict(checkpoint['speaker_state_dict'])
    listener.load_state_dict(checkpoint['listener_state_dict'])
    
    speaker.eval()
    listener.eval()
    return speaker, listener

def verify_ablation_mathematically(speaker, attr_idx=0):
    print(f"\n--- Verifying Ablation Math (Attribute {attr_idx}) ---")
    
    # 1. Estimate Direction
    v = estimate_direction(speaker, attr_idx=attr_idx, method='pca')
    v = v / v.norm()
    
    # 2. Get random embeddings
    inputs, _ = make_minimal_pairs(100, attr_idx)
    h_original = get_embeddings(speaker, inputs)
    
    # 3. Ablate manually
    h_ablated = project_out(h_original, v)
    
    # 4. Check Projections
    proj_orig = torch.matmul(h_original, v.unsqueeze(1)).squeeze()
    proj_abl = torch.matmul(h_ablated, v.unsqueeze(1)).squeeze()
    
    mean_orig = proj_orig.abs().mean().item()
    mean_abl = proj_abl.abs().mean().item()
    
    print(f"Mean Absolute Projection (Original): {mean_orig:.6f}")
    print(f"Mean Absolute Projection (Ablated):  {mean_abl:.6f} (Should be ~0)")
    
    is_success = mean_abl < 1e-5
    print(f"Verification: {'SUCCESS' if is_success else 'FAILURE'}")
    return is_success

def plot_latent_space_pca(speaker, save_path):
    print("Generating PCA Scatter Plots...")
    # Generate random data covering all combinations
    # We want to color by Attr0 and Attr1
    
    N = 1000
    # Generate random inputs
    # inputs: [N] where val = a0 + a1*NUM_SYSTEM
    a0 = np.random.randint(0, NUM_SYSTEM, N)
    a1 = np.random.randint(0, NUM_SYSTEM, N)
    inputs = torch.tensor(a0 + a1 * NUM_SYSTEM).to(DEVICE)
    
    h = get_embeddings(speaker, inputs).cpu().numpy()
    
    pca = PCA(n_components=2)
    h_2d = pca.fit_transform(h)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Color by Attribute 0 (Color)
    sc0 = axes[0].scatter(h_2d[:, 0], h_2d[:, 1], c=a0, cmap='viridis', alpha=0.7)
    axes[0].set_title('PCA: Colored by Color (Attr 0)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    plt.colorbar(sc0, ax=axes[0], label='Color Value')
    
    # Color by Attribute 1 (Shape)
    sc1 = axes[1].scatter(h_2d[:, 0], h_2d[:, 1], c=a1, cmap='plasma', alpha=0.7)
    axes[1].set_title('PCA: Colored by Shape (Attr 1)')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.colorbar(sc1, ax=axes[1], label='Shape Value')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/vis_pca_scatter.png')
    plt.close()

def plot_projection_heatmap(speaker, save_path):
    print("Generating Projection Heatmaps...")
    grid = np.zeros((NUM_SYSTEM, NUM_SYSTEM))
    
    # Estimate direction for Color (Attr 0)
    v_color = estimate_direction(speaker, attr_idx=0, method='pca')
    v_color = v_color / v_color.norm()
    
    # Heatmap 1: Color Projection
    for c in range(NUM_SYSTEM):
        for s in range(NUM_SYSTEM):
            val = c + s * NUM_SYSTEM
            inputs = torch.tensor([val]).to(DEVICE)
            h = get_embeddings(speaker, inputs) 
            proj = torch.matmul(h, v_color.unsqueeze(1)).item()
            grid[s, c] = proj # Row=Shape, Col=Color
            
    plt.figure(figsize=(6, 5))
    sns.heatmap(grid, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Projection onto "Color" Vector\n(X=Color, Y=Shape)')
    plt.xlabel('Color Value')
    plt.ylabel('Shape Value')
    plt.tight_layout()
    plt.savefig(f'{save_path}/vis_heatmap_color_proj.png')
    plt.close()
    
    # Heatmap 2: Shape Projection
    v_shape = estimate_direction(speaker, attr_idx=1, method='pca')
    v_shape = v_shape / v_shape.norm()
    grid_s = np.zeros((NUM_SYSTEM, NUM_SYSTEM))
    
    for c in range(NUM_SYSTEM):
        for s in range(NUM_SYSTEM):
            val = c + s * NUM_SYSTEM
            inputs = torch.tensor([val]).to(DEVICE)
            h = get_embeddings(speaker, inputs) 
            proj = torch.matmul(h, v_shape.unsqueeze(1)).item()
            grid_s[s, c] = proj
            
    plt.figure(figsize=(6, 5))
    sns.heatmap(grid_s, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Projection onto "Shape" Vector\n(X=Color, Y=Shape)')
    plt.xlabel('Color Value')
    plt.ylabel('Shape Value')
    plt.tight_layout()
    plt.savefig(f'{save_path}/vis_heatmap_shape_proj.png')
    plt.close()

def plot_comprehensive_metrics(run_path, save_path):
    print("Generating Comprehensive Trajectory Plot...")
    try:
        sii_history = np.load(f'exp_results/{run_path}/sii_history.npy', allow_pickle=True)
        gens = [x['generation'] for x in sii_history]
        
        # 1. Baseline Accuracies
        acc_c = np.array([x['acc_color_base'] for x in sii_history])
        acc_s = np.array([x['acc_shape_base'] for x in sii_history])
        
        # 2. Relative Error
        acc_c_abl = np.array([x['acc_color_ablate_c'] for x in sii_history])
        acc_s_abl = np.array([x['acc_shape_ablate_s'] for x in sii_history])
        eps = 1e-6
        rel_err_c = ((1-acc_c_abl) - (1-acc_c)) / ((1-acc_c) + eps)
        rel_err_s = ((1-acc_s_abl) - (1-acc_s)) / ((1-acc_s) + eps)
        
        # 3. SII Scores
        sii_c = np.array([x['sii_color'] for x in sii_history])
        sii_s = np.array([x['sii_shape'] for x in sii_history])
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # Plot 1: Baseline Accuracy (Learning Curve)
        axes[0].plot(gens, acc_c, label='Baseline Color Acc', color='blue')
        axes[0].plot(gens, acc_s, label='Baseline Shape Acc', color='green')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('A. Learning Trajectory (Baseline Accuracy)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Relative Error (Impact)
        axes[1].plot(gens, rel_err_c, label='Rel. Error Increase (Ablate Color)', color='blue', linestyle='--')
        axes[1].plot(gens, rel_err_s, label='Rel. Error Increase (Ablate Shape)', color='green', linestyle='--')
        axes[1].set_ylabel('Rel. Error Incr.')
        axes[1].set_title('B. Impact of Selective Ablation (Values > 0 imply impairment)')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot 3: SII Score (Disentanglement)
        axes[2].plot(gens, sii_c, label='SII (Color)', color='blue')
        axes[2].plot(gens, sii_s, label='SII (Shape)', color='green')
        axes[2].axhline(0, color='gray', linewidth=0.5)
        axes[2].set_ylabel('SII Score')
        axes[2].set_xlabel('Generation')
        axes[2].set_title('C. Selective Impairment Index (Overall Disentanglement)')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/vis_comprehensive.png', bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Plotting error: {e}")


if __name__ == '__main__':
    run_path = 'run_v3_retry' # default
    if len(sys.argv) > 1:
        run_path = sys.argv[1]
        
    print(f"Analyzing {run_path}...")
    save_path = f'exp_results/{run_path}'
    
    agent, listener = load_model(run_path)
    
    # 1. Verification
    verify_ablation_mathematically(agent, attr_idx=0)
    verify_ablation_mathematically(agent, attr_idx=1)
    
    # 2. Plots
    plot_latent_space_pca(agent, save_path)
    plot_projection_heatmap(agent, save_path)
    plot_comprehensive_metrics(run_path, save_path)
    
    print("Done.")
