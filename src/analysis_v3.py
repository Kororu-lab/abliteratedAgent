
import torch
import numpy as np
import torch.nn.functional as F
# from models.model import DATA_ENCODER # Removed

# Re-defining constants or passing them? 
# Ideally passed as args or imported from utils.conf if generic
from utils.conf import ATTRI_SIZE, NUM_SYSTEM, DEVICE

def make_minimal_pairs(num_samples=1000, attribute_index=0):
    """
    Generates minimal pairs that differ ONLY in the specified attribute.
    Returns: (list of int array inputs, list of int array inputs)
    """
    inputs1 = []
    inputs2 = []
    
    for _ in range(num_samples):
        # Base input
        base = np.random.randint(0, NUM_SYSTEM, size=ATTRI_SIZE)
        
        # Create pair
        alt = base.copy()
        # Change only the target attribute
        new_val = np.random.randint(0, NUM_SYSTEM)
        while new_val == base[attribute_index]:
            new_val = np.random.randint(0, NUM_SYSTEM)
        alt[attribute_index] = new_val
        
        if attribute_index == 0:
            # Inputs are [attr0, attr1] -> integer representation?
            # Model expects integer input?
            # data item is int: val = attr0 + attr1*NUM_SYSTEM
            # But wait, original code: 
            # tmp = int(np.mod(data_batch[b] / (NUM_SYSTEM**i), NUM_SYSTEM))
            # so input is a single integer.
            val1 = base[0] + base[1] * NUM_SYSTEM
            val2 = alt[0] + alt[1] * NUM_SYSTEM
        else:
             val1 = base[0] + base[1] * NUM_SYSTEM
             val2 = alt[0] + alt[1] * NUM_SYSTEM
        
        inputs1.append(val1)
        inputs2.append(val2)
        
    return torch.tensor(inputs1).to(DEVICE), torch.tensor(inputs2).to(DEVICE)

def get_embeddings(agent, inputs_tensor):
    """
    Extracts h_A (hidden representation) from the speaker's encoder.
    """
    agent.eval()
    with torch.no_grad():
        # Agent encoder forward:
        # data_embeddings = self.gen_embedding(data_batch) 
        # return self.lin(data_embeddings)
        hidden = agent.encoder(inputs_tensor)
    return hidden.squeeze(1) # [N, Hidden]

def estimate_direction(agent, num_samples=1000, attr_idx=0, method='pca'):
    inputs1, inputs2 = make_minimal_pairs(num_samples, attr_idx)
    h1 = get_embeddings(agent, inputs1)
    h2 = get_embeddings(agent, inputs2)
    
    diffs = h1 - h2
    
    if method == 'pca':
        # PCA/SVD
        # Center? If we assume centered diffs represent direction.
        # SVD on diffs matrix
        _, _, V = torch.svd(diffs)
        direction = V[:, 0] # First component
    else:
        # Mean difference
        direction = diffs.mean(dim=0)
        direction = direction / direction.norm()
        
    return direction

def check_orthogonality(dir1, dir2):
    return F.cosine_similarity(dir1.unsqueeze(0), dir2.unsqueeze(0)).item()

def project_out(vectors, direction):
    """
    Removes the component of 'vectors' along 'direction'.
    vectors: [N, D]
    direction: [D] (assumed normalized)
    """
    # Proj_v(u) = (u . v) * v
    direction = direction / direction.norm()
    dot_prods = torch.matmul(vectors, direction.unsqueeze(1)) # [N, 1]
    projection = dot_prods * direction.unsqueeze(0) # [N, D]
    return vectors - projection

class AblatedEncoder(torch.nn.Module):
    def __init__(self, original_encoder, direction):
        super().__init__()
        self.original = original_encoder
        self.direction = direction

    def forward(self, input_data):
        # Original forward
        h = self.original(input_data) # [N, 1, Hidden]
        
        # Ablate
        h_squeezed = h.squeeze(1)
        h_ablated = project_out(h_squeezed, self.direction)
        
        return h_ablated.unsqueeze(1)
        
    def gen_embedding(self, data_batch):
        return self.original.gen_embedding(data_batch)

def evaluate_accuracy(speaker, listener, attr_idx, ablated_direction=None):
    """
    Evaluates communication accuracy on the specific attribute.
    If ablated_direction is provided, ablates that direction from speaker.
    """
    # Create test set varying ONLY in attr_idx? 
    # Or standard evaluation?
    # Standard evaluation is 2-val (target vs distractor)
    # We want to test if they can distinguish standard pairs.
    
    # Let's generate random batch, but measure attribute match?
    # Actually, simpler: Minimal pairs task.
    # Given target and distractor differing ONLY in attr_idx.
    # Can listener pick target?
    
    speaker.eval()
    listener.eval()
    
    if ablated_direction is not None:
        original_encoder = speaker.encoder
        speaker.encoder = AblatedEncoder(original_encoder, ablated_direction)
    
    correct = 0
    total = 200 # samples
    
    inputs1, inputs2 = make_minimal_pairs(total, attr_idx)
    # inputs1 is target, inputs2 is distractor
    
    # Candidates tensor: [N, 2]? No, model expects [N, SEL_CANDID]
    # We must pad with randoms.
    from utils.conf import SEL_CANDID
    
    # We construct batch where:
    # Item 0 is Target (inputs1)
    # Item 1 is Distractor (inputs2)
    # Items 2..K are random
    
    target_batch = inputs1
    
    # Candidates: [N, SEL_CANDID]
    cands = torch.zeros((total, SEL_CANDID), dtype=torch.long).to(DEVICE)
    cands[:, 0] = inputs1
    cands[:, 1] = inputs2
    # Fill rest with random
    for k in range(2, SEL_CANDID):
         cands[:, k] = torch.randint(0, NUM_SYSTEM*NUM_SYSTEM, (total,)).to(DEVICE)
         
    # Run communication
    # 1. Speaker generates msg for target
    # 2. Listener picks from cands
    
    # Speaker forward
    # msg: [N, L]
    # We need full generation logic. 
    # Calling internal methods might be complex. 
    # Use existing `train_phaseC` or similar? 
    # Or just `msg_generator`? `msg_generator` does all.
    
    # But `msg_generator` is for analysis of ALL items.
    # Let's use `speaker.forward`? No, speaker is LSTM.
    # We need `speaker(data)` -> prob -> sample. 
    # Usually `train_phaseA` does this training. 
    # Let's replicate simple generation (Argmax).
    
    with torch.no_grad():
        # Speaker Generate
        # In eval mode, speaker returns greedy argmax predictions as one-hot vectors
        # Returns: msg, log_prob, entropy, digits
        msg, _, _, _ = speaker(target_batch)
        
        # Listener Predict
        # Input: candidates, msg
        # pred_vector: [N, SEL_CANDID] (logits)
        pred_vector = listener(cands, msg)
        
        # Calculate Accuracy
        # Target is index 0
        pred_idx = pred_vector.argmax(dim=1)
        correct = (pred_idx == 0).sum().item()
        
    if ablated_direction is not None:
        speaker.encoder = original_encoder # Restore
        
    return correct / total

def calculate_sii(speaker, listener):
    """
    Calculates SII for both Color and Shape and returns metrics.
    """
    # 1. Estimate Directions
    v_color = estimate_direction(speaker, attr_idx=0, method='pca')
    v_shape = estimate_direction(speaker, attr_idx=1, method='pca')
    
    # 2. Check Orthogonality
    ortho = check_orthogonality(v_color, v_shape)
    
    # 3. Baseline Accuracy
    acc_color_base = evaluate_accuracy(speaker, listener, attr_idx=0)
    acc_shape_base = evaluate_accuracy(speaker, listener, attr_idx=1)
    
    # 4. Ablated Accuracy (Color Ablation)
    acc_color_ablate_c = evaluate_accuracy(speaker, listener, attr_idx=0, ablated_direction=v_color)
    acc_shape_ablate_c = evaluate_accuracy(speaker, listener, attr_idx=1, ablated_direction=v_color)
    
    # 5. Ablated Accuracy (Shape Ablation)
    acc_color_ablate_s = evaluate_accuracy(speaker, listener, attr_idx=0, ablated_direction=v_shape)
    acc_shape_ablate_s = evaluate_accuracy(speaker, listener, attr_idx=1, ablated_direction=v_shape)
    
    # 6. Calculate SII
    # SII(Color) = (Acc_Shape(Abl_C) - Acc_Color(Abl_C)) - (Acc_Shape(Base) - Acc_Color(Base)) ?
    # Or simplified: SII = Drop_Target - Drop_Control
    # Drop_Target = Acc_Base - Acc_Ablated
    # Drop_Control = Acc_Base_Control - Acc_Ablated_Control (should be ~0)
    # SII = (Acc_Target_Base - Acc_Target_Abl) - (Acc_Control_Base - Acc_Control_Abl)
    
    # Color SII
    drop_color_c = acc_color_base - acc_color_ablate_c # Should be large
    drop_shape_c = acc_shape_base - acc_shape_ablate_c # Should be small
    sii_color = drop_color_c - drop_shape_c
    
    # Shape SII
    drop_shape_s = acc_shape_base - acc_shape_ablate_s # Should be large
    drop_color_s = acc_color_base - acc_color_ablate_s # Should be small
    sii_shape = drop_shape_s - drop_color_s
    
    return {
        'sii_color': sii_color,
        'sii_shape': sii_shape,
        'orthogonality': ortho,
        'acc_color_base': acc_color_base,
        'acc_shape_base': acc_shape_base,
        'acc_color_ablate_c': acc_color_ablate_c,
        'acc_shape_ablate_c': acc_shape_ablate_c,
        'acc_color_ablate_s': acc_color_ablate_s,
        'acc_shape_ablate_s': acc_shape_ablate_s
    }


