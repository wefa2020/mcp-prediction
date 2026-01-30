"""
SageMaker inference handler - PURE MODEL INFERENCE ONLY.
Takes preprocessed features, returns predictions.
No Neptune access - that's handled by Lambda.

UPDATED: Now uses new feature structure matching data_preprocessor.py and dataset.py
- node_observable_time, node_observable_other
- node_realized_time, node_realized_other
- Simplified node categorical (no from/to split)
- Edge features (continuous only)
- No lookahead features (model handles causality)
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Union, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model components
model = None
preprocessor = None
device = None


def model_fn(model_dir: str):
    """Load model from model_dir."""
    global model, preprocessor, device
    
    import gc
    
    logger.info(f"Loading model from {model_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Import modules
    from config import Config  # <-- Changed: Import Config, not ModelConfig
    from models.event_predictor import EventTimePredictor
    from data.data_preprocessor import PackageLifecyclePreprocessor
    
    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    logger.info(f"Loading preprocessor from {preprocessor_path}")
    preprocessor = PackageLifecyclePreprocessor.load(preprocessor_path)
    logger.info(f"Preprocessor loaded from {preprocessor_path}")
    
    # Load checkpoint
    checkpoint_path = os.path.join(model_dir, 'best_model.pt')
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract from checkpoint (matching working code)
    vocab_sizes = checkpoint['vocab_sizes']
    feature_dims = checkpoint['feature_dims']  # <-- Added: Need feature_dims
    
    # Load FULL config (not just model config)
    full_config = Config.from_dict(checkpoint['config'])  # <-- Changed: Use 'config' not 'model_config'
    
    logger.info(f"vocab_sizes: {vocab_sizes}")
    logger.info(f"feature_dims: {feature_dims}")
    
    # Use from_config() method (matching working code)
    model = EventTimePredictor.from_config(
        config=full_config,
        vocab_sizes=vocab_sizes,
        feature_dims=feature_dims,
        device=device,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Clear checkpoint from memory
    del checkpoint
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    logger.info(f"Model loaded successfully")
    
    # Simple CUDA warmup
    if device.type == 'cuda':
        logger.info("CUDA warmup (simple)...")
        try:
            _ = torch.randn(1, 1, device=device) * 2
            torch.cuda.synchronize()
            logger.info("CUDA warmup complete")
        except Exception as e:
            logger.warning(f"CUDA warmup failed: {e}")
    
    return {'model': model, 'preprocessor': preprocessor, 'device': device}

def input_fn(request_body: Union[str, bytes], request_content_type: str) -> Dict:
    """
    Deserialize input data.
    Handle both bytes and string input.
    """
    logger.info(f"input_fn called with content_type: {request_content_type}")
    logger.info(f"request_body type: {type(request_body)}")
    
    if request_content_type == 'application/json':
        # Handle bytes input
        if isinstance(request_body, bytes):
            logger.info(f"Converting bytes to string, length: {len(request_body)}")
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
                try:
                    request_body = request_body.decode(encoding)
                    break
                except (UnicodeDecodeError, AttributeError):
                    continue
        
        if isinstance(request_body, bytes):
            request_body = request_body.decode('utf-8', errors='replace')
        
        logger.info(f"Parsing JSON, string length: {len(request_body)}")
        
        try:
            data = json.loads(request_body)
            logger.info(f"JSON parsed successfully, keys: {list(data.keys())}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise ValueError(f"Invalid JSON: {e}")
    
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: Dict, model_components: Dict) -> Dict:
    """
    Run prediction on preprocessed package data.
    """
    global model, preprocessor, device
    
    logger.info(f"predict_fn called with input keys: {list(input_data.keys())}")
    
    from torch_geometric.data import Data, Batch
    
    action = input_data.get('action', 'predict')
    logger.info(f"Action: {action}")
    
    if action == 'health':
        gpu_mem = None
        if device.type == 'cuda':
            gpu_mem = {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
            }
        return {
            'status': 'healthy',
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_memory': gpu_mem,
            'model_loaded': model is not None
        }
    
    if action != 'predict':
        return {'error': f'Unknown action: {action}', 'status': 'error'}
    
    packages = input_data.get('packages', [])
    if not packages:
        return {'error': 'No packages provided', 'status': 'error'}
    
    logger.info(f"Processing {len(packages)} packages")
    results = []
    
    for i, pkg in enumerate(packages):
        pkg_id = pkg.get('package_id', f'unknown_{i}')
        logger.info(f"Processing package {i+1}/{len(packages)}: {pkg_id}")
        
        try:
            # Process package through preprocessor
            logger.info(f"[{pkg_id}] Processing features...")
            features = preprocessor.process_lifecycle(pkg, return_labels=True)
            
            if features is None:
                logger.warning(f"[{pkg_id}] Failed to process features")
                results.append({
                    'package_id': pkg_id,
                    'status': 'error',
                    'error': 'Failed to process package features'
                })
                continue
            
            logger.info(f"[{pkg_id}] Features processed, num_nodes: {features.get('num_nodes')}")
            
            # Convert to PyG Data
            logger.info(f"[{pkg_id}] Converting to PyG Data...")
            graph_data = _features_to_pyg_data(features)
            graph_data = graph_data.to(device)
            batch = Batch.from_data_list([graph_data])
            
            # Add batch metadata
            batch.node_counts = torch.tensor([graph_data.num_nodes], dtype=torch.long, device=device)
            batch.edge_counts = torch.tensor([graph_data.edge_index.shape[1]], dtype=torch.long, device=device)
            
            # Run inference
            logger.info(f"[{pkg_id}] Running inference...")
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    predictions = model(batch)
            
            logger.info(f"[{pkg_id}] Inference complete, extracting predictions...")
            
            # Extract predictions
            # Predictions are per-edge (transition times between consecutive events)
            num_edges = graph_data.edge_index.shape[1]
            
            if predictions.dim() > 1:
                preds = predictions[:num_edges].squeeze(-1)
            else:
                preds = predictions[:num_edges]
            
            preds_scaled = preds.float().cpu().numpy()
            preds_hours = preprocessor.inverse_transform_time(preds_scaled).flatten()
            
            logger.info(f"[{pkg_id}] Predictions (hours): {preds_hours.tolist()}")
            
            # Include ground truth labels if available (for debugging)
            result_entry = {
                'package_id': pkg_id,
                'status': 'success',
                'predictions_hours': preds_hours.tolist(),
                'num_transitions': num_edges,
            }
            
            if 'labels_raw' in features and features['labels_raw'] is not None:
                labels_raw = features['labels_raw'].flatten()
                result_entry['ground_truth_hours'] = labels_raw.tolist()
            
            results.append(result_entry)
            
        except Exception as e:
            logger.error(f"[{pkg_id}] Error: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[{pkg_id}] Traceback: {traceback.format_exc()}")
            results.append({
                'package_id': pkg_id,
                'status': 'error',
                'error': str(e)
            })
    
    logger.info(f"Completed processing, {len(results)} results")
    
    return {
        'status': 'success',
        'results': results
    }


def _to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert numpy array to tensor with specified dtype."""
    if dtype == torch.long:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))
    elif dtype == torch.float32:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    elif dtype == torch.bool:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.bool_))
    return torch.from_numpy(arr)


def _features_to_pyg_data(features: Dict) -> 'Data':
    """
    Convert preprocessor features to PyG Data object.
    
    Matches the structure expected by SharedMemoryCollator in dataset.py:
    - node_observable_time, node_observable_other
    - node_realized_time, node_realized_other  
    - Node categorical: event_type, location, postal, region, carrier, leg_type, ship_method
    - edge_index, edge_features
    - package_features, source_postal_idx, dest_postal_idx
    """
    from torch_geometric.data import Data
    
    # Get node categorical indices
    node_cat = features['node_categorical_indices']
    pkg_cat = features['package_categorical']
    
    num_nodes = features['num_nodes']
    num_edges = features['edge_index'].shape[1] if features['edge_index'].size > 0 else 0
    
    # Build Data object matching dataset.py structure
    data = Data(
        # === Node continuous features (Time2Vec ready) ===
        node_observable_time=_to_tensor(features['node_observable_time'], torch.float32),
        node_observable_other=_to_tensor(features['node_observable_other'], torch.float32),
        node_realized_time=_to_tensor(features['node_realized_time'], torch.float32),
        node_realized_other=_to_tensor(features['node_realized_other'], torch.float32),
        
        # === Node categorical indices ===
        event_type_idx=_to_tensor(node_cat['event_type'], torch.long),
        location_idx=_to_tensor(node_cat['location'], torch.long),
        postal_idx=_to_tensor(node_cat['postal'], torch.long),
        region_idx=_to_tensor(node_cat['region'], torch.long),
        carrier_idx=_to_tensor(node_cat['carrier'], torch.long),
        leg_type_idx=_to_tensor(node_cat['leg_type'], torch.long),
        ship_method_idx=_to_tensor(node_cat['ship_method'], torch.long),
        
        # === Edge features ===
        edge_index=_to_tensor(features['edge_index'], torch.long),
        edge_features=_to_tensor(features['edge_features'], torch.float32),
        
        # === Package features ===
        package_features=_to_tensor(
            features['package_features'].reshape(1, -1) if features['package_features'].ndim == 1 
            else features['package_features'], 
            torch.float32
        ),
        source_postal_idx=torch.tensor([pkg_cat['source_postal']], dtype=torch.long),
        dest_postal_idx=torch.tensor([pkg_cat['dest_postal']], dtype=torch.long),
        
        # === Metadata ===
        num_nodes=num_nodes,
    )
    
    # === Labels (if available) ===
    if 'labels' in features and features['labels'] is not None:
        labels = features['labels']
        if labels.ndim > 1:
            labels = labels.flatten()
        data.edge_labels = _to_tensor(labels, torch.float32)
    
    if 'labels_raw' in features and features['labels_raw'] is not None:
        labels_raw = features['labels_raw']
        if labels_raw.ndim > 1:
            labels_raw = labels_raw.flatten()
        data.edge_labels_raw = _to_tensor(labels_raw, torch.float32)
    
    return data


def output_fn(prediction: Dict, response_content_type: str) -> str:
    """Serialize prediction output."""
    logger.info(f"output_fn called with content_type: {response_content_type}")
    
    if response_content_type == 'application/json':
        result = json.dumps(prediction, default=str)
        logger.info(f"Output JSON length: {len(result)}")
        return result
    
    raise ValueError(f"Unsupported content type: {response_content_type}")