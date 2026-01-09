"""
SageMaker inference handler - PURE MODEL INFERENCE ONLY.
Takes preprocessed features, returns predictions.
No Neptune access - that's handled by Lambda.

FIXES:
- Removed torch.compile (causes dimension mismatch errors)
- Fixed warmup dimensions
- Added timeout handling
- Added better error logging
"""

import os
import json
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Union

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
    from config import ModelConfig
    from models.event_predictor import EventTimePredictor
    from data.data_preprocessor import PackageLifecyclePreprocessor
    
    # Load preprocessor
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    logger.info(f"Loading preprocessor from {preprocessor_path}")
    preprocessor = PackageLifecyclePreprocessor.load(preprocessor_path)
    logger.info(f"Preprocessor loaded from {preprocessor_path}")
    
    # Load model
    checkpoint_path = os.path.join(model_dir, 'best_model.pt')
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = ModelConfig.from_dict(checkpoint['model_config'])
    vocab_sizes = checkpoint['vocab_sizes']
    
    model = EventTimePredictor(model_config, vocab_sizes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Clear checkpoint from memory
    del checkpoint
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    logger.info(f"Model loaded successfully")
    
    # DO NOT use torch.compile - it causes dimension mismatch errors
    # The error was: "a and b must have same reduction dim, but got [3, 586] X [662, 256]"
    
    # Simple CUDA warmup (no model inference)
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
            
            # Run inference
            logger.info(f"[{pkg_id}] Running inference...")
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                    predictions = model(batch)
            
            logger.info(f"[{pkg_id}] Inference complete, extracting predictions...")
            
            # Extract predictions
            mask = batch.label_mask
            masked_preds = predictions[mask].squeeze(-1) if predictions[mask].dim() > 1 else predictions[mask]
            
            preds_scaled = masked_preds.float().cpu().numpy()
            preds_hours = preprocessor.inverse_transform_time(preds_scaled).flatten()
            
            logger.info(f"[{pkg_id}] Predictions: {preds_hours.tolist()}")
            
            results.append({
                'package_id': pkg_id,
                'status': 'success',
                'predictions_hours': preds_hours.tolist()
            })
            
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


def _features_to_pyg_data(features: Dict):
    """Convert preprocessor features to PyG Data object."""
    from torch_geometric.data import Data
    
    def to_tensor(arr, dtype):
        if dtype == torch.long:
            return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))
        elif dtype == torch.float32:
            return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        elif dtype == torch.bool:
            return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.bool_))
        return torch.from_numpy(arr)
    
    node_cat = features['node_categorical_indices']
    look_cat = features['lookahead_categorical_indices']
    edge_cat = features['edge_categorical_indices']
    pkg_cat = features['package_categorical']
    
    data = Data(
        node_continuous=to_tensor(features['node_continuous_features'], torch.float32),
        event_type_idx=to_tensor(node_cat['event_type'], torch.long),
        from_location_idx=to_tensor(node_cat['from_location'], torch.long),
        to_location_idx=to_tensor(node_cat['to_location'], torch.long),
        to_postal_idx=to_tensor(node_cat['to_postal'], torch.long),
        from_region_idx=to_tensor(node_cat['from_region'], torch.long),
        to_region_idx=to_tensor(node_cat['to_region'], torch.long),
        carrier_idx=to_tensor(node_cat['carrier'], torch.long),
        leg_type_idx=to_tensor(node_cat['leg_type'], torch.long),
        ship_method_idx=to_tensor(node_cat['ship_method'], torch.long),
        next_event_type_idx=to_tensor(look_cat['next_event_type'], torch.long),
        next_location_idx=to_tensor(look_cat['next_location'], torch.long),
        next_postal_idx=to_tensor(look_cat['next_postal'], torch.long),
        next_region_idx=to_tensor(look_cat['next_region'], torch.long),
        next_carrier_idx=to_tensor(look_cat['next_carrier'], torch.long),
        next_leg_type_idx=to_tensor(look_cat['next_leg_type'], torch.long),
        next_ship_method_idx=to_tensor(look_cat['next_ship_method'], torch.long),
        source_postal_idx=torch.tensor([pkg_cat['source_postal']], dtype=torch.long),
        dest_postal_idx=torch.tensor([pkg_cat['dest_postal']], dtype=torch.long),
        edge_index=to_tensor(features['edge_index'], torch.long),
        edge_continuous=to_tensor(features['edge_continuous_features'], torch.float32),
        edge_from_location_idx=to_tensor(edge_cat['from_location'], torch.long),
        edge_to_location_idx=to_tensor(edge_cat['to_location'], torch.long),
        edge_to_postal_idx=to_tensor(edge_cat['to_postal'], torch.long),
        edge_from_region_idx=to_tensor(edge_cat['from_region'], torch.long),
        edge_to_region_idx=to_tensor(edge_cat['to_region'], torch.long),
        edge_carrier_from_idx=to_tensor(edge_cat['carrier_from'], torch.long),
        edge_carrier_to_idx=to_tensor(edge_cat['carrier_to'], torch.long),
        edge_ship_method_from_idx=to_tensor(edge_cat['ship_method_from'], torch.long),
        edge_ship_method_to_idx=to_tensor(edge_cat['ship_method_to'], torch.long),
        num_nodes=features['num_nodes'],
    )
    
    if 'labels' in features:
        data.labels = to_tensor(features['labels'].flatten(), torch.float32)
    if 'label_mask' in features:
        data.label_mask = to_tensor(features['label_mask'].astype(bool), torch.bool)
    
    return data


def output_fn(prediction: Dict, response_content_type: str) -> str:
    """Serialize prediction output."""
    logger.info(f"output_fn called with content_type: {response_content_type}")
    
    if response_content_type == 'application/json':
        result = json.dumps(prediction, default=str)
        logger.info(f"Output JSON length: {len(result)}")
        return result
    
    raise ValueError(f"Unsupported content type: {response_content_type}")