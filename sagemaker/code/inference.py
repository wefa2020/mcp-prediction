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
import multiprocessing
import time
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
    Optimized batch prediction with parallel preprocessing and last-prediction-only results.
    """
    global model, preprocessor, device
    
    logger.info(f"predict_fn called with input keys: {list(input_data.keys())}")
    
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
    
    logger.info(f"Processing {len(packages)} packages with optimized batch processing")
    
    # Always use batch processing for optimal performance
    return _predict_batch_optimized(packages)


def _predict_batch_optimized(packages: List[Dict]) -> Dict:
    """
    Optimized batch processing with maximum CPU utilization and last-prediction-only results.
    """
    global model, preprocessor, device
    
    from torch_geometric.data import Batch
    
    logger.info(f"[BATCH] Processing {len(packages)} packages with batch optimization")
    start_time = time.time()
    
    # Step 1: Parallel preprocessing - use max CPU power (CPU count - 1)
    max_cpu_workers = max(1, multiprocessing.cpu_count() - 1)
    actual_workers = min(max_cpu_workers, len(packages))
    
    logger.info(f"[BATCH] Starting parallel preprocessing with {actual_workers} workers (max available: {max_cpu_workers})")
    preprocessing_start = time.time()
    
    try:
        # Use threaded batch processing to avoid serialization issues
        features_list = preprocessor.process_lifecycle_batch_threaded(
            packages, 
            return_labels=True,
            max_workers=actual_workers
        )
    except Exception as e:
        logger.warning(f"[BATCH] Batch preprocessing failed, falling back to sequential: {e}")
        features_list = []
        for pkg in packages:
            try:
                features = preprocessor.process_lifecycle(pkg, return_labels=True)
                features_list.append(features)
            except Exception as pkg_e:
                logger.error(f"[BATCH] Failed to process {pkg.get('package_id', 'unknown')}: {pkg_e}")
                features_list.append(None)
    
    preprocessing_time = time.time() - preprocessing_start
    logger.info(f"[BATCH] Preprocessing completed in {preprocessing_time:.3f}s")
    
    # Step 2: Convert to PyG graphs and create batch
    logger.info("[BATCH] Converting to PyG graphs...")
    conversion_start = time.time()
    
    valid_graphs = []
    valid_indices = []
    results = []
    
    for i, (pkg, features) in enumerate(zip(packages, features_list)):
        pkg_id = pkg.get('package_id', f'unknown_{i}')
        
        if features is None:
            results.append({
                'package_id': pkg_id,
                'status': 'error',
                'error': 'Failed to process package features'
            })
            continue
        
        try:
            graph_data = _features_to_pyg_data(features)
            graph_data.pkg_id = pkg_id  # Store package ID for later reference
            graph_data.pkg_index = i
            valid_graphs.append(graph_data)
            valid_indices.append(i)
            
            # Add placeholder result that will be updated
            results.append({
                'package_id': pkg_id,
                'status': 'pending'
            })
            
        except Exception as e:
            logger.error(f"[BATCH] Failed to convert {pkg_id} to PyG: {e}")
            results.append({
                'package_id': pkg_id,
                'status': 'error',
                'error': f'Failed to convert to graph: {str(e)}'
            })
    
    if not valid_graphs:
        logger.warning("[BATCH] No valid graphs to process")
        return {'status': 'success', 'results': results}
    
    conversion_time = time.time() - conversion_start
    logger.info(f"[BATCH] Graph conversion completed in {conversion_time:.3f}s, {len(valid_graphs)} valid graphs")
    
    # Step 3: Batch inference
    logger.info("[BATCH] Running batch inference...")
    inference_start = time.time()
    
    try:
        # Move all graphs to device and create batch
        device_graphs = [graph.to(device) for graph in valid_graphs]
        batch = Batch.from_data_list(device_graphs)
        
        # Add batch metadata
        node_counts = [graph.num_nodes for graph in device_graphs]
        edge_counts = [graph.edge_index.shape[1] for graph in device_graphs]
        
        batch.node_counts = torch.tensor(node_counts, dtype=torch.long, device=device)
        batch.edge_counts = torch.tensor(edge_counts, dtype=torch.long, device=device)
        
        # Run inference with automatic mixed precision
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                predictions = model(batch)
        
        inference_time = time.time() - inference_start
        logger.info(f"[BATCH] Batch inference completed in {inference_time:.3f}s")
        
        # Step 4: Extract LAST prediction per package (next unhappened event)
        logger.info("[BATCH] Extracting last predictions per package...")
        extraction_start = time.time()
        
        edge_offset = 0
        
        for graph_idx, (graph, pkg_idx) in enumerate(zip(device_graphs, valid_indices)):
            pkg_id = graph.pkg_id
            num_edges = edge_counts[graph_idx]
            
            if num_edges > 0:
                # Extract all predictions for this package
                pkg_predictions = predictions[edge_offset:edge_offset + num_edges]
                
                if pkg_predictions.dim() > 1:
                    preds = pkg_predictions.squeeze(-1)
                else:
                    preds = pkg_predictions
                
                preds_scaled = preds.float().cpu().numpy()
                preds_hours = preprocessor.inverse_transform_time(preds_scaled).flatten()
                
                # OPTIMIZATION: Only return the LAST prediction (next unhappened event)
                last_prediction = float(preds_hours[-1]) if len(preds_hours) > 0 else None
                
                # Update result with last prediction only
                results[pkg_idx] = {
                    'package_id': pkg_id,
                    'status': 'success',
                    'next_event_prediction_hours': last_prediction,
                    'total_transitions': num_edges,
                }
                
                # Add ground truth for the last transition if available (for debugging)
                features = features_list[pkg_idx]
                if features and 'labels_raw' in features and features['labels_raw'] is not None:
                    labels_raw = features['labels_raw'].flatten()
                    if len(labels_raw) > 0:
                        results[pkg_idx]['ground_truth_last_transition_hours'] = float(labels_raw[-1])
                
            else:
                results[pkg_idx] = {
                    'package_id': pkg_id,
                    'status': 'success',
                    'next_event_prediction_hours': None,
                    'total_transitions': 0,
                }
            
            edge_offset += num_edges
        
        extraction_time = time.time() - extraction_start
        total_time = time.time() - start_time
        
        logger.info(f"[BATCH] Result extraction completed in {extraction_time:.3f}s")
        logger.info(f"[BATCH] Total batch processing time: {total_time:.3f}s")
        logger.info(f"[BATCH] Performance breakdown:")
        logger.info(f"  - Preprocessing: {preprocessing_time:.3f}s ({preprocessing_time/total_time*100:.1f}%)")
        logger.info(f"  - Graph conversion: {conversion_time:.3f}s ({conversion_time/total_time*100:.1f}%)")
        logger.info(f"  - Batch inference: {inference_time:.3f}s ({inference_time/total_time*100:.1f}%)")
        logger.info(f"  - Result extraction: {extraction_time:.3f}s ({extraction_time/total_time*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"[BATCH] Batch inference failed: {e}")
        import traceback
        logger.error(f"[BATCH] Traceback: {traceback.format_exc()}")
        
        # Update all pending results to error
        for i, result in enumerate(results):
            if result.get('status') == 'pending':
                results[i] = {
                    'package_id': result['package_id'],
                    'status': 'error',
                    'error': f'Batch inference failed: {str(e)}'
                }
    
    success_count = sum(1 for r in results if r.get('status') == 'success')
    logger.info(f"[BATCH] Completed: {success_count}/{len(results)} successful")
    
    return {
        'status': 'success',
        'results': results,
        'batch_stats': {
            'total_packages': len(packages),
            'successful': success_count,
            'failed': len(results) - success_count,
            'workers_used': actual_workers,
            'max_workers_available': max_cpu_workers,
            'preprocessing_time': preprocessing_time,
            'inference_time': inference_time if 'inference_time' in locals() else 0,
            'total_time': time.time() - start_time,
            'optimization': 'last_prediction_only'
        }
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
