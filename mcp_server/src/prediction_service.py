"""
Prediction service - orchestrates Neptune data + SageMaker inference.

Key behaviors:
- Validates package_id format (only TBA packages supported for Amazon network)
- Finds INDUCT event by event_type (not by index)
- Filters out events before INDUCT from both Neptune events and result
- Re-indexes events starting from INDUCT = 0
- Predicts event times for ALL events except INDUCT
- Uses postal code as location for DELIVERY events
- Calculates AE (absolute error) for events that have happened
"""

import json
import logging
import boto3
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from neptune_client import NeptuneClient
from skeleton_builder import SkeletonBuilder

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PredictionService:
    """
    Orchestrates package predictions.
    
    Flow:
    1. Validate package_id format (must start with 'TBA')
    2. Fetch data from Neptune
    3. Find INDUCT event by event_type
    4. Filter Neptune events (remove pre-INDUCT events)
    5. Build skeleton starting from induction node
    6. Match events to skeleton
    7. Run predictions
    8. Return formatted results (starting from INDUCT, index 0)
    """
    
    def __init__(self, sagemaker_endpoint: str, neptune_endpoint: str):
        init_start = time.time()
        logger.info("=" * 60)
        logger.info("INITIALIZING PredictionService")
        logger.info("=" * 60)
        
        self.sagemaker_endpoint = sagemaker_endpoint
        logger.info(f"[INIT] SageMaker endpoint: {sagemaker_endpoint}")
        
        logger.info(f"[INIT] Creating SageMaker runtime client...")
        sagemaker_start = time.time()
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        logger.info(f"[INIT] SageMaker client created in {time.time() - sagemaker_start:.3f}s")
        
        logger.info(f"[INIT] Creating Neptune client for: {neptune_endpoint}")
        neptune_start = time.time()
        self.neptune = NeptuneClient(neptune_endpoint, allow_undelivered=True)
        logger.info(f"[INIT] Neptune client created in {time.time() - neptune_start:.3f}s")
        
        logger.info(f"[INIT] Creating SkeletonBuilder...")
        self.skeleton_builder = SkeletonBuilder()
        logger.info(f"[INIT] SkeletonBuilder created")
        
        total_init_time = time.time() - init_start
        logger.info(f"[INIT] PredictionService initialized in {total_init_time:.3f}s")
        logger.info("=" * 60)
    
    def _validate_package_id(self, package_id: str) -> Optional[Dict]:
        """
        Validate package_id format.
        
        Only TBA packages (Amazon network) are supported.
        
        Args:
            package_id: Package identifier to validate
            
        Returns:
            None if valid, error/skip dict if invalid
        """
        if not package_id:
            logger.warning("[VALIDATE] Empty package_id provided")
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'Package ID is required',
            }
        
        # Check if package_id starts with 'TBA'
        if not package_id.startswith('TBA'):
            logger.warning(f"[VALIDATE] Non-TBA package rejected: {package_id}")
            return {
                'package_id': package_id,
                'status': 'skipped',
                'warning': 'We only support Amazon network',
            }
        
        return None
    
    def _is_induct_event(self, event: Dict) -> bool:
        """Check if event is INDUCT type."""
        return event.get('event_type') == 'INDUCT'
    
    def _find_induct_event(self, events: List[Dict]) -> Optional[Tuple[int, Dict]]:
        """
        Find INDUCT event by event_type.
        
        Args:
            events: List of events
            
        Returns:
            Tuple of (index, event) or None if not found
        """
        for idx, event in enumerate(events):
            if self._is_induct_event(event):
                return (idx, event)
        return None
    
    def _invoke_sagemaker(self, packages: List[Dict]) -> Dict:
        """Call SageMaker endpoint with package data."""
        logger.info(f"[SAGEMAKER] Preparing to invoke endpoint: {self.sagemaker_endpoint}")
        logger.info(f"[SAGEMAKER] Number of packages: {len(packages)}")
        
        payload = {
            'action': 'predict',
            'packages': packages
        }
        
        payload_str = json.dumps(payload, default=str)
        logger.info(f"[SAGEMAKER] Payload size: {len(payload_str)} bytes")
        
        invoke_start = time.time()
        logger.info(f"[SAGEMAKER] Invoking endpoint...")
        
        try:
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.sagemaker_endpoint,
                ContentType='application/json',
                Body=payload_str
            )
            
            invoke_time = time.time() - invoke_start
            logger.info(f"[SAGEMAKER] Endpoint responded in {invoke_time:.3f}s")
            
            result = json.loads(response['Body'].read().decode())
            logger.info(f"[SAGEMAKER] Response status: {result.get('status')}")
            
            return result
            
        except Exception as e:
            logger.error(f"[SAGEMAKER] ERROR invoking endpoint: {type(e).__name__}: {e}")
            raise
    
    def _build_synthetic_package(
        self, 
        package_data: Dict, 
        events_with_times: List[Dict]
    ) -> Dict:
        """Build synthetic package for SageMaker inference."""
        synthetic = {
            'package_id': package_data.get('package_id'),
            'tracking_id': package_data.get('package_id'),
            'source_postal': package_data.get('source_postal'),
            'dest_postal': package_data.get('dest_postal'),
            'pdd': package_data.get('pdd'),
            'weight': package_data.get('weight', 0),
            'length': package_data.get('length', 0),
            'width': package_data.get('width', 0),
            'height': package_data.get('height', 0),
            'events': []
        }
        
        for evt in events_with_times:
            context = evt.get('context', {})
            event_type = evt.get('event_type')
            location = evt.get('location')
            
            event = {
                'event_type': event_type,
                'event_time': evt.get('event_time'),
                'sort_center': location if event_type != 'DELIVERY' else None,
                'delivery_station': location if event_type == 'DELIVERY' else None,
                'delivery_location': {'id': package_data.get('dest_postal')} if event_type == 'DELIVERY' else None,
                'carrier_id': context.get('carrier', 'AMZN_US'),
                'leg_type': context.get('leg_type', 'FORWARD'),
                'ship_method': context.get('ship_method') or evt.get('ship_method'),
                'plan_time': evt.get('plan_time'),
                'cpt': evt.get('cpt') or context.get('cpt'),
                'dwelling_seconds': context.get('dwelling_seconds', 0),
                'missort': context.get('missort', False),
                'problem': context.get('problem'),
            }
            synthetic['events'].append(event)
        
        return synthetic
    
    def _parse_event_time(self, event_time) -> Optional[datetime]:
        """Parse event time string to datetime."""
        if event_time is None:
            return None
        if isinstance(event_time, datetime):
            return event_time
        if isinstance(event_time, str):
            formats = [
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(event_time, fmt)
                except ValueError:
                    continue
            
            try:
                from dateutil import parser
                return parser.parse(event_time)
            except:
                pass
        return None
    
    def _format_event_time(self, event_time) -> Optional[str]:
        """Format event time to ISO string."""
        if event_time is None:
            return None
        if isinstance(event_time, str):
            dt = self._parse_event_time(event_time)
            if dt:
                return dt.isoformat()
            return event_time
        if isinstance(event_time, datetime):
            return event_time.isoformat()
        return None
    
    def _calculate_predicted_datetime(
        self, 
        prev_event_time, 
        predicted_hours: float
    ) -> Optional[str]:
        """Calculate predicted datetime from previous event time and predicted hours."""
        prev_dt = self._parse_event_time(prev_event_time)
        if prev_dt is None or predicted_hours is None:
            return None
        predicted_dt = prev_dt + timedelta(hours=predicted_hours)
        return predicted_dt.isoformat()
    
    def _calculate_ae(
        self,
        predicted_time: Optional[str],
        actual_time: Optional[str]
    ) -> Optional[float]:
        """Calculate Absolute Error (AE) in hours."""
        if predicted_time is None or actual_time is None:
            return None
        
        pred_dt = self._parse_event_time(predicted_time)
        actual_dt = self._parse_event_time(actual_time)
        
        if pred_dt is None or actual_dt is None:
            return None
        
        diff_seconds = abs((pred_dt - actual_dt).total_seconds())
        diff_hours = diff_seconds / 3600.0
        
        return round(diff_hours, 4)
    
    def _run_predictions_for_all_events(
        self, 
        package_data: Dict, 
        skeleton: List[Dict]
    ) -> Dict[int, str]:
        """
        Run predictions for ALL events except INDUCT.
        
        Args:
            package_data: Package metadata
            skeleton: Skeleton starting from INDUCT (index 0 must be INDUCT)
        
        Returns:
            Dict mapping event_idx -> predicted_time
        """
        iter_start = time.time()
        logger.info("[PREDICT_ALL] Starting predictions for all events")
        
        predicted_times = {}
        
        if len(skeleton) < 1:
            logger.warning("[PREDICT_ALL] Empty skeleton")
            return predicted_times
        
        # Validate first event is INDUCT
        induct_event = skeleton[0]
        if not self._is_induct_event(induct_event):
            logger.error(f"[PREDICT_ALL] First event is not INDUCT: {induct_event.get('event_type')}")
            return predicted_times
        
        # INDUCT must have happened
        if not induct_event.get('neptune_matched', False):
            logger.error("[PREDICT_ALL] INDUCT event has not happened yet")
            return predicted_times
        
        logger.info(f"[PREDICT_ALL] Skeleton has {len(skeleton)} events (starting from INDUCT at idx 0)")
        
        # Start prediction chain with INDUCT event
        events_for_prediction = [dict(induct_event)]
        
        # Predict for every event after INDUCT (index 1 onwards)
        for target_idx in range(1, len(skeleton)):
            target_skel = skeleton[target_idx]
            
            prev_event_time = events_for_prediction[-1].get('event_time')
            if not prev_event_time:
                logger.warning(f"[PREDICT_ALL] No prev_event_time for event {target_idx}")
                break
            
            target_event = dict(target_skel)
            target_event['event_time'] = prev_event_time
            
            inference_events = events_for_prediction + [target_event]
            synthetic = self._build_synthetic_package(package_data, inference_events)
            
            try:
                result = self._invoke_sagemaker([synthetic])
                
                if result.get('status') == 'success' and result.get('results'):
                    pkg_result = result['results'][0]
                    
                    if pkg_result.get('status') == 'success':
                        preds = pkg_result.get('predictions_hours', [])
                        
                        if preds:
                            pred_hours = float(preds[-1])
                            predicted_time = self._calculate_predicted_datetime(prev_event_time, pred_hours)
                            
                            if predicted_time:
                                predicted_times[target_idx] = predicted_time
                                logger.debug(f"[PREDICT_ALL] Event {target_idx} ({target_skel['event_type']}): {predicted_time}")
                            
                            next_event = dict(target_skel)
                            if target_skel.get('neptune_matched', False):
                                next_event['event_time'] = target_skel.get('event_time')
                            else:
                                next_event['event_time'] = predicted_time
                            
                            events_for_prediction.append(next_event)
                            continue
                
                logger.warning(f"[PREDICT_ALL] Failed prediction for event {target_idx}")
                
                if target_skel.get('neptune_matched', False):
                    next_event = dict(target_skel)
                    events_for_prediction.append(next_event)
                else:
                    break
                    
            except Exception as e:
                logger.error(f"[PREDICT_ALL] Error for event {target_idx}: {e}")
                
                if target_skel.get('neptune_matched', False):
                    next_event = dict(target_skel)
                    events_for_prediction.append(next_event)
                else:
                    break
        
        logger.info(f"[PREDICT_ALL] Completed in {time.time() - iter_start:.3f}s, {len(predicted_times)} predictions")
        
        return predicted_times
    
    def _build_output_events(
        self,
        skeleton: List[Dict],
        predicted_times: Dict[int, str],
        dest_postal: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Build final output events.
        
        Args:
            skeleton: Skeleton starting from INDUCT (already filtered and re-indexed)
            predicted_times: Dict mapping event index -> predicted_time
            dest_postal: Destination postal code
        
        Returns:
            Tuple of (output_events, warnings)
        """
        output_events = []
        warnings = []
        
        for i, skel in enumerate(skeleton):
            neptune_matched = skel.get('neptune_matched', False)
            is_predicted = not neptune_matched
            event_type = skel.get('event_type')
            
            actual_event_time = None
            if neptune_matched:
                actual_event_time = skel.get('event_time')
            
            # No prediction for INDUCT
            predicted_time = None
            if not self._is_induct_event(skel):
                predicted_time = predicted_times.get(i)
            
            # DELIVERY uses dest_postal as location
            if event_type == 'DELIVERY':
                location = dest_postal
            else:
                location = skel.get('location')
            
            event_output = {
                'is_predicted': is_predicted,
                'event_idx': i,  # Already re-indexed, starts from 0
                'event_type': event_type,
                'location': location,
                'plan_time': self._format_event_time(skel.get('plan_time')),
                'event_time': self._format_event_time(actual_event_time),
                'predicted_time': self._format_event_time(predicted_time),
                'context': skel.get('context', {}),
            }
            
            # Calculate AE if event happened and we have prediction
            if neptune_matched and actual_event_time and predicted_time:
                ae = self._calculate_ae(predicted_time, actual_event_time)
                if ae is not None:
                    event_output['ae'] = ae
            
            # Check for delivery station mismatch
            if skel.get('delivery_station_mismatch'):
                expected_station = skel.get('expected_delivery_station')
                actual_station = skel.get('actual_delivery_station')
                
                warning = {
                    'type': 'DELIVERY_STATION_MISMATCH',
                    'message': f"Package delivered from different station than planned. Expected: {expected_station}, Actual: {actual_station}",
                    'expected_station': expected_station,
                    'actual_station': actual_station
                }
                event_output['warning'] = warning
                warnings.append({
                    'event_idx': i,
                    **warning
                })
                
                event_output['context']['expected_delivery_station'] = expected_station
                event_output['context']['actual_delivery_station'] = actual_station
            
            output_events.append(event_output)
        
        return output_events, warnings
        
    def _calculate_metrics(self, output_events: List[Dict]) -> Dict:
        """Calculate aggregate metrics from output events."""
        all_ae = []
        
        for evt in output_events:
            if evt.get('ae') is not None:
                all_ae.append(evt['ae'])
        
        metrics = {}
        
        if all_ae:
            metrics['mae'] = round(sum(all_ae) / len(all_ae), 4)
            metrics['max_ae'] = round(max(all_ae), 4)
            metrics['min_ae'] = round(min(all_ae), 4)
            metrics['total_ae'] = round(sum(all_ae), 4)
            metrics['num_events_with_ae'] = len(all_ae)
        
        return metrics
    
    def predict_single(self, package_id: str) -> Dict:
        """
        Predict event times for a single package.
        
        Returns events starting from INDUCT (index 0).
        Events before INDUCT are filtered out.
        Only TBA packages (Amazon network) are supported.
        """
        total_start = time.time()
        logger.info(f"[PREDICT] Starting prediction for package: {package_id}")
        
        # Step 0: Validate package_id
        validation_error = self._validate_package_id(package_id)
        if validation_error:
            logger.warning(f"[PREDICT] Package validation failed: {validation_error.get('warning') or validation_error.get('error')}")
            return validation_error
        
        # Step 1: Fetch from Neptune
        package_data = self.neptune.fetch_package(package_id)
        
        if package_data is None:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'Package not found or invalid'
            }
        
        neptune_events = package_data.get('events', [])
        logger.info(f"[PREDICT] Fetched {len(neptune_events)} Neptune events")
        
        # Step 2: Find INDUCT in Neptune events
        induct_result = self.skeleton_builder.find_induction_from_neptune_events(neptune_events)
        
        if induct_result is None:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'No INDUCT event found in Neptune events'
            }
        
        induct_neptune_idx, induction_node, induct_event = induct_result
        logger.info(f"[PREDICT] Found INDUCT at Neptune index {induct_neptune_idx}, node: {induction_node}")
        
        # Step 3: Filter Neptune events - remove events before INDUCT
        filtered_neptune_events, num_removed = self.skeleton_builder.filter_neptune_events_from_induct(neptune_events)
        
        if num_removed > 0:
            logger.info(f"[PREDICT] Removed {num_removed} events before INDUCT from Neptune events")
        
        # Step 4: Parse leg plan
        leg_plan_str = package_data.get('leg_plan')
        leg_plan = self.skeleton_builder.parse_leg_plan(leg_plan_str)
        
        if not leg_plan:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'No leg_plan available'
            }
        
        # Step 5: Create skeleton starting from induction node
        dest_postal = package_data.get('dest_postal')
        skeleton = self.skeleton_builder.create_skeleton(
            leg_plan, 
            dest_postal,
            induction_node=induction_node  # Pass induction node from Neptune
        )
        
        if not skeleton:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'Failed to create skeleton from leg_plan'
            }
        
        # Validate skeleton starts with INDUCT
        if not skeleton or skeleton[0].get('event_type') != 'INDUCT':
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'Skeleton does not start with INDUCT event'
            }
        
        logger.info(f"[PREDICT] Created skeleton with {len(skeleton)} events starting from INDUCT at {induction_node}")
        
        # Step 6: Match filtered Neptune events to skeleton
        filled_skeleton = self.skeleton_builder.match_events_to_skeleton(
            skeleton, filtered_neptune_events
        )
        
        # Step 7: Validate INDUCT event has happened
        induct_event_skeleton = filled_skeleton[0]
        if not induct_event_skeleton.get('neptune_matched', False):
            delivery_status = self.skeleton_builder.determine_delivery_status(filled_skeleton)
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'INDUCT event has not happened yet - cannot start predictions',
                'delivery_status': delivery_status,
            }
        
        # Step 8: Determine delivery status
        delivery_status = self.skeleton_builder.determine_delivery_status(filled_skeleton)
        
        # Count events
        actual_count = sum(1 for s in filled_skeleton if s.get('neptune_matched', False))
        predicted_count = len(filled_skeleton) - actual_count
        
        # Step 9: Run predictions for ALL events (except INDUCT)
        predicted_times = self._run_predictions_for_all_events(package_data, filled_skeleton)
        
        # Step 10: Build output events
        output_events, warnings = self._build_output_events(filled_skeleton, predicted_times, dest_postal)
        
        # Step 11: Calculate ETA and remaining time
        eta = None
        remaining_hours = None
        
        if output_events:
            # Find DELIVERY event by type
            delivery_event = None
            for evt in output_events:
                if evt.get('event_type') == 'DELIVERY':
                    delivery_event = evt
                    break
            
            if delivery_event:
                if delivery_event.get('is_predicted'):
                    eta = delivery_event.get('predicted_time')
                    
                    last_actual_time = None
                    for evt in reversed(output_events):
                        if not evt.get('is_predicted') and evt.get('event_time'):
                            last_actual_time = evt.get('event_time')
                            break
                    
                    if last_actual_time and eta:
                        last_dt = self._parse_event_time(last_actual_time)
                        eta_dt = self._parse_event_time(eta)
                        if last_dt and eta_dt:
                            remaining_hours = round((eta_dt - last_dt).total_seconds() / 3600.0, 4)
                else:
                    eta = delivery_event.get('event_time')
        
        # Get last known location for in-transit packages
        last_known_location = None
        if delivery_status == 'IN_TRANSIT':
            last_known_location = self.skeleton_builder.get_last_known_location(filled_skeleton)
        
        # Get induction info
        induction_info = self.skeleton_builder.get_induction_info(filled_skeleton)
        
        # Calculate aggregate metrics
        metrics = self._calculate_metrics(output_events)
        
        # Build result
        result = {
            'package_id': package_id,
            'status': 'success',
            'delivery_status': delivery_status,
            'num_events': len(output_events),
            'actual_events': actual_count,
            'predicted_events': predicted_count,
            'source_postal': package_data.get('source_postal'),
            'dest_postal': dest_postal,
            'induction_node': induction_node,
            'pdd': self._format_event_time(package_data.get('pdd')),
            'eta': eta,
            'remaining_hours': remaining_hours,
            'last_known_location': last_known_location,
            'metrics': metrics,
            'events': output_events,
        }
        
        # Add info about removed events
        if num_removed > 0:
            result['events_removed_before_induct'] = num_removed
        
        if warnings:
            result['warnings'] = warnings
        
        logger.info(f"[PREDICT] Completed for {package_id} in {time.time() - total_start:.3f}s")
        if warnings:
            logger.info(f"[PREDICT] {len(warnings)} warning(s) generated")
        
        return result
    
    def predict_batch(self, package_ids: List[str]) -> Dict:
        """Predict for multiple packages."""
        batch_start = time.time()
        logger.info(f"[BATCH] Starting batch prediction for {len(package_ids)} packages")
        
        results = []
        total_warnings = 0
        skipped_count = 0
        
        for i, pkg_id in enumerate(package_ids):
            logger.info(f"[BATCH] Processing package {i + 1}/{len(package_ids)}: {pkg_id}")
            
            try:
                result = self.predict_single(pkg_id)
                results.append(result)
                
                # Track skipped packages (non-TBA)
                if result.get('status') == 'skipped':
                    skipped_count += 1
                
                if result.get('warnings'):
                    total_warnings += len(result['warnings'])
            except Exception as e:
                logger.error(f"[BATCH] Error predicting {pkg_id}: {e}")
                results.append({
                    'package_id': pkg_id,
                    'status': 'error',
                    'error': str(e)
                })
        
        success_count = sum(1 for r in results if r.get('status') == 'success')
        failed_count = sum(1 for r in results if r.get('status') == 'error')
        
        logger.info(f"[BATCH] Completed in {time.time() - batch_start:.3f}s")
        logger.info(f"[BATCH] Success: {success_count}, Failed: {failed_count}, Skipped: {skipped_count}, Warnings: {total_warnings}")
        
        batch_result = {
            'status': 'success',
            'total': len(results),
            'successful': success_count,
            'failed': failed_count,
            'skipped': skipped_count,
            'total_warnings': total_warnings,
            'predictions': results
        }
        
        return batch_result
    
    def get_package_status(self, package_id: str) -> Dict:
        """Get package status without full prediction."""
        
        # Validate package_id
        validation_error = self._validate_package_id(package_id)
        if validation_error:
            logger.warning(f"[STATUS] Package validation failed: {validation_error.get('warning') or validation_error.get('error')}")
            return validation_error
        
        package_data = self.neptune.fetch_package(package_id)
        
        if package_data is None:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'Package not found'
            }
        
        neptune_events = package_data.get('events', [])
        
        # Find INDUCT and filter
        induct_result = self.skeleton_builder.find_induction_from_neptune_events(neptune_events)
        
        if induct_result is None:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'No INDUCT event found'
            }
        
        _, induction_node, _ = induct_result
        filtered_neptune_events, _ = self.skeleton_builder.filter_neptune_events_from_induct(neptune_events)
        
        leg_plan_str = package_data.get('leg_plan')
        leg_plan = self.skeleton_builder.parse_leg_plan(leg_plan_str)
        
        if not leg_plan:
            return {
                'package_id': package_id,
                'status': 'error',
                'error': 'No leg_plan available'
            }
        
        dest_postal = package_data.get('dest_postal')
        skeleton = self.skeleton_builder.create_skeleton(
            leg_plan, 
            dest_postal,
            induction_node=induction_node
        )
        filled_skeleton = self.skeleton_builder.match_events_to_skeleton(
            skeleton, filtered_neptune_events
        )
        
        delivery_status = self.skeleton_builder.determine_delivery_status(filled_skeleton)
        last_known_location = self.skeleton_builder.get_last_known_location(filled_skeleton)
        
        completed = sum(1 for s in filled_skeleton if s.get('neptune_matched', False))
        
        # Check for warnings
        warnings = []
        for i, skel in enumerate(filled_skeleton):
            if skel.get('delivery_station_mismatch'):
                warnings.append({
                    'event_idx': i,
                    'type': 'DELIVERY_STATION_MISMATCH',
                    'message': f"Expected: {skel.get('expected_delivery_station')}, Actual: {skel.get('actual_delivery_station')}",
                    'expected_station': skel.get('expected_delivery_station'),
                    'actual_station': skel.get('actual_delivery_station')
                })
        
        result = {
            'package_id': package_id,
            'status': 'success',
            'delivery_status': delivery_status,
            'last_known_location': last_known_location,
            'source_postal': package_data.get('source_postal'),
            'dest_postal': dest_postal,
            'induction_node': induction_node,
            'pdd': self._format_event_time(package_data.get('pdd')),
            'total_events': len(filled_skeleton),
            'completed_events': completed,
        }
        
        if warnings:
            result['warnings'] = warnings
        
        return result
    
    def close(self):
        """Clean up resources."""
        logger.info("[CLEANUP] Closing PredictionService resources...")
        if self.neptune:
            self.neptune.close()
        logger.info("[CLEANUP] Done")