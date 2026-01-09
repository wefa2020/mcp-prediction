"""
Skeleton builder - creates event skeleton from leg_plan.

Key behaviors:
- Induction node is determined by INDUCT event_type, not by index
- Events before INDUCT are filtered out
- Event indices are re-numbered starting from INDUCT = 0
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SkeletonBuilder:
    """Builds event skeleton from leg_plan and matches with Neptune events."""
    
    def __init__(self):
        pass
    
    def parse_leg_plan(self, leg_plan_str: str) -> Optional[Dict]:
        """Parse leg plan JSON string."""
        if not leg_plan_str:
            logger.warning("[SKELETON] leg_plan_str is empty or None")
            return None
        try:
            leg_plan = json.loads(leg_plan_str)
            if not isinstance(leg_plan, dict) or len(leg_plan) == 0:
                logger.warning("[SKELETON] leg_plan is not a dict or is empty")
                return None
            logger.info(f"[SKELETON] Parsed leg_plan with {len(leg_plan)} entries")
            return leg_plan
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.error(f"[SKELETON] Failed to parse leg_plan: {e}")
            return None
    
    def _is_origin_id(self, location_id: str) -> bool:
        """Check if location ID is an origin (long alphanumeric)."""
        return len(location_id) > 20 and location_id.isalnum()
    
    def _is_postal_code(self, location_id: str) -> bool:
        """Check if location ID is a postal code (5 digits)."""
        return bool(re.match(r'^\d{5}$', location_id))
    
    def _is_sort_center(self, location_id: str) -> bool:
        """Check if location ID is a sort center (3-4 alphanumeric)."""
        return bool(re.match(r'^[A-Z0-9]{3,4}$', location_id))
    
    def find_induction_from_neptune_events(
        self, 
        neptune_events: List[Dict]
    ) -> Optional[Tuple[int, str, Dict]]:
        """
        Find INDUCT event from Neptune events by event_type.
        
        Args:
            neptune_events: List of events from Neptune
            
        Returns:
            Tuple of (index, location, event) or None if not found
        """
        for idx, event in enumerate(neptune_events):
            if event.get('event_type') == 'INDUCT':
                location = event.get('sort_center') or event.get('location')
                logger.info(f"[SKELETON] Found INDUCT event at Neptune index {idx}, location: {location}")
                return (idx, location, event)
        
        logger.warning("[SKELETON] No INDUCT event found in Neptune events")
        return None
    
    def find_induction_node_from_leg_plan(self, leg_plan: Dict) -> Optional[str]:
        """
        Find induction node from leg_plan.
        Returns the first sort center in leg_plan.
        
        Args:
            leg_plan: Parsed leg plan dictionary
            
        Returns:
            Sort center ID or None
        """
        for loc_id in leg_plan.keys():
            if self._is_sort_center(loc_id):
                logger.info(f"[SKELETON] First sort center in leg_plan: {loc_id}")
                return loc_id
        return None
    
    def create_skeleton(
        self, 
        leg_plan: Dict, 
        dest_postal: str,
        induction_node: Optional[str] = None
    ) -> List[Dict]:
        """
        Create event skeleton from leg_plan.
        
        Args:
            leg_plan: Parsed leg plan dictionary
            dest_postal: Destination postal code
            induction_node: Optional - if provided, this node will be used as INDUCT location.
                           If not provided, first sort center in leg_plan is used.
        
        Returns:
            List of skeleton events starting with INDUCT
        """
        logger.info("[SKELETON] Creating skeleton from leg_plan")
        logger.info(f"[SKELETON] leg_plan keys: {list(leg_plan.keys())}")
        logger.info(f"[SKELETON] dest_postal: {dest_postal}")
        logger.info(f"[SKELETON] induction_node override: {induction_node}")
        
        skeleton = []
        location_ids = list(leg_plan.keys())
        
        origin_id = None
        sort_centers = []
        dest_postal_entry = None
        
        for loc_id in location_ids:
            if self._is_origin_id(loc_id):
                origin_id = loc_id
                logger.debug(f"[SKELETON] Found origin: {loc_id[:20]}...")
            elif self._is_postal_code(loc_id):
                dest_postal_entry = (loc_id, leg_plan[loc_id])
                logger.debug(f"[SKELETON] Found postal code: {loc_id}")
            elif self._is_sort_center(loc_id):
                sort_centers.append((loc_id, leg_plan[loc_id]))
                logger.debug(f"[SKELETON] Found sort center: {loc_id}")
            else:
                logger.debug(f"[SKELETON] Unknown location type: {loc_id}")
        
        logger.info(f"[SKELETON] Found {len(sort_centers)} sort centers")
        
        if not sort_centers:
            logger.error("[SKELETON] No sort centers found in leg_plan")
            return []
        
        # If induction_node is provided, reorder sort_centers to start from it
        if induction_node:
            induction_idx = None
            for i, (sc_id, _) in enumerate(sort_centers):
                if sc_id == induction_node:
                    induction_idx = i
                    break
            
            if induction_idx is not None and induction_idx > 0:
                logger.info(f"[SKELETON] Reordering sort_centers to start from {induction_node} (was at index {induction_idx})")
                # Remove sort centers before induction node
                sort_centers = sort_centers[induction_idx:]
            elif induction_idx is None:
                logger.warning(f"[SKELETON] Induction node {induction_node} not found in leg_plan sort centers")
        
        event_idx = 0
        
        for i, (sc_id, sc_data) in enumerate(sort_centers):
            is_first_sc = (i == 0)
            
            if is_first_sc:
                # INDUCT event
                skeleton.append({
                    'event_idx': event_idx,
                    'event_type': 'INDUCT',
                    'location': sc_id,
                    'plan_time': sc_data.get('plan_time'),
                    'cpt': sc_data.get('cpt'),
                    'ship_method': sc_data.get('ship_method'),
                    'is_first_event': True,
                })
                logger.debug(f"[SKELETON] Added INDUCT at {sc_id} (idx={event_idx})")
                event_idx += 1
                
                # EXIT event for first sort center
                skeleton.append({
                    'event_idx': event_idx,
                    'event_type': 'EXIT',
                    'location': sc_id,
                    'plan_time': sc_data.get('cpt'),
                    'cpt': None,
                    'ship_method': None,
                    'is_first_event': False,
                })
                logger.debug(f"[SKELETON] Added EXIT at {sc_id} (idx={event_idx})")
                event_idx += 1
            else:
                # LINEHAUL event
                skeleton.append({
                    'event_idx': event_idx,
                    'event_type': 'LINEHAUL',
                    'location': sc_id,
                    'plan_time': sc_data.get('plan_time'),
                    'cpt': sc_data.get('cpt'),
                    'ship_method': sc_data.get('ship_method'),
                    'is_first_event': False,
                })
                logger.debug(f"[SKELETON] Added LINEHAUL at {sc_id} (idx={event_idx})")
                event_idx += 1
                
                # EXIT event
                skeleton.append({
                    'event_idx': event_idx,
                    'event_type': 'EXIT',
                    'location': sc_id,
                    'plan_time': sc_data.get('cpt'),
                    'cpt': None,
                    'ship_method': None,
                    'is_first_event': False,
                })
                logger.debug(f"[SKELETON] Added EXIT at {sc_id} (idx={event_idx})")
                event_idx += 1
        
        # DELIVERY event
        if sort_centers:
            last_sc_id = sort_centers[-1][0]
            delivery_plan_time = None
            delivery_ship_method = None
            
            if dest_postal_entry:
                delivery_plan_time = dest_postal_entry[1].get('plan_time')
                delivery_ship_method = dest_postal_entry[1].get('ship_method')
            
            skeleton.append({
                'event_idx': event_idx,
                'event_type': 'DELIVERY',
                'location': last_sc_id,
                'expected_delivery_station': last_sc_id,
                'plan_time': delivery_plan_time,
                'cpt': None,
                'ship_method': delivery_ship_method,
                'is_first_event': False,
            })
            logger.debug(f"[SKELETON] Added DELIVERY at {last_sc_id} (idx={event_idx})")
        
        logger.info(f"[SKELETON] Created skeleton with {len(skeleton)} events")
        return skeleton
    
    def filter_neptune_events_from_induct(
        self, 
        neptune_events: List[Dict]
    ) -> Tuple[List[Dict], int]:
        """
        Filter Neptune events to start from INDUCT event.
        Events before INDUCT are removed.
        
        Args:
            neptune_events: List of Neptune events
            
        Returns:
            Tuple of (filtered_events, num_removed)
        """
        induct_result = self.find_induction_from_neptune_events(neptune_events)
        
        if induct_result is None:
            logger.warning("[SKELETON] No INDUCT found, returning all events")
            return neptune_events, 0
        
        induct_idx, _, _ = induct_result
        
        if induct_idx == 0:
            logger.info("[SKELETON] INDUCT is first event, no filtering needed")
            return neptune_events, 0
        
        # Remove events before INDUCT
        removed_events = neptune_events[:induct_idx]
        filtered_events = neptune_events[induct_idx:]
        
        logger.info(f"[SKELETON] Removed {len(removed_events)} events before INDUCT:")
        for evt in removed_events:
            logger.debug(f"[SKELETON]   Removed: {evt.get('event_type')} at {evt.get('sort_center')}")
        
        return filtered_events, len(removed_events)
    
    def _get_event_location(self, event: Dict) -> str:
        """Extract location from Neptune event."""
        event_type = event.get('event_type', '')
        
        if event_type == 'DELIVERY':
            station = event.get('delivery_station')
            if station:
                return str(station)
            delivery_loc = event.get('delivery_location')
            if delivery_loc and isinstance(delivery_loc, dict):
                loc_id = delivery_loc.get('id', '')
                if loc_id:
                    return str(loc_id)
            return ''
        else:
            sort_center = event.get('sort_center')
            if sort_center:
                return str(sort_center)
            return ''
    
    def _get_plan_time_for_event(
        self, 
        event_idx: int, 
        event_type: str,
        skeleton: List[Dict],
        neptune_event: Optional[Dict]
    ) -> Optional[str]:
        """Get plan_time for an event."""
        if neptune_event:
            neptune_plan_time = neptune_event.get('plan_time')
            if neptune_plan_time:
                return neptune_plan_time
        
        if event_type == 'EXIT' and event_idx > 0:
            prev_skel = skeleton[event_idx - 1]
            prev_cpt = prev_skel.get('cpt')
            if prev_cpt:
                return prev_cpt
            prev_context = prev_skel.get('context', {})
            if prev_context.get('cpt'):
                return prev_context.get('cpt')
        
        skel = skeleton[event_idx]
        return skel.get('plan_time')
    
    def _find_delivery_event(
        self, 
        neptune_events: List[Dict], 
        matched_indices: set
    ) -> Optional[Tuple[int, Dict]]:
        """Find a DELIVERY event from Neptune events."""
        for idx, event in enumerate(neptune_events):
            if idx not in matched_indices and event.get('event_type') == 'DELIVERY':
                return (idx, event)
        return None
    
    def match_events_to_skeleton(
        self, 
        skeleton: List[Dict], 
        neptune_events: List[Dict]
    ) -> List[Dict]:
        """
        Match Neptune events to skeleton.
        
        Note: Neptune events should already be filtered to start from INDUCT.
        """
        logger.info("=" * 60)
        logger.info("[MATCH] Starting event matching")
        logger.info("=" * 60)
        logger.info(f"[MATCH] Skeleton events: {len(skeleton)}")
        logger.info(f"[MATCH] Neptune events: {len(neptune_events)}")
        
        # Validate skeleton starts with INDUCT
        if skeleton and skeleton[0].get('event_type') != 'INDUCT':
            logger.error(f"[MATCH] Skeleton does not start with INDUCT! First event: {skeleton[0].get('event_type')}")
        
        logger.info("[MATCH] SKELETON EVENTS:")
        for skel in skeleton:
            logger.info(f"[MATCH]   idx={skel['event_idx']}: {skel['event_type']} at '{skel['location']}'")
        
        logger.info("[MATCH] NEPTUNE EVENTS:")
        for i, evt in enumerate(neptune_events):
            loc = self._get_event_location(evt)
            evt_type = evt.get('event_type', 'UNKNOWN')
            logger.info(f"[MATCH]   {i}: {evt_type} at '{loc}'")
        
        filled_skeleton = []
        for skel in skeleton:
            filled_skeleton.append(dict(skel))
        
        # Build index of Neptune events
        neptune_index = {}
        for i, event in enumerate(neptune_events):
            loc = self._get_event_location(event)
            event_type = event.get('event_type', '')
            key = (loc, event_type)
            if key not in neptune_index:
                neptune_index[key] = []
            neptune_index[key].append((i, event))
        
        logger.info(f"[MATCH] Neptune index keys: {list(neptune_index.keys())}")
        
        matched_neptune_indices = set()
        
        for skel_idx, skel in enumerate(filled_skeleton):
            skel_loc = skel['location']
            skel_type = skel['event_type']
            
            matched_event = None
            delivery_station_mismatch = False
            actual_delivery_station = None
            
            if skel_type == 'DELIVERY':
                logger.info(f"[MATCH] Looking for skeleton[{skel_idx}]: DELIVERY (expected station: '{skel_loc}')")
                
                delivery_result = self._find_delivery_event(neptune_events, matched_neptune_indices)
                
                if delivery_result:
                    idx, matched_event = delivery_result
                    matched_neptune_indices.add(idx)
                    actual_delivery_station = self._get_event_location(matched_event)
                    
                    expected_station = skel.get('expected_delivery_station') or skel_loc
                    if actual_delivery_station and expected_station:
                        expected_normalized = str(expected_station).strip().upper()
                        actual_normalized = str(actual_delivery_station).strip().upper()
                        
                        if expected_normalized != actual_normalized:
                            delivery_station_mismatch = True
                            logger.warning(f"[MATCH]   ⚠ DELIVERY STATION MISMATCH: expected '{expected_station}', actual '{actual_delivery_station}'")
                        else:
                            logger.info(f"[MATCH]   ✓ MATCHED DELIVERY with Neptune event {idx}")
                else:
                    logger.warning(f"[MATCH]   ✗ NO DELIVERY event found")
            else:
                key = (skel_loc, skel_type)
                logger.info(f"[MATCH] Looking for skeleton[{skel_idx}]: {skel_type} at '{skel_loc}'")
                
                neptune_matches = neptune_index.get(key, [])
                
                for idx, event in neptune_matches:
                    if idx not in matched_neptune_indices:
                        matched_event = event
                        matched_neptune_indices.add(idx)
                        logger.info(f"[MATCH]   ✓ MATCHED with Neptune event {idx}")
                        break
                
                if not matched_event:
                    logger.warning(f"[MATCH]   ✗ NO MATCH for {skel_type} at '{skel_loc}'")
            
            if matched_event:
                skel['event_time'] = matched_event.get('event_time')
                skel['neptune_matched'] = True
                skel['is_predicted'] = False
                
                if skel_type == 'DELIVERY':
                    skel['actual_delivery_station'] = actual_delivery_station
                    if delivery_station_mismatch:
                        skel['delivery_station_mismatch'] = True
                        skel['location'] = actual_delivery_station
                
                plan_time = self._get_plan_time_for_event(
                    skel_idx, skel_type, filled_skeleton, matched_event
                )
                
                context = {'has_problem': False}
                
                problem = matched_event.get('problem')
                if problem:
                    context['problem'] = problem
                    context['has_problem'] = True
                
                missort = matched_event.get('missort')
                if missort is not None:
                    context['missort'] = bool(missort)
                
                dwelling_seconds = matched_event.get('dwelling_seconds')
                if dwelling_seconds and dwelling_seconds > 0:
                    context['dwelling_seconds'] = float(dwelling_seconds)
                    context['dwelling_hours'] = round(dwelling_seconds / 3600.0, 2)
                
                carrier = matched_event.get('carrier_id')
                if carrier:
                    context['carrier'] = carrier
                
                leg_type = matched_event.get('leg_type')
                if leg_type:
                    context['leg_type'] = leg_type
                
                ship_method = matched_event.get('ship_method') or skel.get('ship_method')
                if ship_method:
                    context['ship_method'] = ship_method
                
                sort_center = matched_event.get('sort_center')
                if sort_center:
                    context['sort_center'] = sort_center
                
                delivery_station = matched_event.get('delivery_station')
                if delivery_station:
                    context['delivery_station'] = delivery_station
                
                cpt = matched_event.get('cpt') or skel.get('cpt')
                if cpt:
                    context['cpt'] = cpt
                    skel['cpt'] = cpt
                
                skel['plan_time'] = plan_time
                skel['context'] = context
            else:
                skel['event_time'] = None
                skel['neptune_matched'] = False
                skel['is_predicted'] = True
                
                plan_time = self._get_plan_time_for_event(
                    skel_idx, skel_type, filled_skeleton, None
                )
                skel['plan_time'] = plan_time
                
                context = {'has_problem': False}
                if skel.get('ship_method'):
                    context['ship_method'] = skel['ship_method']
                if skel.get('cpt'):
                    context['cpt'] = skel['cpt']
                context['sort_center'] = skel['location']
                
                skel['context'] = context
        
        matched_count = sum(1 for s in filled_skeleton if s.get('neptune_matched', False))
        unmatched_count = len(filled_skeleton) - matched_count
        mismatch_count = sum(1 for s in filled_skeleton if s.get('delivery_station_mismatch', False))
        
        logger.info("=" * 60)
        logger.info(f"[MATCH] SUMMARY: {matched_count} matched, {unmatched_count} unmatched, {mismatch_count} mismatch(es)")
        logger.info("=" * 60)
        
        return filled_skeleton
    
    def reindex_events(self, events: List[Dict]) -> List[Dict]:
        """
        Re-index events starting from 0.
        
        Args:
            events: List of events (may have non-sequential indices)
            
        Returns:
            List of events with sequential indices starting from 0
        """
        reindexed = []
        for new_idx, event in enumerate(events):
            event_copy = dict(event)
            event_copy['event_idx'] = new_idx
            reindexed.append(event_copy)
        
        logger.info(f"[SKELETON] Re-indexed {len(reindexed)} events (0 to {len(reindexed) - 1})")
        return reindexed
    
    def determine_delivery_status(self, filled_skeleton: List[Dict]) -> str:
        """Determine delivery status from filled skeleton."""
        if not filled_skeleton:
            return 'UNKNOWN'
        
        # Find DELIVERY event by type, not by position
        delivery_event = None
        for event in filled_skeleton:
            if event.get('event_type') == 'DELIVERY':
                delivery_event = event
                break
        
        if delivery_event and delivery_event.get('neptune_matched', False):
            return 'DELIVERED'
        
        matched_count = sum(1 for s in filled_skeleton if s.get('neptune_matched', False))
        
        if matched_count == 0:
            return 'NOT_STARTED'
        else:
            return 'IN_TRANSIT'
    
    def get_last_known_location(self, filled_skeleton: List[Dict]) -> Optional[Dict]:
        """Get last known location from filled skeleton."""
        for skel in reversed(filled_skeleton):
            if skel.get('neptune_matched', False):
                return {
                    'event_idx': skel['event_idx'],
                    'event_type': skel['event_type'],
                    'location': skel['location'],
                    'event_time': skel.get('event_time')
                }
        return None
    
    def get_induction_info(self, filled_skeleton: List[Dict]) -> Optional[Dict]:
        """
        Get induction information from filled skeleton.
        
        Returns:
            Dict with induction_node, induction_time, event_idx or None
        """
        for event in filled_skeleton:
            if event.get('event_type') == 'INDUCT':
                return {
                    'induction_node': event.get('location'),
                    'induction_time': event.get('event_time'),
                    'event_idx': event.get('event_idx'),
                    'is_matched': event.get('neptune_matched', False)
                }
        return None