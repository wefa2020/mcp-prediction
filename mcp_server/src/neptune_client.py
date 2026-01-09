"""
Neptune client for fetching package data.
Self-contained - matches original neptune_extractor query structure.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Any
from gremlin_python.driver import client, serializer
from gremlin_python.driver.protocol import GremlinServerError

logger = logging.getLogger(__name__)


class NeptuneClient:
    """Client for Neptune database operations."""
    
    def __init__(self, endpoint: str, allow_undelivered: bool = True):
        """
        Initialize Neptune client.
        
        Args:
            endpoint: Neptune endpoint (host:port)
            allow_undelivered: If True, allow packages without DELIVERY event
        """
        self.endpoint = endpoint
        self.allow_undelivered = allow_undelivered
        self._client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neptune."""
        connect_start = time.time()
        logger.info(f"[NEPTUNE] Connecting to Neptune at {self.endpoint}")
        
        try:
            self._client = client.Client(
                f'wss://{self.endpoint}',
                'g',
                message_serializer=serializer.GraphSONSerializersV2d0()
            )
            
            # Test connection
            logger.debug("[NEPTUNE] Testing connection...")
            test_result = self._client.submit("g.V().limit(1).count()").all().result()
            logger.debug(f"[NEPTUNE] Connection test result: {test_result}")
            
            logger.info(f"[NEPTUNE] Connection established in {time.time() - connect_start:.3f}s")
            
        except Exception as e:
            logger.error(f"[NEPTUNE] Failed to connect: {type(e).__name__}: {e}")
            raise
    
    def _execute_query(self, query: str) -> List[Any]:
        """Execute a Gremlin query and return results."""
        try:
            result = self._client.submit(query).all().result()
            return result
        except GremlinServerError as e:
            logger.error(f"[NEPTUNE] Gremlin error: {e}")
            raise
        except Exception as e:
            logger.error(f"[NEPTUNE] Query error: {type(e).__name__}: {e}")
            raise
    
    def _sanitize_id(self, package_id: str) -> str:
        """Sanitize package ID for Gremlin query."""
        return package_id.replace("'", "\\'")
    
    def fetch_package(self, package_id: str) -> Optional[Dict]:
        """
        Fetch package data from Neptune using optimized single query.
        Matches original _extract_package_edges_optimized structure.
        """
        fetch_start = time.time()
        logger.info(f"[NEPTUNE] Fetching package: {package_id}")
        
        escaped_id = self._sanitize_id(package_id)
        
        try:
            # OPTIMIZED: Single query to get package properties AND all edges
            # This matches the original neptune_extractor query
            combined_query = f"""
            g.V()
             .has('Package', 'id', '{escaped_id}')
             .project('package', 'induct', 'exit', 'linehaul', 'problem', 'missort', 'delivery')
             .by(elementMap())
             .by(outE('Induct').project('edge', 'sc').by(elementMap()).by(inV().values('id')).fold())
             .by(outE('Exit202').project('edge', 'sc').by(elementMap()).by(inV().values('id')).fold())
             .by(outE('LineHaul').project('edge', 'sc').by(elementMap()).by(inV().values('id')).fold())
             .by(outE('Problem').project('edge', 'sc').by(elementMap()).by(inV().values('id')).fold())
             .by(outE('Missort').project('edge', 'sc').by(elementMap()).by(inV().values('id')).fold())
             .by(outE('Delivery').project('edge', 'node').by(elementMap()).by(inV().elementMap()).fold())
            """
            
            logger.debug(f"[NEPTUNE] Executing combined query...")
            query_start = time.time()
            results = self._execute_query(combined_query)
            logger.info(f"[NEPTUNE] Query completed in {time.time() - query_start:.3f}s")
            
            if not results:
                logger.warning(f"[NEPTUNE] Package not found: {package_id}")
                return None
            
            data = results[0]
            package_props = data['package']
            
            logger.debug(f"[NEPTUNE] Package properties: {list(package_props.keys())}")
            
            # Build package data structure
            package_data = {
                'package_id': package_id,
                'source_postal': package_props.get('source_postal_code'),
                'dest_postal': package_props.get('destination_postal_code'),
                'leg_plan': package_props.get('leg_plan'),
                'pdd': package_props.get('pdd'),
                'weight': package_props.get('weight', 0),
                'length': package_props.get('length', 0),
                'width': package_props.get('width', 0),
                'height': package_props.get('height', 0),
                'events': []
            }
            
            logger.debug(f"[NEPTUNE] source_postal: {package_data['source_postal']}")
            logger.debug(f"[NEPTUNE] dest_postal: {package_data['dest_postal']}")
            logger.debug(f"[NEPTUNE] leg_plan length: {len(package_data.get('leg_plan', '') or '')}")
            
            # Build lookup dictionaries for Problem and Missort
            problems_by_sc = defaultdict(list)
            missorts_by_sc = defaultdict(set)
            
            # Process Problem edges
            for edge_data in data.get('problem', []):
                edge = edge_data['edge']
                event_time = edge.get('event_time')
                if event_time:
                    problems_by_sc[edge_data['sc']].append({
                        'event_time': event_time,
                        'container_problems': edge.get('container_problems', '')
                    })
            
            for sc in problems_by_sc:
                problems_by_sc[sc].sort(key=lambda x: x['event_time'])
            
            # Process Missort edges
            for edge_data in data.get('missort', []):
                edge = edge_data['edge']
                event_time = edge.get('event_time')
                if event_time:
                    missorts_by_sc[edge_data['sc']].add(event_time)
            
            logger.debug(f"[NEPTUNE] Problems by SC: {dict(problems_by_sc)}")
            logger.debug(f"[NEPTUNE] Missorts by SC: {dict(missorts_by_sc)}")
            
            # Process Induct edges
            induct_count = len(data.get('induct', []))
            logger.info(f"[NEPTUNE] Processing {induct_count} INDUCT events")
            
            for edge_data in data.get('induct', []):
                edge = edge_data['edge']
                event_time = edge.get('event_time')
                
                if not event_time:
                    if not self.allow_undelivered:
                        logger.warning(f"[NEPTUNE] INDUCT event missing event_time")
                        return None
                    continue
                
                sort_center = edge_data['sc']
                has_missort = sort_center in missorts_by_sc and len(missorts_by_sc[sort_center]) > 0
                
                package_data['events'].append({
                    'event_type': 'INDUCT',
                    'sort_center': sort_center,
                    'event_time': event_time,
                    'plan_time': edge.get('plan_time'),
                    'cpt': edge.get('cpt'),
                    'leg_type': edge.get('leg_type'),
                    'carrier_id': edge.get('carrier_id'),
                    'load_id': edge.get('load_id'),
                    'ship_method': edge.get('ship_method'),
                    'missort': has_missort
                })
                logger.debug(f"[NEPTUNE]   INDUCT at {sort_center}: {event_time}")
            
            # Process Exit202 edges
            exit_count = len(data.get('exit', []))
            logger.info(f"[NEPTUNE] Processing {exit_count} EXIT events")
            
            for edge_data in data.get('exit', []):
                edge = edge_data['edge']
                event_time = edge.get('event_time')
                
                if not event_time:
                    if not self.allow_undelivered:
                        logger.warning(f"[NEPTUNE] EXIT event missing event_time")
                        return None
                    continue
                
                sort_center = edge_data['sc']
                
                # Find relevant problem
                problem_info = None
                if sort_center in problems_by_sc:
                    relevant_problems = [
                        p for p in problems_by_sc[sort_center]
                        if p['event_time'] <= event_time
                    ]
                    if relevant_problems:
                        problem_info = relevant_problems[-1]['container_problems']
                
                package_data['events'].append({
                    'event_type': 'EXIT',
                    'sort_center': sort_center,
                    'event_time': event_time,
                    'dwelling_seconds': edge.get('dwelling_seconds'),
                    'leg_type': edge.get('leg_type'),
                    'carrier_id': edge.get('carrier_id'),
                    'problem': problem_info
                })
                logger.debug(f"[NEPTUNE]   EXIT at {sort_center}: {event_time}")
            
            # Process LineHaul edges
            linehaul_count = len(data.get('linehaul', []))
            logger.info(f"[NEPTUNE] Processing {linehaul_count} LINEHAUL events")
            
            for edge_data in data.get('linehaul', []):
                edge = edge_data['edge']
                event_time = edge.get('event_time')
                
                if not event_time:
                    if not self.allow_undelivered:
                        logger.warning(f"[NEPTUNE] LINEHAUL event missing event_time")
                        return None
                    continue
                
                sort_center = edge_data['sc']
                has_missort = sort_center in missorts_by_sc and len(missorts_by_sc[sort_center]) > 0
                
                package_data['events'].append({
                    'event_type': 'LINEHAUL',
                    'sort_center': sort_center,
                    'event_time': event_time,
                    'plan_time': edge.get('plan_time'),
                    'cpt': edge.get('cpt'),
                    'leg_type': edge.get('leg_type'),
                    'carrier_id': edge.get('carrier_id'),
                    'ship_method': edge.get('ship_method'),
                    'missort': has_missort
                })
                logger.debug(f"[NEPTUNE]   LINEHAUL at {sort_center}: {event_time}")
            
            # Process Delivery edge
            delivery_count = len(data.get('delivery', []))
            logger.info(f"[NEPTUNE] Processing {delivery_count} DELIVERY events")
            
            for edge_data in data.get('delivery', []):
                edge = edge_data['edge']
                event_time = edge.get('event_time')
                
                if not event_time:
                    if not self.allow_undelivered:
                        logger.warning(f"[NEPTUNE] DELIVERY event missing event_time")
                        return None
                    continue
                
                delivery_node = edge_data.get('node', {})
                
                package_data['events'].append({
                    'event_type': 'DELIVERY',
                    'event_time': event_time,
                    'plan_time': edge.get('plan_time'),
                    'delivery_station': edge.get('delivery_station'),
                    'ship_method': edge.get('ship_method'),
                    'delivery_location': {
                        'id': delivery_node.get('id'),
                        'city': delivery_node.get('city'),
                        'county': delivery_node.get('county'),
                        'state': delivery_node.get('state_id'),
                        'lat': delivery_node.get('lat'),
                        'lng': delivery_node.get('lng')
                    }
                })
                logger.debug(f"[NEPTUNE]   DELIVERY: {event_time}")
            
            # Sort events by time
            package_data['events'].sort(key=lambda e: e.get('event_time', '') or '')
            
            # Deduplicate events
            package_data = self._deduplicate_events(package_data)
            
            # Validate if required
            if not self.allow_undelivered:
                is_valid, reason = self._validate_package_sequence(package_data)
                if not is_valid:
                    logger.warning(f"[NEPTUNE] Package invalid: {reason}")
                    return None
            
            total_time = time.time() - fetch_start
            logger.info(f"[NEPTUNE] Fetch completed in {total_time:.3f}s")
            logger.info(f"[NEPTUNE] Package {package_id}: {len(package_data['events'])} events")
            
            # Log event summary
            for i, evt in enumerate(package_data['events']):
                loc = evt.get('sort_center') or evt.get('delivery_station') or 'N/A'
                logger.info(f"[NEPTUNE]   Event {i}: {evt['event_type']} at {loc}")
            
            return package_data
            
        except GremlinServerError as e:
            logger.error(f"[NEPTUNE] Gremlin error fetching {package_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"[NEPTUNE] Error fetching {package_id}: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[NEPTUNE] Traceback: {traceback.format_exc()}")
            return None
    
    def _deduplicate_events(self, package_data: Dict) -> Dict:
        """Remove duplicate events."""
        events = package_data.get('events', [])
        seen = set()
        unique_events = []
        
        for event in events:
            location = event.get('sort_center') or event.get('delivery_station') or ''
            key = (
                event.get('event_type'),
                location,
                event.get('event_time')
            )
            
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
        
        dedup_count = len(events) - len(unique_events)
        if dedup_count > 0:
            logger.info(f"[NEPTUNE] Removed {dedup_count} duplicate events")
        
        package_data['events'] = unique_events
        return package_data
    
    def _validate_package_sequence(self, package_data: Dict) -> tuple:
        """Validate package has required events."""
        events = package_data.get('events', [])
        
        if not events:
            return False, "No events found"
        
        event_types = [e.get('event_type') for e in events]
        
        if 'INDUCT' not in event_types:
            return False, "Missing INDUCT event"
        
        if 'DELIVERY' not in event_types:
            return False, "Missing DELIVERY event"
        
        return True, None
    
    def close(self):
        """Close Neptune connection."""
        logger.info("[NEPTUNE] Closing connection...")
        if self._client:
            try:
                self._client.close()
                logger.info("[NEPTUNE] Connection closed")
            except Exception as e:
                logger.warning(f"[NEPTUNE] Error closing connection: {e}")
            self._client = None