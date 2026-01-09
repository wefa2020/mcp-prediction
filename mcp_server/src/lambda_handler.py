"""
MCP Server Lambda Handler
- Handles MCP protocol
- Connects to Neptune for package data
- Orchestrates predictions via SageMaker
"""

import json
import logging
import os
from typing import Dict, Any, Optional

from prediction_service import PredictionService

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize prediction service (lazy loaded)
_prediction_service: Optional[PredictionService] = None


def get_prediction_service() -> PredictionService:
    """Get or create prediction service singleton."""
    global _prediction_service
    
    if _prediction_service is None:
        logger.info("Initializing prediction service...")
        _prediction_service = PredictionService(
            sagemaker_endpoint=os.environ['SAGEMAKER_ENDPOINT_NAME'],
            neptune_endpoint=os.environ['NEPTUNE_ENDPOINT']
        )
        logger.info("Prediction service initialized")
    
    return _prediction_service


def create_mcp_response(result: Any, request_id: str) -> Dict:
    """Format response in MCP-compatible format."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result
    }


def create_mcp_error(code: int, message: str, request_id: str, data: Optional[Dict] = None) -> Dict:
    """Format error in MCP-compatible format."""
    error = {"code": code, "message": message}
    if data:
        error["data"] = data
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error
    }


def handle_initialize(params: Dict, request_id: str) -> Dict:
    """Handle MCP initialize request."""
    return create_mcp_response({
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {
            "name": "package-eta-predictor",
            "version": "1.0.0"
        }
    }, request_id)


def handle_list_tools(request_id: str) -> Dict:
    """Handle MCP tools/list request."""
    tools = [
        {
            "name": "predict_package_eta",
            "description": "Predict delivery time and event timeline for a package. Returns ETA, event predictions, delivery status, and remaining delivery time.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "package_id": {
                        "type": "string",
                        "description": "The tracking ID of the package (e.g., TBA327582930610)"
                    }
                },
                "required": ["package_id"]
            }
        },
        {
            "name": "predict_batch",
            "description": "Predict delivery times for multiple packages. More efficient for bulk predictions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "package_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of package tracking IDs (max 20)",
                        "maxItems": 20
                    }
                },
                "required": ["package_ids"]
            }
        },
        {
            "name": "get_package_status",
            "description": "Get current delivery status and last known location without running full prediction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "package_id": {
                        "type": "string",
                        "description": "The tracking ID of the package"
                    }
                },
                "required": ["package_id"]
            }
        }
    ]
    return create_mcp_response({"tools": tools}, request_id)


def handle_call_tool(params: Dict, request_id: str) -> Dict:
    """Handle MCP tools/call request."""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    logger.info(f"Calling tool: {tool_name}")
    
    try:
        service = get_prediction_service()
        
        if tool_name == "predict_package_eta":
            package_id = arguments.get("package_id")
            if not package_id:
                return create_mcp_error(-32602, "Missing required parameter: package_id", request_id)
            
            result = service.predict_single(package_id)
            
            return create_mcp_response({
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2, default=str)
                }],
                "isError": result.get("status") == "error"
            }, request_id)
        
        elif tool_name == "predict_batch":
            package_ids = arguments.get("package_ids", [])
            if not package_ids:
                return create_mcp_error(-32602, "Missing required parameter: package_ids", request_id)
            if len(package_ids) > 20:
                return create_mcp_error(-32602, "Maximum 20 packages per batch", request_id)
            
            results = service.predict_batch(package_ids)
            
            return create_mcp_response({
                "content": [{
                    "type": "text",
                    "text": json.dumps(results, indent=2, default=str)
                }],
                "isError": False
            }, request_id)
        
        elif tool_name == "get_package_status":
            package_id = arguments.get("package_id")
            if not package_id:
                return create_mcp_error(-32602, "Missing required parameter: package_id", request_id)
            
            result = service.get_package_status(package_id)
            
            return create_mcp_response({
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2, default=str)
                }],
                "isError": result.get("status") == "error"
            }, request_id)
        
        else:
            return create_mcp_error(-32601, f"Unknown tool: {tool_name}", request_id)
    
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
        return create_mcp_error(-32603, f"Internal error: {str(e)}", request_id)


def lambda_handler(event: Dict, context) -> Dict:
    """AWS Lambda handler for MCP server."""
    logger.info(f"Received event: {json.dumps(event)[:500]}")
    
    # Handle API Gateway
    if 'httpMethod' in event or 'requestContext' in event:
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                    'Access-Control-Allow-Methods': 'POST,OPTIONS'
                },
                'body': ''
            }
        
        try:
            body = event.get('body', '{}')
            if isinstance(body, str):
                body = json.loads(body)
        except json.JSONDecodeError as e:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': f'Invalid JSON: {str(e)}'})
            }
    else:
        body = event
    
    # Handle MCP request
    request_id = body.get('id', '1')
    method = body.get('method', '')
    params = body.get('params', {})
    
    logger.info(f"Processing MCP method: {method}")
    
    if method == 'initialize':
        response = handle_initialize(params, request_id)
    elif method == 'tools/list':
        response = handle_list_tools(request_id)
    elif method == 'tools/call':
        response = handle_call_tool(params, request_id)
    elif method == 'ping':
        response = create_mcp_response({"status": "ok"}, request_id)
    else:
        response = create_mcp_error(-32601, f"Method not found: {method}", request_id)
    
    # Format for API Gateway
    if 'httpMethod' in event or 'requestContext' in event:
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Methods': 'POST,OPTIONS'
            },
            'body': json.dumps(response, default=str)
        }
    
    return response