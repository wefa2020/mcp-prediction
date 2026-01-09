#!/usr/bin/env python3
"""Test MCP Server."""

import requests
import json
import argparse
import time
import sys


def call_mcp(endpoint: str, method: str, params: dict = None, timeout: int = 120):
    """Make MCP call."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(int(time.time() * 1000)),
        "method": method,
    }
    if params:
        payload["params"] = params
    
    response = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def test_initialize(endpoint: str):
    """Test initialize."""
    print("Testing initialize...")
    result = call_mcp(endpoint, "initialize", {"protocolVersion": "2024-11-05"})
    assert 'result' in result
    print(f"  ✓ Server: {result['result']['serverInfo']['name']}")
    return True


def test_list_tools(endpoint: str):
    """Test list tools."""
    print("\nTesting tools/list...")
    result = call_mcp(endpoint, "tools/list")
    tools = result['result']['tools']
    for t in tools:
        print(f"  ✓ {t['name']}")
    return True


def test_predict(endpoint: str, package_id: str):
    """Test prediction."""
    print(f"\nTesting predict_package_eta ({package_id})...")
    
    start = time.time()
    result = call_mcp(endpoint, "tools/call", {
        "name": "predict_package_eta",
        "arguments": {"package_id": package_id}
    })
    elapsed = time.time() - start
    
    content = result['result']['content'][0]['text']
    pred = json.loads(content)
    
    print(f"  ✓ Response time: {elapsed:.2f}s")
    print(f"  ✓ Status: {pred.get('status')}")
    print(f"  ✓ Delivery Status: {pred.get('delivery_status')}")
    
    if pred.get('status') == 'success':
        print(f"  ✓ Events: {pred.get('num_events')} ({pred.get('actual_events')} actual, {pred.get('predicted_events')} predicted)")
        if pred.get('eta'):
            print(f"  ✓ ETA: {pred.get('eta')}")
        if pred.get('remaining_hours'):
            print(f"  ✓ Remaining: {pred.get('remaining_hours'):.1f}h")
    
    return pred


def test_status(endpoint: str, package_id: str):
    """Test get status."""
    print(f"\nTesting get_package_status ({package_id})...")
    
    result = call_mcp(endpoint, "tools/call", {
        "name": "get_package_status",
        "arguments": {"package_id": package_id}
    })
    
    content = result['result']['content'][0]['text']
    status = json.loads(content)
    
    print(f"  ✓ Delivery Status: {status.get('delivery_status')}")
    if status.get('last_known_location'):
        loc = status['last_known_location']
        print(f"  ✓ Last Location: {loc.get('event_type')} at {loc.get('location')}")
    
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', required=True, help='MCP API endpoint')
    parser.add_argument('--package-id', default='TBA327582930610', help='Package ID to test')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MCP Server Tests")
    print("=" * 60)
    print(f"Endpoint: {args.endpoint}")
    print(f"Package: {args.package_id}")
    print("=" * 60)
    
    tests = [
        ("Initialize", lambda: test_initialize(args.endpoint)),
        ("List Tools", lambda: test_list_tools(args.endpoint)),
        ("Get Status", lambda: test_status(args.endpoint, args.package_id)),
        ("Predict", lambda: test_predict(args.endpoint, args.package_id)),
    ]
    
    passed = 0
    failed = 0
    
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())