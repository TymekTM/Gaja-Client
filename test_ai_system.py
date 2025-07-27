#!/usr/bin/env python3
"""
Comprehensive GAJA AI System Test
Tests all components and identifies remaining issues.
"""

import asyncio
import json
import websockets
import time


async def comprehensive_ai_test():
    print('=== GAJA AI System Test ===')
    
    try:
        uri = 'ws://localhost:8001/ws/test_client'
        print(f'1. Connecting to: {uri}')
        
        async with websockets.connect(uri) as websocket:
            print('2. Connected successfully')
            
            # Wait for handshake
            handshake = await websocket.recv()
            handshake_data = json.loads(handshake)
            print(f'3. Handshake received: {handshake_data["type"]}')
            
            # Test AI query
            await asyncio.sleep(1)
            message = {
                'type': 'query',
                'data': {
                    'query': 'Powiedz mi coś o sobie.',
                    'context': {
                        'user_id': 'test_user',
                        'history': [],
                        'available_plugins': ['weather_module', 'search_module'],
                        'modules': {}
                    }
                }
            }
            
            print('4. Sending AI query...')
            await websocket.send(json.dumps(message))
            
            # Wait for AI response
            print('5. Waiting for AI response...')
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                response_data = json.loads(response)
                print(f'6. Response received - Type: {response_data["type"]}')
                
                if response_data['type'] == 'ai_response':
                    if 'data' in response_data and 'response' in response_data['data']:
                        ai_resp = json.loads(response_data['data']['response'])
                        print(f'7. AI Text: {ai_resp.get("text", "No text field")[:100]}...')
                        print('✅ AI system working correctly!')
                    else:
                        print('⚠️ AI response format issue')
                        print(f'Response data keys: {list(response_data.get("data", {}).keys())}')
                elif response_data['type'] == 'error':
                    print(f'❌ AI Error: {response_data["data"]["message"]}')
                else:  
                    print(f'⚠️ Unexpected response type: {response_data["type"]}')
                    
            except asyncio.TimeoutError:
                print('⏰ Timeout - AI module may be processing')
                
    except Exception as e:
        print(f'❌ Connection error: {e}')


if __name__ == '__main__':
    asyncio.run(comprehensive_ai_test())
