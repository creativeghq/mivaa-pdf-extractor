"""
WebSocket API Routes

Real-time communication endpoints for:
- Job progress updates
- System health notifications
- Document processing status
- Search result streaming
"""

import logging
import json
from typing import Dict, Set, Any
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasts.
    
    Features:
    - Connection lifecycle management
    - Room-based broadcasting
    - Heartbeat monitoring
    - Automatic cleanup
    """
    
    def __init__(self):
        # Active connections by room
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, room: str = "default"):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        
        if room not in self.active_connections:
            self.active_connections[room] = set()
        
        self.active_connections[room].add(websocket)
        self.connection_metadata[websocket] = {
            "room": room,
            "connected_at": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat()
        }
        
        logger.info(f"WebSocket connected to room '{room}'. Total connections: {len(self.active_connections[room])}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        metadata = self.connection_metadata.get(websocket)
        if metadata:
            room = metadata["room"]
            if room in self.active_connections:
                self.active_connections[room].discard(websocket)
                if not self.active_connections[room]:
                    del self.active_connections[room]
            
            del self.connection_metadata[websocket]
            logger.info(f"WebSocket disconnected from room '{room}'")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")
    
    async def broadcast_to_room(self, message: dict, room: str = "default"):
        """Broadcast a message to all connections in a room."""
        if room not in self.active_connections:
            return
        
        disconnected = set()
        for connection in self.active_connections[room]:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_json(message)
                else:
                    disconnected.add(connection)
            except Exception as e:
                logger.error(f"Failed to broadcast to connection: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast a message to all connections in all rooms."""
        for room in list(self.active_connections.keys()):
            await self.broadcast_to_room(message, room)
    
    def get_connection_count(self, room: str = None) -> int:
        """Get the number of active connections."""
        if room:
            return len(self.active_connections.get(room, set()))
        return sum(len(connections) for connections in self.active_connections.values())


# Global connection manager
manager = ConnectionManager()


@router.websocket("")
async def websocket_endpoint(websocket: WebSocket, room: str = "default"):
    """
    Main WebSocket endpoint for real-time communication.
    
    Query Parameters:
    - room: Room name to join (default: "default")
    
    Message Format:
    ```json
    {
        "type": "message_type",
        "payload": {...},
        "timestamp": "2024-01-15T10:30:00.000Z"
    }
    ```
    
    Supported Message Types:
    - heartbeat: Keep connection alive
    - subscribe: Subscribe to specific events
    - unsubscribe: Unsubscribe from events
    - progress_update: Job progress updates
    - status_update: System status updates
    """
    # #250 D29: require a valid token before accepting the connection. Previously ANY
    # anonymous client could connect to any room (job progress) AND broadcast into it.
    # The token is passed as a query param (browsers can't set WS Authorization headers).
    token = websocket.query_params.get("token")
    claims = None
    if token:
        try:
            from app.dependencies import _get_jwt_middleware
            claims = await _get_jwt_middleware()._validate_token(token)
        except Exception:
            claims = None
    if not claims:
        await websocket.close(code=1008)  # 1008 = policy violation
        return

    await manager.connect(websocket, room)
    
    try:
        # Send welcome message
        await manager.send_personal_message({
            "type": "connected",
            "payload": {
                "room": room,
                "timestamp": datetime.utcnow().isoformat(),
                "message": f"Connected to room '{room}'"
            }
        }, websocket)
        
        # Listen for messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            message_type = message.get("type")
            
            if message_type == "heartbeat":
                # Update last heartbeat
                if websocket in manager.connection_metadata:
                    manager.connection_metadata[websocket]["last_heartbeat"] = datetime.utcnow().isoformat()
                
                # Send heartbeat response
                await manager.send_personal_message({
                    "type": "heartbeat_ack",
                    "payload": {"timestamp": datetime.utcnow().isoformat()}
                }, websocket)
            
            elif message_type == "broadcast":
                # #250 D29: clients may NOT broadcast into a room — only the server pushes
                # progress. Previously any connected client could inject messages to every
                # other client in the room. Ignore client-initiated broadcasts.
                logger.warning("Ignored client-initiated broadcast on room '%s'", room)
            
            else:
                # Echo message back
                await manager.send_personal_message({
                    "type": "echo",
                    "payload": message,
                    "timestamp": datetime.utcnow().isoformat()
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return {
        "total_connections": manager.get_connection_count(),
        "rooms": {
            room: len(connections)
            for room, connections in manager.active_connections.items()
        },
        "timestamp": datetime.utcnow().isoformat()
    }


