"""消息系统"""
from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any
from datetime import datetime

MessageRole = Literal["system", "user", "assistant", "tool"]
class Message(BaseModel):
  
    role: MessageRole
    content: str
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __init__(self, role: MessageRole, content: str, **kwargs):
        super().__init__(role=role, content=content, **kwargs)(
          content=content,
          role=role,
          timestamp=kwargs.get("timestamp", datetime.now()),
          metadata=kwargs.get("metadata",{})
        )
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
        }
    
    def __str__(self):
        return f"{self.role} Message: {self.content}"
          