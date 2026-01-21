from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class DocumentChunk:
    content: str
    source_id: str
    source_type: str   # pdf / wikipedia
    title: str
    metadata: Dict[str, Any]


@dataclass
class AnswerSource:
    source_type: str   # pdf / wikipedia
    title: str
    snippet: str
