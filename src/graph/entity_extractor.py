"""
Entity extraction using LLM-based extraction (Groq/Ollama) and optional GLiNER.
Extracts banking entities from chunks: loan types, limits, currencies, roles, conditions.
"""

import json
import re
from typing import Optional
from dataclasses import dataclass, field, asdict

from openai import OpenAI

from config import get_config, Config, LLMConfig
from src.ingestion.chunker import Chunk
import ollama as ollama_client


@dataclass
class Entity:
    """An extracted entity."""
    entity_id: str
    name: str
    entity_type: str  # "loan_type", "currency", "limit", "percentage", "role", "condition", "regulation", "product", "organization"
    value: Optional[str] = None  # For numeric entities: the actual value
    source_chunk_id: str = ""
    source_doc: str = ""
    mentions: list[str] = field(default_factory=list)  # All text mentions
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Relation:
    """A relationship between two entities or between an entity and a chunk."""
    source_id: str
    target_id: str
    relation_type: str  # "requires", "defines", "refers_to", "prohibits", "depends_on", "has_limit", "belongs_to"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


EXTRACTION_PROMPT = """You are a banking domain entity and relationship extractor.
Given the following text chunk from a banking policy document, extract:

1. ENTITIES: Named things like loan types, currencies, regulatory limits, percentages, roles, conditions, products, organizations, regulations.
2. RELATIONS: How entities relate to each other (e.g., "requires", "defines", "prohibits", "depends_on", "has_limit").

Rules:
- Extract EXACT values for numeric entities (percentages, limits, amounts).
- For conditional logic (IF/THEN), extract the condition and its consequence as separate entities linked by "depends_on".
- Include Arabic entities as-is (do not translate).
- Be comprehensive but precise — only extract clearly stated facts, not inferences.

Respond with ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "entity name", "type": "loan_type|currency|limit|percentage|role|condition|regulation|product|organization|definition", "value": "exact value if numeric, null otherwise"}}
  ],
  "relations": [
    {{"source": "entity name 1", "target": "entity name 2", "type": "requires|defines|refers_to|prohibits|depends_on|has_limit|governs"}}
  ]
}}

TEXT CHUNK:
---
{chunk_text}
---

JSON OUTPUT:"""


def _get_llm_client(llm_config: LLMConfig):
    """Create an OpenAI-compatible client for any provider."""
    if llm_config.provider == "ollama_chat":
        return None  # Use ollama library directly
    return OpenAI(
        api_key=llm_config.api_key or "not-needed",
        base_url=llm_config.base_url,
    )


def _call_llm(llm_config: LLMConfig, system_msg: str, user_msg: str, temperature: float = 0.0, max_tokens: int = 1500) -> str:
    """Call the LLM via appropriate provider."""
    if llm_config.provider == "ollama_chat":
        
        response = ollama_client.chat(
            model=llm_config.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]
    else:
        client = OpenAI(
            api_key=llm_config.api_key or "not-needed",
            base_url=llm_config.base_url,
        )
        response = client.chat.completions.create(
            model=llm_config.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def _extract_json_from_response(text: str) -> dict:
    """Robustly extract JSON from LLM response."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { to last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {"entities": [], "relations": []}


def _make_entity_id(name: str, entity_type: str, source_doc: str) -> str:
    """Generate a deterministic entity ID."""
    import hashlib
    raw = f"{source_doc}::{entity_type}::{name.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def extract_entities_from_chunk(
    chunk: Chunk,
    config: Optional[Config] = None,
    use_fast_llm: bool = True,
) -> tuple[list[Entity], list[Relation]]:
    """
    Extract entities and relations from a single chunk using the LLM.
    
    Args:
        chunk: The text chunk to extract from.
        config: Config object (uses fast_llm by default for cost efficiency).
        use_fast_llm: If True, use the fast/cheap LLM for extraction.
    
    Returns:
        Tuple of (entities, relations)
    """
    config = config or get_config()
    llm_config = config.fast_llm if use_fast_llm else config.main_llm

    prompt = EXTRACTION_PROMPT.format(chunk_text=chunk.content)

    try:
        raw_text = _call_llm(
            llm_config,
            system_msg="You are a precise banking entity extractor. Output only valid JSON.",
            user_msg=prompt,
        )
        data = _extract_json_from_response(raw_text)
    except Exception as e:
        print(f"  LLM extraction failed for chunk {chunk.chunk_id}: {e}")
        return [], []

    # Convert to Entity objects
    entities: list[Entity] = []
    entity_name_to_id: dict[str, str] = {}

    for ent_data in data.get("entities", []):
        name = (ent_data.get("name") or "").strip()
        etype = (ent_data.get("type") or "unknown").strip()
        value = ent_data.get("value")

        if not name:
            continue

        eid = _make_entity_id(name, etype, chunk.doc_name)
        entity_name_to_id[name] = eid

        entities.append(Entity(
            entity_id=eid,
            name=name,
            entity_type=etype,
            value=str(value) if value else None,
            source_chunk_id=chunk.chunk_id,
            source_doc=chunk.doc_name,
            mentions=[name],
        ))

    # Convert to Relation objects
    relations: list[Relation] = []
    for rel_data in data.get("relations", []):
        if not isinstance(rel_data, dict):
            continue
        source_name = (rel_data.get("source") or "").strip()
        target_name = (rel_data.get("target") or "").strip()
        rtype = (rel_data.get("type") or "related_to").strip()

        source_id = entity_name_to_id.get(source_name)
        target_id = entity_name_to_id.get(target_name)

        if source_id and target_id:
            relations.append(Relation(
                source_id=source_id,
                target_id=target_id,
                relation_type=rtype,
            ))

    return entities, relations


def extract_entities_from_chunks(
    chunks: list[Chunk],
    config: Optional[Config] = None,
    use_fast_llm: bool = True,
) -> tuple[list[Entity], list[Relation]]:
    """Batch extract entities from all chunks."""
    config = config or get_config()
    all_entities: list[Entity] = []
    all_relations: list[Relation] = []

    print(f"Extracting entities from {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        if i % 5 == 0:
            print(f"  Processing chunk {i + 1}/{len(chunks)}...")
        entities, relations = extract_entities_from_chunk(chunk, config, use_fast_llm)
        all_entities.extend(entities)
        all_relations.extend(relations)

    print(f"Extracted {len(all_entities)} entities and {len(all_relations)} relations")
    return all_entities, all_relations


def align_entities(entities: list[Entity], similarity_threshold: float = 0.85) -> dict[str, str]:
    """
    Simple entity alignment: merge entities that refer to the same concept.
    
    Returns a mapping of entity_id -> canonical_entity_id.
    Uses exact name matching + normalized string matching.
    For full version, would use embedding similarity.
    """
    canonical_map: dict[str, str] = {}  # entity_id -> canonical_id
    name_to_canonical: dict[str, str] = {}  # normalized_name -> canonical_id

    for entity in entities:
        # Normalize: lowercase, strip whitespace, collapse spaces
        normalized = re.sub(r'\s+', ' ', entity.name.lower().strip())

        if normalized in name_to_canonical:
            # Already seen this exact name — map to canonical
            canonical_map[entity.entity_id] = name_to_canonical[normalized]
        else:
            # New canonical entity
            canonical_map[entity.entity_id] = entity.entity_id
            name_to_canonical[normalized] = entity.entity_id

    merged_count = len(entities) - len(set(canonical_map.values()))
    print(f"Entity alignment: {len(entities)} entities → {len(set(canonical_map.values()))} canonical ({merged_count} merged)")
    return canonical_map
