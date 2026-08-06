"""Neo4j knowledge graph builder. Scaffold only in v1."""
from __future__ import annotations

import os

from neo4j import GraphDatabase

from core.policy_types import POLICY_TYPES

# Flag to run without Neo4j if unavailable
NEO4J_ENABLED = bool(os.getenv("NEO4J_URI"))


def build_knowledge_graph(
    brand_name: str,
    policy_type: str,
    policy_variables: dict,
    seed_posts: list[dict] | None = None,
) -> None:
    """
    Build Neo4j graph with nodes: Brand, Policy, Persona (x5).
    Relationships: (Brand)-[:ANNOUNCED]->(Policy), (Policy)-[:AFFECTS]->(Persona) x5.

    Scaffolding for v2 — in v1, graph is built but not queried during simulation.
    """
    if not NEO4J_ENABLED:
        return

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")

    policy_cfg = POLICY_TYPES.get(policy_type, {})
    announcement_text = policy_cfg.get("announcement_template", "").format(
        date="TBD", **policy_variables
    )

    driver = GraphDatabase.driver(uri, auth=(user, password))

    personas = ["loyal", "casual", "deal_seeker", "influencer", "sustainability"]

    with driver.session() as session:
        # Create Brand node
        session.run(
            "MERGE (b:Brand {name: $name})",
            name=brand_name,
        )

        # Create Policy node
        session.run(
            "MERGE (p:Policy {type: $type, announcement: $announcement})"
            " SET p.variables = $variables",
            type=policy_type,
            announcement=announcement_text,
            variables=str(policy_variables),
        )

        # Brand -> Policy
        session.run(
            "MATCH (b:Brand {name: $brand}), (p:Policy {type: $type})"
            " MERGE (b)-[:ANNOUNCED]->(p)",
            brand=brand_name,
            type=policy_type,
        )

        # Policy -> Persona (x5)
        for persona in personas:
            session.run(
                "MERGE (per:Persona {type: $persona})",
                persona=persona,
            )
            session.run(
                "MATCH (p:Policy {type: $type}), (per:Persona {type: $persona})"
                " MERGE (p)-[:AFFECTS]->(per)",
                type=policy_type,
                persona=persona,
            )

    driver.close()
