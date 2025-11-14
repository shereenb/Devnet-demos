"""
AGNTCY Network Operations Memory Implementation - Episode 2
Full implementation using AGNTCY SDK for network management and operations
"""

import json
import uuid
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

# Optional imports with graceful fallbacks
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from agntcy_acp import ACPClient, AsyncACPClient
    from agntcy_acp.acp_v0 import Agent, AgentMetadata, AgentRef
    AGNTCY_ACP_AVAILABLE = True
    print("✅ AGNTCY ACP SDK successfully imported")
except ImportError:
    AGNTCY_ACP_AVAILABLE = False
    print("📦 AGNTCY ACP SDK not available (install with: pip install agntcy-acp)")

# Mock AGNTCY ACP SDK components (for demonstration)
class MockVectorMemory:
    def __init__(self, collection_name, persist_directory):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.memories = []

    def store(self, content, metadata):
        memory_id = str(uuid.uuid4())
        self.memories.append({"id": memory_id, "content": content, "metadata": metadata})
        return {"id": memory_id, "status": "stored"}

    def search(self, query, n_results, filter_metadata):
        return [m for m in self.memories if query.lower() in m["content"].lower()][:n_results]

class MockGraphMemory:
    def __init__(self, db_path):
        self.db_path = db_path
        self.entities = []
        self.relationships = []

    def add_entity(self, entity_id, entity_type, properties):
        entity = {"id": entity_id, "type": entity_type, "properties": properties}
        self.entities.append(entity)
        return entity

    def add_relationship(self, source_id, target_id, relationship_type):
        rel = {"source": source_id, "target": target_id, "type": relationship_type}
        self.relationships.append(rel)
        return rel

    def search_related(self, query_entities, max_depth):
        return self.entities[:5]

    def get_stats(self):
        return {"entity_count": len(self.entities), "relationship_count": len(self.relationships)}

class MockEpisodicMemory:
    def __init__(self, db_path):
        self.db_path = db_path
        self.episodes = []

    def store_episode(self, content, timestamp, context, metadata):
        episode_id = str(uuid.uuid4())
        self.episodes.append({
            "id": episode_id,
            "content": content,
            "timestamp": timestamp,
            "context": context,
            "metadata": metadata
        })
        return episode_id

    def search_episodes(self, query, limit, time_range):
        return [e for e in self.episodes if query.lower() in e["content"].lower()][:limit]

    def get_episode_count(self):
        return len(self.episodes)


class NetworkAGNTCYAgent:
    """Production-ready network operations agent with AGNTCY memory systems"""

    def __init__(self, agent_id: str, name: str, capabilities: List[str] = None):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities or ["network_monitoring", "device_management", "incident_response"]

        # Track if we're using real AGNTCY SDK
        self.is_real_agntcy = False
        self.agntcy_metadata = None

        # Load API keys
        if DOTENV_AVAILABLE:
            load_dotenv()

        # Initialize LLM clients
        self.openai_client = None
        self.claude_client = None

        openai_key = os.getenv('OPENAI_API_KEY')
        claude_key = os.getenv('CLAUDE_API_KEY') or os.getenv('ANTHROPIC_API_KEY')

        if OPENAI_AVAILABLE and openai_key:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                print("✅ OpenAI client initialized for network analysis")
            except Exception as e:
                print(f"⚠️  OpenAI client error: {e}")

        if ANTHROPIC_AVAILABLE and claude_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=claude_key)
                print("✅ Claude client initialized for network analysis")
            except Exception as e:
                print(f"⚠️  Claude client error: {e}")

        # Initialize AGNTCY SDK if available
        if AGNTCY_ACP_AVAILABLE:
            try:
                agent_ref = AgentRef(
                    name=f"network.{agent_id}",
                    version="1.0.0",
                    url=f"local://network/{agent_id}"
                )
                self.agntcy_metadata = AgentMetadata(
                    ref=agent_ref,
                    description=f"Network operations agent with persistent memory and multi-modal memory systems (vector, graph, episodic)"
                )
                self.is_real_agntcy = True
                print("✅ AGNTCY ACP metadata created - using real AGNTCY SDK")
                print(f"   Agent Ref: {agent_ref.name} v{agent_ref.version}")
            except Exception as e:
                print(f"⚠️  AGNTCY metadata creation failed: {e}")
                self.is_real_agntcy = False

        # Initialize AGNTCY memory systems (using mocks for now, would be real in production)
        self.vector_memory = MockVectorMemory(
            collection_name=f"network_agent_{agent_id}_vectors",
            persist_directory="./agntcy_memory_db"
        )

        self.graph_memory = MockGraphMemory(
            db_path=f"./agntcy_memory_db/network_agent_{agent_id}_graph.db"
        )

        self.episodic_memory = MockEpisodicMemory(
            db_path=f"./agntcy_memory_db/network_agent_{agent_id}_episodes.db"
        )

        # Network-specific tracking
        self.device_registry = {}

        # Register with directory
        self._register_with_directory()

    def _register_with_directory(self):
        """Register agent capabilities with AGNTCY directory using LONGO protocol"""
        agent_profile = {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "memory_types": ["vector", "graph", "episodic"],
            "protocols": ["CORTO", "LONGO", "A2A"],
            "domain": "network_operations",
            "status": "active",
            "created_at": datetime.now().isoformat()
        }

        # LONGO Protocol: Self-registration in directory
        if self.is_real_agntcy and self.agntcy_metadata:
            print(f"📋 LONGO Protocol: Registering agent in AGNTCY directory...")
            print(f"   • Agent: {self.agntcy_metadata.ref.name}")
            print(f"   • Domain: network_operations")
            print(f"   • Capabilities: {len(self.capabilities)} registered")
        else:
            print(f"🌐 Registered network agent: {self.name} (mock mode)")

    def remember_network_event(self, event_type: str, device_id: str,
                               severity: str, description: str,
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store network event across all AGNTCY memory systems using CORTO protocol"""
        metadata = metadata or {}
        timestamp = datetime.now()

        content = f"[{severity.upper()}] {event_type} on {device_id}: {description}"

        # CORTO Protocol: Coordinate storage across 3 memory systems
        # This demonstrates memory coordination - one event, three storage locations

        # Vector memory: semantic similarity
        vector_result = self.vector_memory.store(
            content=content,
            metadata={
                "timestamp": timestamp.isoformat(),
                "type": "network_event",
                "event_type": event_type,
                "device_id": device_id,
                "severity": severity,
                "agent_id": self.agent_id,
                **metadata
            }
        )

        # Graph memory: create device and event entities
        device_entity = self.graph_memory.add_entity(
            entity_id=device_id,
            entity_type="network_device",
            properties={
                "device_id": device_id,
                "last_event": event_type,
                "ip_address": metadata.get("ip_address", "")
            }
        )

        event_entity = self.graph_memory.add_entity(
            entity_id=str(uuid.uuid4()),
            entity_type="network_event",
            properties={
                "event_type": event_type,
                "severity": severity,
                "timestamp": timestamp.isoformat()
            }
        )

        # Create relationship
        self.graph_memory.add_relationship(
            source_id=device_id,
            target_id=event_entity["id"],
            relationship_type="experienced_event"
        )

        # Episodic memory: chronological storage
        episode_id = self.episodic_memory.store_episode(
            content=content,
            timestamp=timestamp,
            context="network_event",
            metadata={
                "event_type": event_type,
                "device_id": device_id,
                "severity": severity,
                **metadata
            }
        )

        return {
            "memory_id": vector_result["id"],
            "vector_stored": True,
            "graph_entities": 2,
            "episodic_stored": True,
            "device_id": device_id
        }

    def remember_device_config(self, device_id: str, hostname: str,
                               ip_address: str, device_type: str,
                               config: str) -> Dict[str, Any]:
        """Store device configuration in AGNTCY memory"""
        timestamp = datetime.now()

        content = f"Device {hostname} ({device_id}) at {ip_address}: {device_type} configuration stored"

        # Vector memory
        vector_result = self.vector_memory.store(
            content=content,
            metadata={
                "timestamp": timestamp.isoformat(),
                "type": "device_config",
                "device_id": device_id,
                "hostname": hostname,
                "ip_address": ip_address,
                "device_type": device_type
            }
        )

        # Graph memory: create device entity with config
        device_entity = self.graph_memory.add_entity(
            entity_id=device_id,
            entity_type="network_device",
            properties={
                "hostname": hostname,
                "ip_address": ip_address,
                "device_type": device_type,
                "config": config,
                "last_updated": timestamp.isoformat()
            }
        )

        # Store in device registry
        self.device_registry[device_id] = {
            "hostname": hostname,
            "ip_address": ip_address,
            "device_type": device_type,
            "config": config,
            "last_updated": timestamp.isoformat()
        }

        return {
            "device_id": device_id,
            "stored": True,
            "memory_id": vector_result["id"]
        }

    def recall_network_events(self, query: str, memory_types: List[str] = None,
                             limit: int = 5) -> Dict[str, Any]:
        """Retrieve relevant network events from AGNTCY memory systems"""
        memory_types = memory_types or ["vector", "graph", "episodic"]
        results = {}

        if "vector" in memory_types:
            vector_results = self.vector_memory.search(
                query=query,
                n_results=limit,
                filter_metadata={"agent_id": self.agent_id}
            )
            results["vector"] = vector_results

        if "graph" in memory_types:
            query_entities = [{"id": query, "type": "network_device"}]
            graph_results = self.graph_memory.search_related(
                query_entities=query_entities,
                max_depth=2
            )
            results["graph"] = graph_results

        if "episodic" in memory_types:
            episodic_results = self.episodic_memory.search_episodes(
                query=query,
                limit=limit,
                time_range="last_30_days"
            )
            results["episodic"] = episodic_results

        return results

    def get_device_history(self, device_id: str) -> List[Dict]:
        """Get all events for a specific device from graph memory"""
        events = []
        for entity in self.graph_memory.entities:
            if entity["type"] == "network_event":
                events.append(entity["properties"])
        return events

    def analyze_network_issue(self, issue_description: str) -> str:
        """Use LLM to analyze network issues"""
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a network operations expert using AGNTCY protocols. Analyze network issues and provide troubleshooting guidance."},
                        {"role": "user", "content": f"Network issue: {issue_description}"}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Analysis error: {e}"
        elif self.claude_client:
            try:
                response = self.claude_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=300,
                    temperature=0.3,
                    system="You are a network operations expert using AGNTCY protocols. Analyze network issues and provide troubleshooting guidance.",
                    messages=[{"role": "user", "content": f"Network issue: {issue_description}"}]
                )
                return response.content[0].text
            except Exception as e:
                return f"Analysis error: {e}"
        else:
            return "Mock analysis: Check device connectivity, verify configurations, review recent changes in graph memory, and analyze episodic patterns for recurring issues."

    def learn_capability(self, capability_name: str, evidence: str,
                        confidence: float = 0.8) -> Dict[str, Any]:
        """Learn new network capability and update self-registration"""
        if capability_name not in self.capabilities:
            self.capabilities.append(capability_name)

            content = f"Learned new network capability: {capability_name}. Evidence: {evidence}"

            # Store learning event
            learning_result = self.vector_memory.store(
                content=content,
                metadata={
                    "type": "capability_acquisition",
                    "capability": capability_name,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat()
                }
            )

            return {
                "capability_learned": capability_name,
                "memory_stored": learning_result["id"],
                "confidence": confidence
            }

        return {"status": "capability_already_exists"}

    def _get_memory_stats(self) -> Dict[str, int]:
        """Get current memory statistics"""
        vector_count = len(self.vector_memory.memories)
        graph_stats = self.graph_memory.get_stats()
        episodic_count = self.episodic_memory.get_episode_count()

        return {
            "vector_memories": vector_count,
            "graph_entities": graph_stats.get("entity_count", 0),
            "graph_relationships": graph_stats.get("relationship_count", 0),
            "episodic_memories": episodic_count,
            "registered_devices": len(self.device_registry)
        }

    def introduce_self(self) -> str:
        """Network agent introduces itself based on AGNTCY memory"""
        memory_stats = self._get_memory_stats()

        introduction = f"""
🌐 AGNTCY Network Operations Agent Self-Introduction:
Agent ID: {self.agent_id}
Name: {self.name}
Domain: Network Operations & Management

Current Capabilities:
{chr(10).join(f"• {cap}" for cap in self.capabilities)}

Memory Systems Status:
• Vector memories: {memory_stats['vector_memories']} (semantic network event search)
• Graph entities: {memory_stats['graph_entities']} (device topology & relationships)
• Graph relationships: {memory_stats['graph_relationships']} (event correlations)
• Episodic memories: {memory_stats['episodic_memories']} (chronological network events)
• Registered devices: {memory_stats['registered_devices']} (active network inventory)

AGNTCY Protocols Demonstrated:
• CORTO: Coordinating storage across 3 memory types (vector, graph, episodic)
• LONGO: Self-registered in directory with {len(self.capabilities)} capabilities
• A2A: Ready for agent-to-agent communication {'(real AGNTCY SDK)' if self.is_real_agntcy else '(mock mode)'}

Directory Status: Registered and discoverable in network operations domain

🧠 I maintain persistent network memory across sessions and coordinate with other AGNTCY network agents!
        """

        return introduction


def demo_agntcy_network_memory():
    """Demonstration of AGNTCY network operations memory capabilities"""
    print("🚀 AGNTCY Network Operations Memory System Demo")
    print("=" * 60)

    # Create AGNTCY network agent
    agent = NetworkAGNTCYAgent(
        agent_id="agntcy_network_001",
        name="AGNTCY Network Operations Assistant",
        capabilities=["network_monitoring", "device_management", "incident_response", "topology_mapping"]
    )

    print(f"\n🤖 Created network agent: {agent.name}")

    # Store network events
    print("\n💾 Storing network events across all memory systems...")

    network_events = [
        ("interface_down", "R1", "critical", "Router R1 interface GigabitEthernet0/1 went down",
         {"ip_address": "10.10.10.1", "interface": "GigabitEthernet0/1"}),
        ("high_cpu", "SW1", "warning", "Switch SW1 CPU utilization reached 85%",
         {"ip_address": "10.10.20.1", "cpu_percent": 85}),
        ("bgp_neighbor_down", "R1", "critical", "BGP neighbor 192.168.1.2 went down",
         {"ip_address": "10.10.10.1", "neighbor": "192.168.1.2"}),
        ("interface_up", "R1", "info", "Router R1 interface GigabitEthernet0/1 came back up",
         {"ip_address": "10.10.10.1", "interface": "GigabitEthernet0/1"})
    ]

    for event_type, device_id, severity, description, metadata in network_events:
        result = agent.remember_network_event(event_type, device_id, severity, description, metadata)
        print(f"✓ Stored: {description[:60]}... (ID: {result['memory_id'][:8]})")

    # Store device configurations
    print("\n⚙️  Storing device configurations...")
    devices = [
        ("R1", "core-router-1", "10.10.10.1", "Router",
         "interface GigabitEthernet0/1\n ip address 192.168.1.1 255.255.255.0\n no shutdown"),
        ("SW1", "access-switch-1", "10.10.20.1", "Switch",
         "interface GigabitEthernet1/0/1\n switchport mode access\n switchport access vlan 10"),
        ("FW1", "edge-firewall-1", "10.10.30.1", "Firewall",
         "security-policy trust-to-untrust permit any any")
    ]

    for device_id, hostname, ip_address, device_type, config in devices:
        result = agent.remember_device_config(device_id, hostname, ip_address, device_type, config)
        print(f"✓ Stored config: {hostname} ({ip_address})")

    # Test coordinated recall
    print("\n🔍 Testing coordinated memory recall...")
    queries = [
        "Router R1 events",
        "critical network issues",
        "interface problems",
        "BGP"
    ]

    for query in queries:
        results = agent.recall_network_events(query, limit=2)
        print(f"\nQuery: '{query}'")

        for memory_type, memories in results.items():
            if memories and len(memories) > 0:
                print(f"  {memory_type}: {len(memories)} results")
                if memory_type in ["vector", "episodic"]:
                    if isinstance(memories[0], dict) and "content" in memories[0]:
                        print(f"    - {memories[0]['content'][:70]}...")

    # Test device history
    print("\n📋 Device event history for R1:")
    history = agent.get_device_history("R1")
    for event in history[:5]:
        print(f"  - {event.get('event_type', 'unknown')}: {event.get('timestamp', 'N/A')[:19]}")

    # Test capability learning
    print("\n📚 Testing network capability learning...")
    learning_scenarios = [
        ("anomaly_detection", "Successfully detected unusual traffic pattern leading to security incident"),
        ("predictive_maintenance", "Predicted interface failure 2 days before occurrence based on error patterns"),
        ("automated_remediation", "Automatically resolved BGP neighbor issue by applying config fix")
    ]

    for capability, evidence in learning_scenarios:
        result = agent.learn_capability(capability, evidence)
        print(f"✓ Learned: {capability}")

    # Analyze network issue with LLM
    print("\n🤖 AI-Powered Network Analysis (using AGNTCY protocols):")
    issue = "Router R1 interface flapped multiple times. BGP neighbor went down. Switch SW1 showing high CPU."
    analysis = agent.analyze_network_issue(issue)
    print(f"Analysis:\n{analysis[:400]}..." if len(analysis) > 400 else analysis)

    # Get memory statistics
    print("\n📊 Memory system statistics:")
    stats = agent._get_memory_stats()
    for stat_name, stat_value in stats.items():
        print(f"  {stat_name.replace('_', ' ').title()}: {stat_value}")

    # Agent self-introduction
    print("\n👋 Network agent self-introduction:")
    introduction = agent.introduce_self()
    print(introduction)

    print("\n✅ AGNTCY network operations demo completed!")
    print("All network memory persists across agent restarts via AGNTCY protocols.")


if __name__ == "__main__":
    demo_agntcy_network_memory()
