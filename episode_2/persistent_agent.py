"""
Network Operations Agent Memory Implementation - Episode 2
Full implementation using AGNTCY SDK for network management and monitoring
"""

# persistent_agent.py - Network-focused memory system
import sqlite3
import json
from datetime import datetime
import uuid
import asyncio
import os
from typing import Dict, Any, List, Optional

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

class NetworkPersistentAgent:
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name

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

        # Initialize memory databases
        self.vector_db = sqlite3.connect(f'memory_{agent_id}_vector.db')
        self.episodic_db = sqlite3.connect(f'memory_{agent_id}_episodes.db')
        self.network_db = sqlite3.connect(f'memory_{agent_id}_network.db')
        self.init_databases()

    def init_databases(self):
        # Vector memory table
        self.vector_db.execute('''CREATE TABLE IF NOT EXISTS vectors
                                 (id TEXT, content TEXT, metadata TEXT,
                                  embedding TEXT, timestamp TEXT)''')

        # Episodic memory table
        self.episodic_db.execute('''CREATE TABLE IF NOT EXISTS episodes
                                   (id TEXT, content TEXT, context_type TEXT,
                                    timestamp TEXT, agent_id TEXT)''')

        # Network-specific memory table
        self.network_db.execute('''CREATE TABLE IF NOT EXISTS network_events
                                  (id TEXT, device_id TEXT, event_type TEXT,
                                   severity TEXT, description TEXT,
                                   ip_address TEXT, timestamp TEXT)''')

        self.network_db.execute('''CREATE TABLE IF NOT EXISTS device_configs
                                  (device_id TEXT PRIMARY KEY, hostname TEXT,
                                   ip_address TEXT, device_type TEXT,
                                   last_config TEXT, last_updated TEXT)''')

    def remember(self, content: str, context_type: str = "network_event", **kwargs):
        """Store information across all memory systems"""
        timestamp = datetime.now().isoformat()
        memory_id = str(uuid.uuid4())

        # Vector memory: semantic similarity
        self.vector_db.execute(
            'INSERT INTO vectors VALUES (?, ?, ?, ?, ?)',
            (memory_id, content, json.dumps({"type": context_type, **kwargs}),
             "embedding_placeholder", timestamp)
        )

        # Episodic memory: chronological storage
        self.episodic_db.execute(
            'INSERT INTO episodes VALUES (?, ?, ?, ?, ?)',
            (memory_id, content, context_type, timestamp, self.agent_id)
        )

        # Network-specific storage
        if context_type == "network_event":
            self.network_db.execute(
                'INSERT INTO network_events VALUES (?, ?, ?, ?, ?, ?, ?)',
                (memory_id, kwargs.get('device_id', 'unknown'),
                 kwargs.get('event_type', 'general'),
                 kwargs.get('severity', 'info'), content,
                 kwargs.get('ip_address', ''), timestamp)
            )

        self.vector_db.commit()
        self.episodic_db.commit()
        self.network_db.commit()

        print(f"💾 Stored: {content[:50]}...")
        return memory_id

    def remember_device_config(self, device_id: str, hostname: str,
                              ip_address: str, device_type: str,
                              config: str):
        """Store device configuration"""
        timestamp = datetime.now().isoformat()

        self.network_db.execute('''
            INSERT OR REPLACE INTO device_configs
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (device_id, hostname, ip_address, device_type, config, timestamp))

        self.network_db.commit()
        print(f"💾 Stored config for device: {hostname} ({ip_address})")

    def recall(self, query: str, memory_types: List[str] = ["vector", "episodic"]):
        """Retrieve relevant information from memory systems"""
        results = {}

        if "vector" in memory_types:
            cursor = self.vector_db.execute(
                'SELECT content FROM vectors WHERE content LIKE ?',
                (f'%{query}%',)
            )
            results["vector"] = [row[0] for row in cursor.fetchall()]

        if "episodic" in memory_types:
            cursor = self.episodic_db.execute(
                'SELECT content FROM episodes WHERE content LIKE ? ORDER BY timestamp DESC',
                (f'%{query}%',)
            )
            results["episodic"] = [row[0] for row in cursor.fetchall()]

        if "network" in memory_types:
            cursor = self.network_db.execute(
                'SELECT description, device_id, event_type, severity FROM network_events WHERE description LIKE ? ORDER BY timestamp DESC',
                (f'%{query}%',)
            )
            results["network"] = [
                {"description": row[0], "device_id": row[1], "event_type": row[2], "severity": row[3]}
                for row in cursor.fetchall()
            ]

        return results

    def get_device_history(self, device_id: str) -> List[Dict]:
        """Get all events for a specific device"""
        cursor = self.network_db.execute(
            'SELECT event_type, severity, description, timestamp FROM network_events WHERE device_id = ? ORDER BY timestamp DESC',
            (device_id,)
        )
        return [
            {"event_type": row[0], "severity": row[1], "description": row[2], "timestamp": row[3]}
            for row in cursor.fetchall()
        ]

    def get_device_config(self, device_id: str) -> Optional[Dict]:
        """Retrieve stored device configuration"""
        cursor = self.network_db.execute(
            'SELECT hostname, ip_address, device_type, last_config, last_updated FROM device_configs WHERE device_id = ?',
            (device_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "hostname": row[0],
                "ip_address": row[1],
                "device_type": row[2],
                "config": row[3],
                "last_updated": row[4]
            }
        return None

    async def analyze_network_issue(self, issue_description: str) -> str:
        """Use LLM to analyze network issues"""
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a network operations expert. Analyze network issues and provide troubleshooting guidance."},
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
                    system="You are a network operations expert. Analyze network issues and provide troubleshooting guidance.",
                    messages=[{"role": "user", "content": f"Network issue: {issue_description}"}]
                )
                return response.content[0].text
            except Exception as e:
                return f"Analysis error: {e}"
        else:
            return "Mock analysis: Check device connectivity, verify configurations, review recent changes, and check logs for errors."

# Demo - Network Operations Use Case
async def run_network_demo():
    print("🌐 Network Operations Agent Demo")
    print("="*60)

    agent = NetworkPersistentAgent("network_agent_001", "Network Operations Assistant")

    print("\n📡 Simulating network events...")

    # Store network events
    agent.remember(
        "Router R1 interface GigabitEthernet0/1 went down",
        context_type="network_event",
        device_id="R1",
        event_type="interface_down",
        severity="critical",
        ip_address="10.10.10.1"
    )

    agent.remember(
        "Switch SW1 CPU utilization reached 85%",
        context_type="network_event",
        device_id="SW1",
        event_type="high_cpu",
        severity="warning",
        ip_address="10.10.20.1"
    )

    agent.remember(
        "Router R1 interface GigabitEthernet0/1 came back up",
        context_type="network_event",
        device_id="R1",
        event_type="interface_up",
        severity="info",
        ip_address="10.10.10.1"
    )

    # Store device configurations
    print("\n💾 Storing device configurations...")
    agent.remember_device_config(
        device_id="R1",
        hostname="core-router-1",
        ip_address="10.10.10.1",
        device_type="Router",
        config="interface GigabitEthernet0/1\n ip address 192.168.1.1 255.255.255.0\n no shutdown"
    )

    agent.remember_device_config(
        device_id="SW1",
        hostname="access-switch-1",
        ip_address="10.10.20.1",
        device_type="Switch",
        config="interface GigabitEthernet1/0/1\n switchport mode access\n switchport access vlan 10"
    )

    # Recall information
    print("\n🔍 Querying network memory...")
    query = "Router R1"
    results = agent.recall(query, memory_types=["vector", "episodic", "network"])
    print(f"\nQuery: '{query}'")
    print(f"Found {len(results.get('vector', []))} vector matches")
    print(f"Found {len(results.get('episodic', []))} episodic matches")
    print(f"Found {len(results.get('network', []))} network event matches")

    # Get device history
    print("\n📋 Device history for R1:")
    history = agent.get_device_history("R1")
    for event in history:
        print(f"  [{event['severity']}] {event['event_type']}: {event['description']}")

    # Get device config
    print("\n⚙️  Device configuration for R1:")
    config = agent.get_device_config("R1")
    if config:
        print(f"  Hostname: {config['hostname']}")
        print(f"  IP: {config['ip_address']}")
        print(f"  Type: {config['device_type']}")
        print(f"  Last Updated: {config['last_updated']}")

    # Analyze network issue with LLM
    print("\n🤖 AI-Powered Network Analysis:")
    issue = "Router R1 interface went down briefly and came back up. Switch SW1 showing high CPU utilization."
    analysis = await agent.analyze_network_issue(issue)
    print(f"Analysis:\n{analysis}")

if __name__ == "__main__":
    asyncio.run(run_network_demo())