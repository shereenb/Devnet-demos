"""
Self-Registering Network Agent Implementation - Episode 2
Network operations agent with self-registration and capability learning
"""

import sqlite3
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()

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

class SelfRegisteringNetworkAgent:
    """Network agent that registers itself and learns network capabilities"""

    def __init__(self, base_capabilities=None, use_llm=False):
        # Generate unique identity
        self.agent_id = str(uuid.uuid4())
        self.name = f"NetAgent-{self.agent_id[:8]}"
        self.birth_time = datetime.now()

        # Base network capabilities
        self.capabilities = base_capabilities or [
            "device_monitoring",
            "configuration_management",
            "event_logging"
        ]

        # Network-specific attributes
        self.monitored_devices = []
        self.known_topologies = []

        # LLM integration (optional)
        self.use_llm = use_llm
        self.openai_client = None
        self.claude_client = None

        if use_llm:
            if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
                self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                print("✓ OpenAI API connected")
            if ANTHROPIC_AVAILABLE and os.getenv('CLAUDE_API_KEY'):
                self.claude_client = anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
                print("✓ Claude API connected")

        # Simple memory storage
        self.memory_db = sqlite3.connect(f'network_agent_{self.agent_id}_memory.db')
        self.memory_db.execute('''CREATE TABLE IF NOT EXISTS memories
                                 (id TEXT, content TEXT, type TEXT, timestamp TEXT)''')

        self.memory_db.execute('''CREATE TABLE IF NOT EXISTS devices
                                 (device_id TEXT PRIMARY KEY, hostname TEXT,
                                  ip_address TEXT, device_type TEXT, status TEXT,
                                  last_seen TEXT)''')

        # Register self on creation
        self._register_self()

    def _register_self(self):
        """Register agent's identity and network capabilities"""
        # Create capabilities table
        self.memory_db.execute('''CREATE TABLE IF NOT EXISTS capabilities
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             capability_name TEXT,
             learned_date TEXT,
             evidence TEXT,
             source TEXT)''')

        # Store initial capabilities
        for cap in self.capabilities:
            self.memory_db.execute(
                'INSERT INTO capabilities (capability_name, learned_date, evidence, source) VALUES (?, ?, ?, ?)',
                (cap, datetime.now().isoformat(), "Base capability on initialization", "system")
            )

        # Store self-identity in memory
        identity_content = (
            f"I am {self.name}, a network operations agent created at {self.birth_time}. "
            f"My capabilities: {', '.join(self.capabilities)}. "
            f"I monitor network devices, manage configurations, and learn from network events."
        )

        self.memory_db.execute(
            'INSERT INTO memories VALUES (?, ?, ?, ?)',
            (str(uuid.uuid4()),
             identity_content,
             "self_identity",
             datetime.now().isoformat())
        )
        self.memory_db.commit()

        print(f"🌐 Network Agent {self.name} registered")
        print(f"📋 Capabilities: {', '.join(self.capabilities)}")

    def register_device(self, device_id: str, hostname: str, ip_address: str,
                       device_type: str, status: str = "online"):
        """Register a network device for monitoring"""
        self.memory_db.execute('''
            INSERT OR REPLACE INTO devices
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (device_id, hostname, ip_address, device_type, status,
              datetime.now().isoformat()))
        self.memory_db.commit()

        if device_id not in self.monitored_devices:
            self.monitored_devices.append(device_id)

        # Remember this registration
        self.memory_db.execute(
            'INSERT INTO memories VALUES (?, ?, ?, ?)',
            (str(uuid.uuid4()),
             f"Registered device {hostname} ({device_id}) at {ip_address} as {device_type}",
             "device_registration",
             datetime.now().isoformat())
        )
        self.memory_db.commit()

        print(f"  ✓ Registered device: {hostname} ({ip_address})")

    def learn_capability(self, new_capability: str, evidence: str):
        """Learn new network capabilities and update registration"""
        if new_capability not in self.capabilities:
            self.capabilities.append(new_capability)

            # Store in capabilities table
            self.memory_db.execute(
                'INSERT INTO capabilities (capability_name, learned_date, evidence, source) VALUES (?, ?, ?, ?)',
                (new_capability, datetime.now().isoformat(), evidence, "learned")
            )

            # Remember learning this capability
            learning_content = (
                f"I learned new network capability: {new_capability}. "
                f"Evidence: {evidence}"
            )

            self.memory_db.execute(
                'INSERT INTO memories VALUES (?, ?, ?, ?)',
                (str(uuid.uuid4()),
                 learning_content,
                 "capability_acquisition",
                 datetime.now().isoformat())
            )
            self.memory_db.commit()

            print(f"  📚 {self.name} learned: {new_capability}")
            print(f"      → {evidence[:80]}...")

    def log_network_event(self, device_id: str, event_type: str,
                         severity: str, description: str):
        """Log a network event and potentially learn from it"""
        event_content = (
            f"[{severity.upper()}] {event_type} on device {device_id}: {description}"
        )

        self.memory_db.execute(
            'INSERT INTO memories VALUES (?, ?, ?, ?)',
            (str(uuid.uuid4()),
             event_content,
             "network_event",
             datetime.now().isoformat())
        )
        self.memory_db.commit()

        print(f"  📡 Logged: {event_content[:80]}...")

        # Learn from critical events
        if severity == "critical" and "troubleshooting" not in [c.lower() for c in self.capabilities]:
            self.learn_capability(
                "incident_troubleshooting",
                f"Encountered critical {event_type} event requiring investigation"
            )

    def get_device_count(self) -> int:
        """Get number of monitored devices"""
        cursor = self.memory_db.execute('SELECT COUNT(*) FROM devices')
        return cursor.fetchone()[0]

    def get_capability_history(self) -> list:
        """Get history of learned capabilities"""
        cursor = self.memory_db.execute(
            'SELECT content, timestamp FROM memories WHERE type="capability_acquisition" ORDER BY timestamp DESC'
        )
        return cursor.fetchall()

    def introduce_self(self):
        """Network agent introduces itself from memory"""
        cursor = self.memory_db.execute(
            'SELECT content FROM memories WHERE type="self_identity" ORDER BY timestamp DESC LIMIT 1'
        )
        identity_record = cursor.fetchone()

        # Get device statistics
        device_count = self.get_device_count()

        # Get recent capabilities
        capability_history = self.get_capability_history()

        introduction = f"""
🌐 Network Agent Self-Introduction:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {self.name}
Agent ID: {self.agent_id}
Age: {(datetime.now() - self.birth_time).total_seconds():.1f} seconds old
Domain: Network Operations

Current Capabilities ({len(self.capabilities)}):
{chr(10).join(f"  • {cap}" for cap in self.capabilities)}

Network Status:
  • Monitored Devices: {device_count}
  • Active Monitoring: {"✅ Yes" if device_count > 0 else "❌ No"}

Recent Learning History:
{chr(10).join(f"  • {cap[0][:80]}... ({cap[1][:19]})" for cap in capability_history[:3]) if capability_history else "  • No capabilities learned yet"}

Identity Record:
{identity_record[0] if identity_record else 'None found'}

🧠 I remember my network experiences and continuously learn new capabilities!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        return introduction

    def display_capabilities_table(self):
        """Display all capabilities in a clear table format"""
        cursor = self.memory_db.execute('''
            SELECT id, capability_name, learned_date, source, evidence
            FROM capabilities
            ORDER BY id
        ''')
        capabilities = cursor.fetchall()

        print("\n" + "="*80)
        print("CAPABILITIES TABLE - All Agent Capabilities")
        print("="*80)
        print(f"{'ID':<5} {'Capability':<30} {'Source':<12} {'Learned Date':<20}")
        print("-"*80)

        for cap in capabilities:
            cap_id, name, learned_date, source, evidence = cap
            date_short = learned_date[:19] if learned_date else "N/A"
            print(f"{cap_id:<5} {name:<30} {source:<12} {date_short:<20}")

        print("-"*80)
        print(f"Total Capabilities: {len(capabilities)}")
        print("="*80)

        # Show detailed evidence for learned capabilities
        cursor = self.memory_db.execute('''
            SELECT capability_name, evidence
            FROM capabilities
            WHERE source = 'learned'
            ORDER BY id
        ''')
        learned_caps = cursor.fetchall()

        if learned_caps:
            print("\nLearned Capability Evidence:")
            print("-"*80)
            for name, evidence in learned_caps:
                print(f"\n  {name}:")
                print(f"    → {evidence}")
            print("-"*80)

    def analyze_capability_needs(self, issue_description: str):
        """Use LLM to analyze what capabilities are needed for an issue"""
        if not self.use_llm or not self.openai_client:
            return "LLM analysis not available (set use_llm=True and configure API keys)"

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a network operations expert. Analyze network issues and suggest what capabilities an agent would need to handle them."},
                    {"role": "user", "content": f"Network issue: {issue_description}\n\nWhat capabilities would a network agent need to handle this? List 2-3 specific capabilities."}
                ],
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM analysis error: {str(e)}"

    def get_network_summary(self):
        """Provide summary of monitored network"""
        cursor = self.memory_db.execute('''
            SELECT device_type, COUNT(*) FROM devices GROUP BY device_type
        ''')
        device_summary = cursor.fetchall()

        cursor = self.memory_db.execute('''
            SELECT status, COUNT(*) FROM devices GROUP BY status
        ''')
        status_summary = cursor.fetchall()

        summary = f"""
📊 Network Summary for {self.name}:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Devices by Type:
{chr(10).join(f"  • {dtype}: {count}" for dtype, count in device_summary) if device_summary else "  • No devices registered"}

Devices by Status:
{chr(10).join(f"  • {status}: {count}" for status, count in status_summary) if status_summary else "  • No devices registered"}

Total Devices: {self.get_device_count()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        return summary


# Demo usage
def demo_self_registering_network_agent(use_llm=False):
    """Demonstrate self-registering network agent"""
    print("🚀 Self-Registering Network Agent Demo")
    print("=" * 60)

    if use_llm:
        print("\n🤖 LLM Analysis Mode: ENABLED")
        print("   Using OpenAI API for capability recommendations")
    else:
        print("\n📋 Standard Mode (pass use_llm=True for LLM analysis)")

    print("=" * 60)

    # Create network agent with fixed ID for demo purposes
    # In production, each agent would have a unique ID
    agent = SelfRegisteringNetworkAgent(
        base_capabilities=[
            "device_monitoring",
            "configuration_management",
            "event_logging"
        ],
        use_llm=use_llm
    )

    # Override with fixed ID for clean demo
    old_db_path = f'network_agent_{agent.agent_id}_memory.db'
    old_db = agent.memory_db
    agent.agent_id = "demo_agent_003"
    agent.name = "NetAgent-demo_003"

    # Close and remove temp database
    old_db.close()
    import os
    if os.path.exists(old_db_path):
        os.remove(old_db_path)

    # Reconnect to fixed database
    agent.memory_db = sqlite3.connect(f'network_agent_{agent.agent_id}_memory.db')

    # Recreate tables with fixed ID
    agent.memory_db.execute('''CREATE TABLE IF NOT EXISTS memories
                             (id TEXT, content TEXT, type TEXT, timestamp TEXT)''')
    agent.memory_db.execute('''CREATE TABLE IF NOT EXISTS devices
                             (device_id TEXT PRIMARY KEY, hostname TEXT,
                              ip_address TEXT, device_type TEXT, status TEXT,
                              last_seen TEXT)''')
    agent.memory_db.execute('''CREATE TABLE IF NOT EXISTS capabilities
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         capability_name TEXT,
         learned_date TEXT,
         evidence TEXT,
         source TEXT)''')

    # Re-register with fixed ID
    for cap in agent.capabilities:
        agent.memory_db.execute(
            'INSERT INTO capabilities (capability_name, learned_date, evidence, source) VALUES (?, ?, ?, ?)',
            (cap, datetime.now().isoformat(), "Base capability on initialization", "system")
        )
    agent.memory_db.commit()

    # Agent introduces itself
    print("\n👋 Initial self-introduction:")
    print(agent.introduce_self())

    # Register network devices
    print(f"\n📡 {agent.name} is registering network devices...")
    devices = [
        ("R1", "core-router-1", "10.10.10.1", "Router", "online"),
        ("SW1", "access-switch-1", "10.10.20.1", "Switch", "online"),
        ("FW1", "edge-firewall-1", "10.10.30.1", "Firewall", "online"),
        ("R2", "core-router-2", "10.10.10.2", "Router", "degraded"),
        ("SW2", "access-switch-2", "10.10.20.2", "Switch", "maintenance")
    ]

    for device_id, hostname, ip_address, device_type, status in devices:
        agent.register_device(device_id, hostname, ip_address, device_type, status)

    # Log network events
    print(f"\n📝 {agent.name} is logging network events...")
    events = [
        ("R1", "interface_down", "critical", "Interface GigabitEthernet0/1 went down"),
        ("SW1", "high_cpu", "warning", "CPU utilization reached 85%"),
        ("R1", "interface_up", "info", "Interface GigabitEthernet0/1 came back up"),
        ("FW1", "policy_violation", "critical", "Unauthorized access attempt blocked")
    ]

    for device_id, event_type, severity, description in events:
        agent.log_network_event(device_id, event_type, severity, description)

    # Simulate learning from successful operations
    print(f"\n🎓 {agent.name} is learning new capabilities from network operations...")
    learning_scenarios = [
        ("bgp_troubleshooting", "Successfully diagnosed and resolved BGP neighbor issue on R1"),
        ("automated_failover", "Implemented automatic failover between R1 and R2"),
        ("security_analysis", "Detected and mitigated security threat on FW1"),
        ("performance_optimization", "Optimized switch configurations reducing latency by 30%")
    ]

    for capability, evidence in learning_scenarios:
        agent.learn_capability(capability, evidence)

    # Agent introduces itself again with new capabilities
    print("\n👋 Updated self-introduction with learned capabilities:")
    print(agent.introduce_self())

    # Show network summary
    print("\n📊 Network monitoring summary:")
    print(agent.get_network_summary())

    # Optional: LLM analysis of future capability needs
    if use_llm and agent.openai_client:
        print("\n\n🤖 LLM Analysis: Future Capability Recommendations")
        print("=" * 80)

        future_issue = "We need to implement automated network topology discovery and map device dependencies across multi-vendor environments"
        print(f"\nScenario: {future_issue}\n")

        analysis = agent.analyze_capability_needs(future_issue)
        print("Recommended capabilities:")
        print(analysis)
        print("=" * 80)

    print("\n✅ Self-registering network agent demo completed!")
    print(f"   Database file: network_agent_{agent.agent_id}_memory.db")
    print(f"   Open the .db file to view the capabilities table!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # Check if user wants LLM analysis
    use_llm = "--llm" in sys.argv or "-l" in sys.argv

    demo_self_registering_network_agent(use_llm=use_llm)
