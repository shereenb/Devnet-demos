#!/bin/bash
# Clean up all database files before recording demo

echo "Removing all .db files..."
rm -f *.db

echo "✓ All database files removed"
echo ""
echo "Ready for clean demo recording!"
echo "Run the demos and they will create fresh database files:"
echo "  - Demo 1: memory_network_agent_001_*.db (3 files)"
echo "  - Demo 2: network_benchmark_*.db (2 files)"
echo "  - Demo 3: network_agent_demo_agent_003_memory.db (1 file)"
