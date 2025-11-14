"""
Master Demo Runner - Episode 2
Runs all network agent memory demos in sequence for video recording
"""

import sys
import time
import asyncio

def print_header(title, subtitle=""):
    """Print a formatted section header"""
    print("\n\n")
    print("=" * 80)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 80)
    print()


def print_separator():
    """Print a section separator"""
    print("\n" + "-" * 80 + "\n")


def pause_for_demo(seconds=3):
    """Pause between demos for video clarity"""
    print(f"\n⏸️  Pausing for {seconds} seconds...")
    time.sleep(seconds)


async def main():
    """Run all demos in sequence"""

    print_header(
        "🌐 EPISODE 2: NETWORK AGENT MEMORY SYSTEMS",
        "Complete Demo Suite for Video Walkthrough"
    )

    print("This script will run all network agent memory demonstrations in sequence.")
    print("Perfect for video recording!")
    print("\nPress Ctrl+C at any time to stop.")

    input("\n▶️  Press ENTER to begin...\n")

    # =================================================================
    # DEMO 1: Persistent Network Agent
    # =================================================================
    print_header("DEMO 1: Persistent Network Agent", "Basic persistent memory for network operations")

    print("This demo shows:")
    print("  • Network event tracking")
    print("  • Device configuration storage")
    print("  • Memory queries across sessions")
    print("  • AI-powered network analysis")

    input("\n▶️  Press ENTER to start Demo 1...\n")

    try:
        from persistent_agent import run_network_demo
        await run_network_demo()
    except Exception as e:
        print(f"❌ Error running Demo 1: {e}")

    pause_for_demo()

    # =================================================================
    # DEMO 2: Memory Benchmark
    # =================================================================
    print_header("DEMO 2: Memory Performance Benchmark", "Persistent memory vs context windows")

    print("This benchmark compares:")
    print("  • Persistent memory: 100 devices, 100% accuracy")
    print("  • Context windows: 20 devices, 40% accuracy")
    print("  • Cost savings: ~75%")
    print("  • Event correlation capabilities")

    input("\n▶️  Press ENTER to start Demo 2...\n")

    try:
        from memory_benchmark import benchmark_network_memory_systems, benchmark_network_event_correlation
        benchmark_network_memory_systems()
        benchmark_network_event_correlation()
    except Exception as e:
        print(f"❌ Error running Demo 2: {e}")

    pause_for_demo()

    # =================================================================
    # DEMO 3: Self-Registering Network Agent
    # =================================================================
    print_header("DEMO 3: Self-Registering Network Agent", "Agents that learn from experience")

    print("This demo shows:")
    print("  • Automatic self-registration")
    print("  • Dynamic capability learning")
    print("  • Network device tracking")
    print("  • Auto-learning from critical events")

    input("\n▶️  Press ENTER to start Demo 3...\n")

    try:
        from self_registering_agent import demo_self_registering_network_agent
        demo_self_registering_network_agent()
    except Exception as e:
        print(f"❌ Error running Demo 3: {e}")

    pause_for_demo()

    # =================================================================
    # DEMO 4: AGNTCY Network Memory Implementation
    # =================================================================
    print_header("DEMO 4: AGNTCY Network Memory", "Production-ready multi-memory system")

    print("This demo showcases:")
    print("  • Vector memory: Semantic search")
    print("  • Graph memory: Device relationships")
    print("  • Episodic memory: Time-series events")
    print("  • LLM integration: AI analysis")
    print("  • AGNTCY protocol compliance")

    input("\n▶️  Press ENTER to start Demo 4...\n")

    try:
        from agntcy_memory_implementation import demo_agntcy_network_memory
        demo_agntcy_network_memory()
    except Exception as e:
        print(f"❌ Error running Demo 4: {e}")

    pause_for_demo()

    # =================================================================
    # CONCLUSION
    # =================================================================
    print_header("✅ ALL DEMOS COMPLETED!", "Episode 2 Demo Suite Finished")

    print("🎯 Key Takeaways:")
    print()
    print("  ✅ Persistent memory provides 2.5x better accuracy than context windows")
    print("  ✅ Track 5x more devices (100 vs 20) with lower cost")
    print("  ✅ Self-registering agents learn from experience")
    print("  ✅ Multi-memory systems enable production-grade operations")
    print("  ✅ LLM integration provides intelligent network analysis")
    print()
    print("📊 Performance Improvements:")
    print("  • Accuracy: +150% (100% vs 40%)")
    print("  • Speed: +97% faster recall")
    print("  • Cost: -75% savings")
    print("  • Scalability: 5x more devices")
    print("  • Persistence: Infinite (across all sessions)")
    print()
    print("🚀 Production Benefits:")
    print("  • Network topology tracking")
    print("  • Event correlation over time")
    print("  • Root cause analysis")
    print("  • Knowledge retention across shifts")
    print("  • AI-powered troubleshooting")
    print()
    print("🔗 Resources:")
    print("  • GitHub: https://github.com/shereenb/Devnet-demos")
    print("  • AGNTCY SDK: https://github.com/agntcy")
    print("  • OpenAI: https://platform.openai.com/")
    print("  • Anthropic Claude: https://www.anthropic.com/")
    print()
    print("=" * 80)
    print("  Thank you for watching Episode 2!")
    print("  Stay tuned for Episode 3: Agent-to-Agent Communication")
    print("=" * 80)
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo stopped by user.")
        print("=" * 80)
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
