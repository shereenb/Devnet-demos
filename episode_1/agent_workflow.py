
class Agent:
    def __init__(self, name, specialty):
        self.name = name
        self.specialty = specialty
        self.memory = []
    
    def think(self, task):
        print(f" {self.name} ({self.specialty}): {task}")
        
        if "research" in self.specialty.lower():
            result = {"findings": f"{task} - 15% growth, strong drivers"}
        elif "summary" in self.specialty.lower():
            result = {"recommendation": "Strategic investment recommended"}
        else:
            result = {"analysis": f"Processed {task}"}
            
        self.memory.append(result)
        return result

# Multi-agent workflow
researcher = Agent("DataBot", "Research Specialist") 
strategist = Agent("StrategyBot", "Executive Summary")

task = "renewable energy market opportunities"
research = researcher.think(task)
strategy = strategist.think(f"Based on: {research}")

print(f"Workflow: {len(researcher.memory + strategist.memory)} structured outputs")
