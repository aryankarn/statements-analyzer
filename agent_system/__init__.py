"""
Main module for Bank Statement Analyzer Agent System.
Implements an agentic approach for analyzing bank statements using a multi-agent architecture.
"""

import os
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, description: str = None):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent's purpose
        """
        self.name = name
        self.description = description or f"{name} Agent"
        self.state: Dict[str, Any] = {}
        
    async def run(self, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the agent on the given context.
        
        Args:
            context: Input context containing data for the agent to process
            
        Yields:
            Results from the agent processing
        """
        logger.info(f"Agent {self.name} started processing")
        
        try:
            async for result in self._process(context):
                yield result
        except Exception as e:
            logger.error(f"Error in agent {self.name}: {str(e)}")
            yield {"status": "error", "agent": self.name, "error": str(e)}
        
        logger.info(f"Agent {self.name} finished processing")
    
    async def _process(self, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Internal processing method to be implemented by subclasses.
        
        Args:
            context: Input context containing data for the agent to process
            
        Yields:
            Results from the agent processing
        """
        raise NotImplementedError("Subclasses must implement this method")


class AgentPipeline:
    """Orchestrates a pipeline of agents."""
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize the agent pipeline.
        
        Args:
            agents: List of agents in the pipeline
        """
        self.agents = agents
        self.state: Dict[str, Any] = {}
        
    async def run(self, initial_context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the pipeline of agents.
        
        Args:
            initial_context: Initial context to provide to the first agent
            
        Yields:
            Results from each agent in the pipeline
        """
        context = initial_context.copy()
        
        for agent in self.agents:
            logger.info(f"Pipeline running agent: {agent.name}")
            
            # Update context with current pipeline state
            context.update({"pipeline_state": self.state})
            
            # Run the agent
            async for result in agent.run(context):
                # Update pipeline state with agent results
                if "state_updates" in result:
                    self.state.update(result["state_updates"])
                
                # Update context for next agent
                context.update(result.get("context_updates", {}))
                
                # Yield the result
                yield {
                    "agent": agent.name,
                    "result": result
                }


class ParallelAgentExecutor:
    """Executes multiple agents in parallel."""
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize the parallel agent executor.
        
        Args:
            agents: List of agents to execute in parallel
        """
        self.agents = agents
        
    async def run(self, context: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run multiple agents in parallel on the same context.
        
        Args:
            context: Context to provide to all agents
            
        Returns:
            Dictionary with agent names as keys and lists of their results as values
        """
        import asyncio
        
        async def run_agent(agent):
            results = []
            async for result in agent.run(context):
                results.append(result)
            return agent.name, results
        
        tasks = [run_agent(agent) for agent in self.agents]
        results = await asyncio.gather(*tasks)
        
        return {name: agent_results for name, agent_results in results}