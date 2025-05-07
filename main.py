import os
import asyncio
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from agent_system import AgentPipeline
from agent_system.pdf_agent import PDFProcessingAgent
from agent_system.data_agent import DataProcessingAgent
from agent_system.categorization_agent import TransactionCategorizationAgent
from analysis.analyzer import AnalysisAgent
from storage.db_agent import DatabaseStorageAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BankStatementAnalyzer:
    """
    Main application for analyzing bank statements.
    Uses an agent-based approach to extract and analyze transaction data.
    """
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the Bank Statement Analyzer.
        
        Args:
            storage_path: Path to store analysis results. If None, uses a default path.
        """
        self.storage_path = storage_path or os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize agents
        self.pdf_agent = PDFProcessingAgent()
        self.data_agent = DataProcessingAgent()
        self.categorization_agent = TransactionCategorizationAgent()
        self.analysis_agent = AnalysisAgent()
        
        # Create a storage agent if SQLite is available
        try:
            self.storage_agent = DatabaseStorageAgent()
            self.has_storage = True
        except ImportError:
            logger.warning("SQLite support not available. Results will be stored as JSON files only.")
            self.storage_agent = None
            self.has_storage = False
        
        # Create the agent pipeline
        agents = [
            self.pdf_agent,
            self.data_agent,
            self.categorization_agent,
            self.analysis_agent
        ]
        
        # Add storage agent if available
        if self.has_storage:
            agents.append(self.storage_agent)
            
        self.pipeline = AgentPipeline(agents)
    
    async def analyze_statement(self, pdf_path: str, bank_type: str = None) -> Dict[str, Any]:
        """
        Analyze a bank statement PDF.
        
        Args:
            pdf_path: Path to the PDF file
            bank_type: Type of bank statement (optional)
            
        Returns:
            Dictionary with analysis results
        """
        # Create initial context
        context = {
            "pdf_path": pdf_path,
            "bank_type": bank_type,
            "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Process through agent pipeline
        logger.info(f"Starting analysis of {pdf_path}")
        results = {}
        
        async for result in self.pipeline.run(context):
            agent_name = result.get("agent")
            agent_result = result.get("result", {})
            
            # Store results from each agent
            results[agent_name] = agent_result
            
            # Log progress
            status = agent_result.get("status", "unknown")
            message = agent_result.get("message", "No message")
            logger.info(f"Agent {agent_name}: {status} - {message}")
            
        # Get final analysis results
        analysis_results = results.get("Analyzer", {}).get("state_updates", {}).get("analysis_results", {})
        insights = results.get("Analyzer", {}).get("state_updates", {}).get("financial_insights", [])
        
        # Save results to file
        output_file = os.path.join(self.storage_path, f"{context['analysis_id']}.json")
        with open(output_file, 'w') as f:
            json.dump({
                "context": context,
                "results": results,
                "analysis": analysis_results,
                "insights": insights
            }, f, indent=2)
            
        logger.info(f"Analysis complete. Results saved to {output_file}")
        
        # Return the final results
        return {
            "analysis_id": context["analysis_id"],
            "pdf_path": pdf_path,
            "timestamp": context["timestamp"],
            "analysis_results": analysis_results,
            "insights": insights,
            "output_file": output_file
        }
    
    def print_insights(self, insights: List[Dict[str, Any]]):
        """
        Print formatted insights to the console.
        
        Args:
            insights: List of insight dictionaries
        """
        if not insights:
            print("No insights available.")
            return
            
        print("\n===== FINANCIAL INSIGHTS =====\n")
        
        for i, insight in enumerate(insights, 1):
            title = insight.get("title", "Insight")
            description = insight.get("description", "No description available.")
            
            print(f"{i}. {title}")
            print(f"   {description}")
            print()
            
        print("============================\n")


async def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Bank Statement Analyzer")
    parser.add_argument("pdf_path", help="Path to the bank statement PDF file")
    parser.add_argument("--bank", help="Type of bank statement (e.g., 'chase', 'bofa')", default=None)
    parser.add_argument("--output", help="Directory to store results", default=None)
    args = parser.parse_args()
    
    analyzer = BankStatementAnalyzer(storage_path=args.output)
    results = await analyzer.analyze_statement(args.pdf_path, bank_type=args.bank)
    
    # Print insights to console
    analyzer.print_insights(results.get("insights", []))
    
    print(f"\nFull analysis results saved to: {results['output_file']}")


if __name__ == "__main__":
    asyncio.run(main())