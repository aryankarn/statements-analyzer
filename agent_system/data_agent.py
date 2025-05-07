"""
Data Processing Agent for Bank Statement Analyzer.
Responsible for cleaning and standardizing transaction data.
"""

import logging
from typing import Dict, List, Any, AsyncGenerator
import pandas as pd

from agent_system import BaseAgent
from data_processing.processor import process_transactions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessingAgent(BaseAgent):
    """
    Agent responsible for cleaning and standardizing transaction data.
    """
    
    def __init__(self, name: str = "DataProcessor", description: str = None):
        """
        Initialize the Data Processing Agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent's purpose
        """
        super().__init__(
            name=name, 
            description=description or "Cleans and standardizes transaction data"
        )
    
    async def _process(self, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process the transaction data.
        
        Args:
            context: Input context containing transaction data
            
        Yields:
            Cleaned transaction data
        """
        # Get transactions from context
        transactions = context.get("transactions")
        if not transactions:
            yield {
                "status": "error",
                "message": "No transactions found in context"
            }
            return
            
        # Process the transactions
        logger.info(f"Processing {len(transactions)} transactions")
        try:
            # Clean and standardize the transaction data
            processed_df = process_transactions(transactions)
            
            # Convert back to dict for easier serialization
            cleaned_transactions = processed_df.to_dict(orient="records")
            
            # Calculate some basic statistics
            stats = {
                "total_transactions": len(cleaned_transactions),
                "total_credits": sum(1 for t in cleaned_transactions if t.get("type") == "credit"),
                "total_debits": sum(1 for t in cleaned_transactions if t.get("type") == "debit"),
                "date_range": [
                    processed_df["date"].min().strftime("%Y-%m-%d") if not processed_df.empty else None,
                    processed_df["date"].max().strftime("%Y-%m-%d") if not processed_df.empty else None
                ]
            }
            
            # Update context for next agents in the pipeline
            context_updates = {
                "cleaned_transactions": cleaned_transactions,
                "transaction_stats": stats
            }
            
            # Yield the results
            yield {
                "status": "success",
                "message": f"Successfully processed {len(cleaned_transactions)} transactions",
                "transaction_stats": stats,
                "state_updates": {
                    "cleaned_transactions": cleaned_transactions,
                    "transaction_stats": stats
                },
                "context_updates": context_updates
            }
            
        except Exception as e:
            logger.error(f"Error processing transactions: {str(e)}")
            yield {
                "status": "error",
                "message": f"Failed to process transactions: {str(e)}"
            }