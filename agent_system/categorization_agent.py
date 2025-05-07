"""
Transaction Categorization Agent for Bank Statement Analyzer.
Responsible for assigning categories to transactions using zero-shot classification.
"""

import logging
from typing import Dict, List, Any, AsyncGenerator

from agent_system import BaseAgent
from data_processing.categorizer import TransactionCategorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionCategorizationAgent(BaseAgent):
    """
    Agent responsible for categorizing transactions using zero-shot classification.
    """
    
    def __init__(self, name: str = "Categorizer", description: str = None, 
                 model_name: str = "facebook/bart-large-mnli", categories: List[str] = None):
        """
        Initialize the Transaction Categorization Agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent's purpose
            model_name: HuggingFace model name for zero-shot classification
            categories: Custom categories to use instead of defaults
        """
        super().__init__(
            name=name, 
            description=description or "Categorizes transactions using zero-shot classification"
        )
        
        # Initialize the categorizer
        self.categorizer = TransactionCategorizer(model_name=model_name, categories=categories)
    
    async def _process(self, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Categorize transactions.
        
        Args:
            context: Input context containing cleaned transaction data
            
        Yields:
            Categorized transaction data
        """
        # Get cleaned transactions from context
        transactions = context.get("cleaned_transactions")
        if not transactions:
            yield {
                "status": "error",
                "message": "No cleaned transactions found in context"
            }
            return
            
        # Categorize the transactions
        logger.info(f"Categorizing {len(transactions)} transactions")
        try:
            # Apply categorization
            categorized_transactions = self.categorizer.categorize_transactions(transactions)
            
            # Count transactions by category
            category_counts = {}
            for transaction in categorized_transactions:
                category = transaction.get("category", "Other")
                category_counts[category] = category_counts.get(category, 0) + 1
                
            # Calculate total spend by category
            category_spend = {}
            for transaction in categorized_transactions:
                if transaction.get("type") == "debit":  # Only count outgoing transactions
                    category = transaction.get("category", "Other")
                    amount = abs(transaction.get("amount", 0))
                    category_spend[category] = category_spend.get(category, 0) + amount
                    
            # Update context for next agents in the pipeline
            context_updates = {
                "categorized_transactions": categorized_transactions,
                "category_counts": category_counts,
                "category_spend": category_spend
            }
            
            # Yield the results
            yield {
                "status": "success",
                "message": f"Successfully categorized {len(categorized_transactions)} transactions",
                "category_counts": category_counts,
                "category_spend": category_spend,
                "state_updates": {
                    "categorized_transactions": categorized_transactions,
                    "category_counts": category_counts,
                    "category_spend": category_spend
                },
                "context_updates": context_updates
            }
            
        except Exception as e:
            logger.error(f"Error categorizing transactions: {str(e)}")
            yield {
                "status": "error",
                "message": f"Failed to categorize transactions: {str(e)}"
            }