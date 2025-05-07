"""
PDF Processing Agent for Bank Statement Analyzer.
Responsible for extracting data from PDF bank statements.
"""

import os
import logging
from typing import Dict, List, Any, AsyncGenerator

from agent_system import BaseAgent
from pdf_processing.extractor import BankStatementExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessingAgent(BaseAgent):
    """
    Agent responsible for processing PDF bank statements and extracting transaction data.
    """
    
    def __init__(self, name: str = "PDFProcessor", description: str = None):
        """
        Initialize the PDF Processing Agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent's purpose
        """
        super().__init__(
            name=name, 
            description=description or "Extracts transaction data from PDF bank statements"
        )
        
        # Initialize the extractor for bank statements
        self.extractor = BankStatementExtractor()
    
    async def _process(self, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process the PDF bank statement.
        
        Args:
            context: Input context containing the path to the PDF file and optional bank type
            
        Yields:
            Extracted transaction data
        """
        # Get the PDF file path from context
        pdf_path = context.get("pdf_path")
        if not pdf_path:
            yield {
                "status": "error",
                "message": "PDF path not provided in context"
            }
            return
            
        # Check if the file exists
        if not os.path.exists(pdf_path):
            yield {
                "status": "error",
                "message": f"PDF file not found: {pdf_path}"
            }
            return
            
        # Get the bank type if available
        bank_type = context.get("bank_type")
        if bank_type:
            logger.info(f"Setting bank type to: {bank_type}")
            self.extractor.bank_type = bank_type
            
        # Extract transactions from the PDF
        logger.info(f"Extracting data from PDF: {pdf_path}")
        try:
            transactions_df = self.extractor.extract(pdf_path)
            
            # Convert to dict for easier JSON serialization
            transactions = transactions_df.to_dict(orient="records")
            
            # Update context for next agents in the pipeline
            context_updates = {
                "transactions": transactions,
                "bank_type": bank_type,
                "source_file": pdf_path
            }
            
            # Yield the results
            yield {
                "status": "success",
                "message": f"Successfully extracted {len(transactions)} transactions",
                "transactions_count": len(transactions),
                "state_updates": {
                    "transactions": transactions,
                    "pdf_processed": pdf_path,
                    "bank_type": bank_type
                },
                "context_updates": context_updates
            }
            
        except Exception as e:
            logger.error(f"Error extracting transactions: {str(e)}")
            yield {
                "status": "error",
                "message": f"Failed to extract transactions: {str(e)}"
            }