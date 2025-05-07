"""
Transaction categorization using zero-shot classification with Hugging Face Transformers.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import torch
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionCategorizer:
    """
    Categorizes bank transactions using zero-shot classification.
    """
    
    # Default transaction categories
    DEFAULT_CATEGORIES = [
        "Groceries",
        "Dining",
        "Transportation",
        "Entertainment",
        "Shopping",
        "Housing",
        "Utilities",
        "Healthcare",
        "Education",
        "Travel",
        "Income",
        "Transfer",
        "Subscription",
        "Investment",
        "Withdrawal",
        "Deposit",
        "Other"
    ]
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli", categories: List[str] = None):
        """
        Initialize the transaction categorizer.
        
        Args:
            model_name: HuggingFace model name for zero-shot classification
            categories: List of transaction categories. If None, default categories are used.
        """
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.model_name = model_name
        
        # Initialize pipeline - will be lazy-loaded when needed
        self._classifier = None
        
    @property
    def classifier(self):
        """Lazy load the classification pipeline when first needed."""
        if self._classifier is None:
            logger.info(f"Initializing zero-shot classification pipeline with model: {self.model_name}")
            try:
                # Check for GPU availability
                device = 0 if torch.cuda.is_available() else -1
                self._classifier = pipeline(
                    "zero-shot-classification", 
                    model=self.model_name, 
                    device=device
                )
                logger.info("Classification pipeline initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing classifier: {str(e)}")
                # Fallback to a simple rule-based approach if model fails
                self._classifier = "rule-based"
                logger.warning("Falling back to rule-based categorization")
        
        return self._classifier
    
    def categorize_transaction(self, description: str, amount: float) -> Dict[str, Any]:
        """
        Categorize a single transaction.
        
        Args:
            description: The transaction description
            amount: The transaction amount
            
        Returns:
            Dictionary with category and confidence score
        """
        if not description:
            return {"category": "Other", "confidence": 1.0}
            
        # Special case for income (large positive amounts)
        if amount > 1000 and amount > 0:
            return {"category": "Income", "confidence": 0.9}
            
        # Handle different classification methods
        if self.classifier == "rule-based":
            return self._rule_based_categorization(description)
        else:
            return self._model_based_categorization(description)
    
    def _rule_based_categorization(self, description: str) -> Dict[str, Any]:
        """
        Rule-based fallback categorization.
        
        Args:
            description: The transaction description
            
        Returns:
            Dictionary with category and confidence score
        """
        description = description.lower()
        
        # Simple keyword matching
        category_keywords = {
            "Groceries": ["grocery", "supermarket", "food", "market", "wholefood", "trader joe", "kroger", "safeway"],
            "Dining": ["restaurant", "cafe", "coffee", "dining", "doordash", "grubhub", "ubereats", "seamless"],
            "Transportation": ["uber", "lyft", "taxi", "transport", "transit", "metro", "subway", "train", "gas", "parking"],
            "Entertainment": ["movie", "cinema", "theater", "netflix", "spotify", "hulu", "disney", "ticket", "concert"],
            "Shopping": ["amazon", "walmart", "target", "store", "shop", "retail", "purchase", "buy"],
            "Housing": ["rent", "mortgage", "housing", "apartment", "condo", "lease"],
            "Utilities": ["utility", "water", "electricity", "power", "gas", "internet", "wifi", "phone", "bill"],
            "Healthcare": ["doctor", "pharmacy", "medical", "health", "dental", "hospital", "clinic", "insurance"],
            "Subscription": ["subscription", "monthly", "annual", "recurring"],
            "Withdrawal": ["withdrawal", "atm", "cash"],
            "Deposit": ["deposit", "direct deposit", "credit"],
            "Transfer": ["transfer", "zelle", "venmo", "paypal", "wire"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in description for keyword in keywords):
                return {"category": category, "confidence": 0.7}
                
        # Default category
        return {"category": "Other", "confidence": 0.5}
    
    def _model_based_categorization(self, description: str) -> Dict[str, Any]:
        """
        Model-based categorization using zero-shot classification.
        
        Args:
            description: The transaction description
            
        Returns:
            Dictionary with category and confidence score
        """
        try:
            # Format the query for better results
            query = f"Transaction: {description}"
            
            # Get prediction from model
            result = self.classifier(query, self.categories, multi_label=False)
            
            # Extract top predicted category and score
            top_category = result["labels"][0]
            confidence = result["scores"][0]
            
            return {"category": top_category, "confidence": confidence}
            
        except Exception as e:
            logger.error(f"Error in model-based categorization: {str(e)}")
            # Fallback to rule-based if model fails
            return self._rule_based_categorization(description)
    
    def categorize_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Categorize a list of transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of transactions with added category information
        """
        categorized = []
        
        for transaction in transactions:
            description = transaction.get("description", "")
            amount = transaction.get("amount", 0.0)
            
            # Get category info
            category_info = self.categorize_transaction(description, amount)
            
            # Add category info to transaction
            transaction_with_category = transaction.copy()
            transaction_with_category["category"] = category_info["category"]
            transaction_with_category["category_confidence"] = category_info["confidence"]
            
            categorized.append(transaction_with_category)
            
        return categorized