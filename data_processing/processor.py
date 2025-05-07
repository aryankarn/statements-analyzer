"""
Data processing utilities for bank statement transactions.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

def clean_transaction_descriptions(descriptions: List[str]) -> List[str]:
    """
    Clean and standardize transaction descriptions.
    
    Args:
        descriptions: List of raw transaction descriptions
        
    Returns:
        List of cleaned descriptions
    """
    cleaned = []
    
    for desc in descriptions:
        if not desc:
            cleaned.append("")
            continue
            
        # Convert to string if not already
        desc = str(desc).strip()
        
        # Remove excessive whitespace
        desc = re.sub(r'\s+', ' ', desc)
        
        # Remove common noise patterns in bank statements
        patterns_to_remove = [
            r'#\d+',  # Reference numbers like #12345
            r'REF\s*\d+',  # Reference IDs
            r'ID:\s*\d+',  # ID numbers
            r'TXN\s*\d+',  # Transaction IDs
            r'AUTH CODE:?.*?(?=\s|$)',  # Authorization codes
        ]
        
        for pattern in patterns_to_remove:
            desc = re.sub(pattern, '', desc)
            
        # Remove leading/trailing special characters
        desc = re.sub(r'^[^a-zA-Z0-9]+', '', desc)
        desc = re.sub(r'[^a-zA-Z0-9]+$', '', desc)
        
        # Capitalize first letter of each word for consistency
        desc = desc.title()
        
        cleaned.append(desc.strip())
    
    return cleaned

def standardize_dates(dates: List[Union[str, datetime, pd.Timestamp]]) -> List[datetime]:
    """
    Convert various date formats to standard datetime objects.
    
    Args:
        dates: List of dates in various formats
        
    Returns:
        List of standardized datetime objects
    """
    standardized = []
    
    for date in dates:
        if isinstance(date, (datetime, pd.Timestamp)):
            standardized.append(pd.Timestamp(date).to_pydatetime())
        elif isinstance(date, str):
            try:
                # Try to parse the string to datetime
                parsed = pd.to_datetime(date)
                standardized.append(parsed.to_pydatetime())
            except:
                # If parsing fails, append None
                standardized.append(None)
        else:
            standardized.append(None)
    
    return standardized

def normalize_transaction_amounts(amounts: List[Union[str, float, int]]) -> List[float]:
    """
    Normalize transaction amounts to float values.
    
    Args:
        amounts: List of amounts in various formats
        
    Returns:
        List of normalized float amounts
    """
    normalized = []
    
    for amount in amounts:
        if isinstance(amount, (float, int)):
            normalized.append(float(amount))
        elif isinstance(amount, str):
            # Remove currency symbols and commas
            cleaned = re.sub(r'[^\d.\-+()]', '', amount)
            
            # Handle parentheses as negative numbers
            if '(' in cleaned and ')' in cleaned:
                cleaned = cleaned.replace('(', '-').replace(')', '')
                
            try:
                normalized.append(float(cleaned))
            except:
                normalized.append(0.0)
        else:
            normalized.append(0.0)
    
    return normalized

def process_transactions(transactions: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Process a list of transaction dictionaries into a clean DataFrame.
    
    Args:
        transactions: List of transaction dictionaries
        
    Returns:
        Cleaned and processed DataFrame
    """
    if not transactions:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['date', 'description', 'amount', 'type'])
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Ensure required columns exist
    required_columns = ['date', 'description', 'amount', 'type']
    for col in required_columns:
        if col not in df.columns:
            if col == 'date':
                df[col] = None
            elif col == 'description':
                df[col] = ''
            elif col == 'amount':
                df[col] = 0.0
            elif col == 'type':
                df[col] = 'unknown'
    
    # Clean and standardize data
    if 'description' in df.columns:
        df['description'] = clean_transaction_descriptions(df['description'].tolist())
    
    if 'date' in df.columns:
        df['date'] = standardize_dates(df['date'].tolist())
    
    if 'amount' in df.columns:
        df['amount'] = normalize_transaction_amounts(df['amount'].tolist())
    
    # Derive transaction type if not present
    if 'type' not in df.columns or df['type'].isnull().any():
        df['type'] = df['amount'].apply(lambda x: 'credit' if x >= 0 else 'debit')
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['date', 'amount'])
    
    # Sort by date
    df = df.sort_values('date')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df