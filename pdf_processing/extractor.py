import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import pdfplumber
import pandas as pd

class BankStatementExtractor:
    """Class for extracting transaction data from bank statement PDFs."""
    
    def __init__(self, bank_type: str = None):
        """
        Initialize the extractor.
        
        Args:
            bank_type: Type of bank statement (e.g., 'chase', 'bofa', 'wells_fargo').
                       If provided, will use bank-specific extraction logic.
        """
        self.bank_type = bank_type
        # Dictionary of supported banks and their extraction methods
        self.extraction_methods = {
            'generic': self._extract_generic,
            'chase': self._extract_chase,
            'bofa': self._extract_bofa,
            'wells_fargo': self._extract_wells_fargo,
            # Add more banks as needed
        }
    
    def extract(self, pdf_path: str) -> pd.DataFrame:
        """
        Extract transactions from a bank statement PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DataFrame with transaction data
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if self.bank_type and self.bank_type in self.extraction_methods:
            return self.extraction_methods[self.bank_type](pdf_path)
        else:
            # Use generic extraction if bank type not specified or not supported
            return self.extraction_methods['generic'](pdf_path)
    
    def _extract_generic(self, pdf_path: str) -> pd.DataFrame:
        """
        Generic extraction method that attempts to work with most bank statements.
        Uses a combination of pdfplumber for text extraction and table detection.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DataFrame with transaction data
        """
        transactions = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Try to extract tables first
                tables = page.extract_tables()
                
                if tables:
                    for table in tables:
                        # Process each table
                        if self._is_transaction_table(table):
                            parsed_rows = self._parse_transaction_table(table)
                            transactions.extend(parsed_rows)
                else:
                    # If no tables found, try to extract text and parse transactions
                    text = page.extract_text()
                    parsed_txns = self._parse_transaction_text(text)
                    if parsed_txns:
                        transactions.extend(parsed_txns)
        
        # Convert to DataFrame and clean up
        if transactions:
            df = pd.DataFrame(transactions)
            return self._clean_transaction_df(df)
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['date', 'description', 'amount', 'type'])
    
    def _extract_chase(self, pdf_path: str) -> pd.DataFrame:
        """Chase Bank specific extraction logic."""
        # Implementation for Chase Bank statements
        # This would include specific patterns and layouts for Chase
        pass
    
    def _extract_bofa(self, pdf_path: str) -> pd.DataFrame:
        """Bank of America specific extraction logic."""
        # Implementation for Bank of America statements
        pass
    
    def _extract_wells_fargo(self, pdf_path: str) -> pd.DataFrame:
        """Wells Fargo specific extraction logic."""
        # Implementation for Wells Fargo statements
        pass
    
    def _is_transaction_table(self, table: List[List[str]]) -> bool:
        """
        Check if a table contains transaction data.
        
        Args:
            table: A table extracted from a PDF
            
        Returns:
            True if table appears to contain transaction data
        """
        if not table or len(table) < 2:  # Need at least header + one row
            return False
        
        # Check if header contains typical transaction column names
        header = [str(cell).lower() if cell else '' for cell in table[0]]
        transaction_keywords = ['date', 'description', 'amount', 'balance', 'debit', 'credit', 
                               'withdrawal', 'deposit', 'transaction', 'details']
        
        # If at least 2 keywords are in the header, consider it a transaction table
        matches = sum(1 for keyword in transaction_keywords if any(keyword in col for col in header))
        return matches >= 2
    
    def _parse_transaction_table(self, table: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Parse a transaction table into a list of transaction dictionaries.
        
        Args:
            table: A table extracted from a PDF
            
        Returns:
            List of transaction dictionaries
        """
        if not table or len(table) < 2:
            return []
        
        # Try to identify column meanings from header
        header = [str(cell).lower() if cell else '' for cell in table[0]]
        
        # Find indices for key columns
        date_idx = next((i for i, col in enumerate(header) if 'date' in col), None)
        desc_idx = next((i for i, col in enumerate(header) 
                         if any(kw in col for kw in ['description', 'details', 'transaction'])), None)
        
        # Look for amount columns (could be single amount or separate debit/credit)
        amount_idx = next((i for i, col in enumerate(header) if 'amount' in col), None)
        debit_idx = next((i for i, col in enumerate(header) 
                         if any(kw in col for kw in ['debit', 'withdrawal', 'payment'])), None)
        credit_idx = next((i for i, col in enumerate(header)
                          if any(kw in col for kw in ['credit', 'deposit'])), None)
        
        # Skip header row
        transactions = []
        for row in table[1:]:
            # Skip empty rows
            if not any(cell for cell in row):
                continue
                
            transaction = {}
            
            # Extract date
            if date_idx is not None and date_idx < len(row) and row[date_idx]:
                transaction['date'] = self._parse_date(row[date_idx])
            
            # Extract description
            if desc_idx is not None and desc_idx < len(row) and row[desc_idx]:
                transaction['description'] = str(row[desc_idx]).strip()
            
            # Extract amount (either from amount column or from debit/credit columns)
            if amount_idx is not None and amount_idx < len(row) and row[amount_idx]:
                amount_str = str(row[amount_idx]).strip()
                amount = self._parse_amount(amount_str)
                transaction['amount'] = amount
                # Determine transaction type based on sign
                transaction['type'] = 'debit' if amount < 0 else 'credit'
            elif debit_idx is not None and credit_idx is not None:
                # Handle separate debit/credit columns
                debit_value = self._parse_amount(str(row[debit_idx]).strip()) if debit_idx < len(row) and row[debit_idx] else 0
                credit_value = self._parse_amount(str(row[credit_idx]).strip()) if credit_idx < len(row) and row[credit_idx] else 0
                
                if debit_value != 0:
                    transaction['amount'] = -abs(debit_value)  # Ensure negative for debits
                    transaction['type'] = 'debit'
                elif credit_value != 0:
                    transaction['amount'] = abs(credit_value)  # Ensure positive for credits
                    transaction['type'] = 'credit'
            
            # Only add if we have at least date and amount
            if 'date' in transaction and 'amount' in transaction:
                transactions.append(transaction)
        
        return transactions
    
    def _parse_transaction_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse transaction data from text when tables aren't available.
        Uses regex patterns to identify transaction data.
        
        Args:
            text: Extracted text from PDF
            
        Returns:
            List of transaction dictionaries
        """
        if not text:
            return []
        
        transactions = []
        
        # Common pattern for transactions: date followed by description and amount
        # This is a simplified example - real implementation would be more robust
        transaction_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})[\s\t]+([A-Za-z0-9\s&\.,#-]+)[\s\t]+([-+]?\$?\s?\d{1,3}(?:,\d{3})*\.\d{2})'
        
        matches = re.finditer(transaction_pattern, text)
        for match in matches:
            date_str, desc, amount_str = match.groups()
            
            transaction = {
                'date': self._parse_date(date_str),
                'description': desc.strip(),
                'amount': self._parse_amount(amount_str),
            }
            
            # Determine transaction type based on amount
            transaction['type'] = 'debit' if transaction['amount'] < 0 else 'credit'
            
            transactions.append(transaction)
        
        return transactions
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string into datetime object.
        Handles various date formats commonly found in bank statements.
        
        Args:
            date_str: Date string from bank statement
            
        Returns:
            Parsed datetime object or None if parsing fails
        """
        if not date_str:
            return None
            
        date_str = str(date_str).strip()
        
        # Try various date formats
        date_formats = [
            '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y', '%d-%m-%Y',
            '%m/%d/%y', '%m-%d-%y', '%d/%m/%y', '%d-%m-%y',
            '%b %d, %Y', '%B %d, %Y'
        ]
        
        for date_format in date_formats:
            try:
                return datetime.strptime(date_str, date_format)
            except ValueError:
                continue
                
        # If all formats fail, return None
        return None
    
    def _parse_amount(self, amount_str: str) -> float:
        """
        Parse amount string into float value.
        Handles various formats and determines sign based on format or symbols.
        
        Args:
            amount_str: Amount string from bank statement
            
        Returns:
            Float amount (negative for debits, positive for credits)
        """
        if not amount_str:
            return 0.0
            
        # Remove currency symbols and spaces
        amount_str = str(amount_str).strip()
        amount_str = re.sub(r'[^\d.,\-+()]', '', amount_str)
        
        # Handle parentheses for negative numbers
        if '(' in amount_str and ')' in amount_str:
            amount_str = amount_str.replace('(', '-').replace(')', '')
            
        # Handle cases where minus sign might be separated
        if amount_str.startswith('-') or amount_str.startswith('−'):
            is_negative = True
            amount_str = amount_str[1:]
        else:
            is_negative = False
            
        # Remove any remaining non-numeric chars except decimal
        amount_str = re.sub(r'[^\d.]', '', amount_str)
        
        try:
            amount = float(amount_str)
            return -amount if is_negative else amount
        except ValueError:
            return 0.0
    
    def _clean_transaction_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the transaction DataFrame.
        
        Args:
            df: DataFrame with raw transaction data
            
        Returns:
            Cleaned DataFrame with standardized columns
        """
        # Create copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Ensure required columns exist
        required_columns = ['date', 'description', 'amount', 'type']
        for col in required_columns:
            if col not in cleaned_df.columns:
                cleaned_df[col] = None
        
        # Convert date strings to datetime if needed
        if cleaned_df['date'].dtype == 'object':
            cleaned_df['date'] = pd.to_datetime(cleaned_df['date'], errors='coerce')
            
        # Sort by date
        cleaned_df = cleaned_df.sort_values('date')
        
        # Reset index
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        return cleaned_df