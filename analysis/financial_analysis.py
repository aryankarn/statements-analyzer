"""
Financial analysis utilities for bank statements.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

def calculate_spending_trends(transactions: List[Dict[str, Any]], 
                              date_column: str = 'date', 
                              amount_column: str = 'amount') -> Dict[str, Any]:
    """
    Calculate spending trends over time.
    
    Args:
        transactions: List of transaction dictionaries
        date_column: Name of the date column
        amount_column: Name of the amount column
        
    Returns:
        Dictionary with spending trend information
    """
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    if df.empty or date_column not in df.columns or amount_column not in df.columns:
        return {"error": "Missing required columns for trend analysis"}
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Filter to only include expenses (negative amounts)
    expenses_df = df[df[amount_column] < 0].copy()
    expenses_df['amount_abs'] = expenses_df[amount_column].abs()
    
    if expenses_df.empty:
        return {"warning": "No expense transactions found for trend analysis"}
    
    # Group by month and calculate total spending
    expenses_df['month'] = expenses_df[date_column].dt.to_period('M')
    monthly_spending = expenses_df.groupby('month')['amount_abs'].sum()
    
    # Calculate month-over-month change
    monthly_spending_dict = monthly_spending.to_dict()
    monthly_change = {}
    
    # Convert period keys to strings for JSON serialization
    monthly_spending_str = {str(k): float(v) for k, v in monthly_spending_dict.items()}
    
    # Calculate month-over-month percentage change
    previous_month = None
    for month in sorted(monthly_spending.index):
        if previous_month:
            current = monthly_spending[month]
            previous = monthly_spending[previous_month]
            if previous > 0:
                change_pct = ((current - previous) / previous) * 100
                monthly_change[str(month)] = float(change_pct)
        previous_month = month
    
    # Get top spending month
    if monthly_spending.empty:
        top_month = None
        top_month_amount = 0
    else:
        top_month_idx = monthly_spending.idxmax()
        top_month = str(top_month_idx)
        top_month_amount = float(monthly_spending[top_month_idx])
    
    return {
        "monthly_spending": monthly_spending_str,
        "monthly_change_percentage": monthly_change,
        "top_spending_month": top_month,
        "top_spending_month_amount": top_month_amount
    }

def identify_recurring_transactions(transactions: List[Dict[str, Any]], 
                                    date_column: str = 'date',
                                    description_column: str = 'description',
                                    amount_column: str = 'amount',
                                    similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Identify potential recurring transactions like subscriptions.
    
    Args:
        transactions: List of transaction dictionaries
        date_column: Name of the date column
        description_column: Name of the description column
        amount_column: Name of the amount column
        similarity_threshold: Threshold for description similarity
        
    Returns:
        List of potential recurring transaction groups
    """
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    if df.empty or date_column not in df.columns or description_column not in df.columns or amount_column not in df.columns:
        return []
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by similar descriptions and amounts
    recurring_candidates = {}
    
    # Simple approach: group by exact description match and similar amounts
    for _, row in df.iterrows():
        desc = row[description_column]
        amount = row[amount_column]
        date = row[date_column]
        
        # Skip empty descriptions
        if not desc:
            continue
            
        # Create a key based on description and rounded amount
        key = (desc, round(amount, 2))
        
        if key not in recurring_candidates:
            recurring_candidates[key] = []
            
        recurring_candidates[key].append({
            'date': date,
            'description': desc,
            'amount': amount
        })
    
    # Filter to find potential recurring transactions
    recurring_groups = []
    
    for (desc, amount), transactions in recurring_candidates.items():
        # Only consider groups with at least 2 transactions
        if len(transactions) >= 2:
            # Sort by date
            sorted_txns = sorted(transactions, key=lambda x: x['date'])
            
            # Calculate average time between transactions
            date_diffs = []
            for i in range(1, len(sorted_txns)):
                diff = (sorted_txns[i]['date'] - sorted_txns[i-1]['date']).days
                date_diffs.append(diff)
                
            if date_diffs:
                avg_days_between = sum(date_diffs) / len(date_diffs)
                std_dev = np.std(date_diffs) if len(date_diffs) > 1 else 0
                
                # Low standard deviation suggests consistent timing
                is_regular = std_dev < 10 or (std_dev / avg_days_between < 0.5 if avg_days_between > 0 else False)
                
                if is_regular:
                    # Determine frequency
                    if 25 <= avg_days_between <= 35:
                        frequency = "Monthly"
                    elif 13 <= avg_days_between <= 16:
                        frequency = "Bi-weekly"
                    elif 6 <= avg_days_between <= 8:
                        frequency = "Weekly"
                    elif 85 <= avg_days_between <= 95:
                        frequency = "Quarterly"
                    elif 350 <= avg_days_between <= 380:
                        frequency = "Yearly"
                    else:
                        frequency = f"Every {round(avg_days_between)} days"
                        
                    # Calculate next expected date
                    last_date = sorted_txns[-1]['date']
                    next_expected = last_date + timedelta(days=avg_days_between)
                    
                    recurring_groups.append({
                        'description': desc,
                        'amount': amount,
                        'frequency': frequency,
                        'avg_days_between': round(avg_days_between, 1),
                        'transaction_count': len(sorted_txns),
                        'first_date': sorted_txns[0]['date'].strftime('%Y-%m-%d'),
                        'last_date': last_date.strftime('%Y-%m-%d'),
                        'next_expected': next_expected.strftime('%Y-%m-%d'),
                        'regularity_score': round((1 - (std_dev / max(avg_days_between, 1))) * 100, 1) if avg_days_between > 0 else 0
                    })
    
    # Sort by regularity score (descending)
    recurring_groups.sort(key=lambda x: x['regularity_score'], reverse=True)
    
    return recurring_groups

def get_top_merchants(transactions: List[Dict[str, Any]],
                      description_column: str = 'description',
                      amount_column: str = 'amount',
                      top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Get top merchants by total spending.
    
    Args:
        transactions: List of transaction dictionaries
        description_column: Name of the description column
        amount_column: Name of the amount column
        top_n: Number of top merchants to return
        
    Returns:
        List of top merchant dictionaries
    """
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    if df.empty or description_column not in df.columns or amount_column not in df.columns:
        return []
    
    # Filter to only include expenses (negative amounts)
    expenses_df = df[df[amount_column] < 0].copy()
    expenses_df['amount_abs'] = expenses_df[amount_column].abs()
    
    if expenses_df.empty:
        return []
    
    # Group by merchant/description
    merchant_spending = expenses_df.groupby(description_column)['amount_abs'].agg(['sum', 'count']).reset_index()
    merchant_spending = merchant_spending.sort_values('sum', ascending=False)
    
    # Get top N merchants
    top_merchants = []
    for _, row in merchant_spending.head(top_n).iterrows():
        top_merchants.append({
            'merchant': row[description_column],
            'total_spent': float(row['sum']),
            'transaction_count': int(row['count']),
            'average_transaction': float(row['sum'] / row['count'])
        })
    
    return top_merchants

def analyze_income_sources(transactions: List[Dict[str, Any]],
                           date_column: str = 'date',
                           description_column: str = 'description',
                           amount_column: str = 'amount',
                           min_income_amount: float = 100) -> Dict[str, Any]:
    """
    Analyze income sources based on positive transaction amounts.
    
    Args:
        transactions: List of transaction dictionaries
        date_column: Name of the date column
        description_column: Name of the description column
        amount_column: Name of the amount column
        min_income_amount: Minimum amount to consider as income
        
    Returns:
        Dictionary with income source analysis
    """
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    if df.empty or date_column not in df.columns or description_column not in df.columns or amount_column not in df.columns:
        return {"error": "Missing required columns for income analysis"}
    
    # Ensure date column is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Filter to only include income (positive amounts above threshold)
    income_df = df[(df[amount_column] > min_income_amount)].copy()
    
    if income_df.empty:
        return {"warning": "No income transactions found above threshold"}
    
    # Group by description
    income_sources = income_df.groupby(description_column)[amount_column].agg(['sum', 'count']).reset_index()
    income_sources = income_sources.sort_values('sum', ascending=False)
    
    # Group by month
    income_df['month'] = income_df[date_column].dt.to_period('M')
    monthly_income = income_df.groupby('month')[amount_column].sum()
    
    # Calculate month-over-month change
    monthly_income_dict = monthly_income.to_dict()
    monthly_change = {}
    
    # Convert period keys to strings for JSON serialization
    monthly_income_str = {str(k): float(v) for k, v in monthly_income_dict.items()}
    
    # Calculate month-over-month percentage change
    previous_month = None
    for month in sorted(monthly_income.index):
        if previous_month:
            current = monthly_income[month]
            previous = monthly_income[previous_month]
            if previous > 0:
                change_pct = ((current - previous) / previous) * 100
                monthly_change[str(month)] = float(change_pct)
        previous_month = month
    
    # Format income sources
    sources = []
    for _, row in income_sources.iterrows():
        sources.append({
            'source': row[description_column],
            'total_amount': float(row['sum']),
            'transaction_count': int(row['count']),
            'average_amount': float(row['sum'] / row['count'])
        })
    
    return {
        "income_sources": sources,
        "monthly_income": monthly_income_str,
        "monthly_change_percentage": monthly_change,
        "total_income": float(income_df[amount_column].sum()),
        "average_monthly_income": float(monthly_income.mean()) if not monthly_income.empty else 0
    }

def generate_financial_summary(transactions: List[Dict[str, Any]],
                              category_column: str = 'category',
                              date_column: str = 'date',
                              amount_column: str = 'amount') -> Dict[str, Any]:
    """
    Generate an overall financial summary.
    
    Args:
        transactions: List of transaction dictionaries
        category_column: Name of the category column
        date_column: Name of the date column
        amount_column: Name of the amount column
        
    Returns:
        Dictionary with financial summary information
    """
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    if df.empty:
        return {"error": "No transactions provided for summary"}
    
    # Ensure essential columns exist
    required_columns = [amount_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return {"error": f"Missing required columns: {', '.join(missing_columns)}"}
    
    # Ensure date column is datetime if it exists
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Basic statistics
    total_transactions = len(df)
    total_inflow = float(df[df[amount_column] > 0][amount_column].sum())
    total_outflow = float(df[df[amount_column] < 0][amount_column].sum())
    net_flow = float(df[amount_column].sum())
    average_transaction = float(df[amount_column].mean())
    median_transaction = float(df[amount_column].median())
    
    # Time range
    if date_column in df.columns and not df[date_column].isna().all():
        start_date = df[date_column].min().strftime('%Y-%m-%d')
        end_date = df[date_column].max().strftime('%Y-%m-%d')
        date_range = (df[date_column].max() - df[date_column].min()).days
    else:
        start_date = None
        end_date = None
        date_range = None
    
    # Category breakdown if available
    category_breakdown = None
    if category_column in df.columns:
        category_counts = df[category_column].value_counts().to_dict()
        
        # Category spending (only outflow)
        cat_spending = df[df[amount_column] < 0].groupby(category_column)[amount_column].sum().abs().to_dict()
        
        # Category percentage of total spending
        total_spending = abs(total_outflow)
        cat_percentage = {cat: (amount / total_spending) * 100 for cat, amount in cat_spending.items()} if total_spending else {}
        
        category_breakdown = {
            "counts": category_counts,
            "spending": {k: float(v) for k, v in cat_spending.items()},
            "percentage": {k: float(v) for k, v in cat_percentage.items()}
        }
    
    # Calculate monthly stats if date column exists
    monthly_stats = None
    if date_column in df.columns and not df[date_column].isna().all():
        df['month'] = df[date_column].dt.to_period('M')
        
        monthly_net = df.groupby('month')[amount_column].sum().to_dict()
        monthly_inflow = df[df[amount_column] > 0].groupby('month')[amount_column].sum().to_dict()
        monthly_outflow = df[df[amount_column] < 0].groupby('month')[amount_column].sum().to_dict()
        
        # Convert period keys to strings for JSON serialization
        monthly_stats = {
            "net": {str(k): float(v) for k, v in monthly_net.items()},
            "inflow": {str(k): float(v) for k, v in monthly_inflow.items()},
            "outflow": {str(k): float(v) for k, v in monthly_outflow.items()}
        }
    
    # Assemble the summary
    summary = {
        "total_transactions": total_transactions,
        "total_inflow": total_inflow,
        "total_outflow": total_outflow,
        "net_flow": net_flow,
        "average_transaction": average_transaction,
        "median_transaction": median_transaction,
        "start_date": start_date,
        "end_date": end_date,
        "date_range_days": date_range,
        "category_breakdown": category_breakdown,
        "monthly_stats": monthly_stats
    }
    
    return summary