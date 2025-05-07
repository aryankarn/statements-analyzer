"""
Analysis Agent for Bank Statement Analyzer.
Responsible for generating financial insights from categorized transaction data.
"""

import logging
from typing import Dict, List, Any, AsyncGenerator

from agent_system import BaseAgent
from analysis.financial_analysis import (
    calculate_spending_trends,
    identify_recurring_transactions,
    get_top_merchants,
    analyze_income_sources,
    generate_financial_summary
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisAgent(BaseAgent):
    """
    Agent responsible for analyzing transaction data and generating financial insights.
    """
    
    def __init__(self, name: str = "Analyzer", description: str = None):
        """
        Initialize the Analysis Agent.
        
        Args:
            name: Name of the agent
            description: Description of the agent's purpose
        """
        super().__init__(
            name=name, 
            description=description or "Analyzes transaction data and generates financial insights"
        )
    
    async def _process(self, context: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Analyze transaction data.
        
        Args:
            context: Input context containing categorized transaction data
            
        Yields:
            Financial insights and analysis results
        """
        # Get categorized transactions from context
        transactions = context.get("categorized_transactions") or context.get("cleaned_transactions")
        if not transactions:
            yield {
                "status": "error",
                "message": "No transaction data found in context"
            }
            return
            
        # Analyze the transactions
        logger.info(f"Analyzing {len(transactions)} transactions")
        try:
            # Generate financial summary
            summary = generate_financial_summary(transactions)
            
            # Calculate spending trends
            trends = calculate_spending_trends(transactions)
            
            # Identify recurring transactions
            recurring = identify_recurring_transactions(transactions)
            
            # Get top merchants
            top_merchants = get_top_merchants(transactions)
            
            # Analyze income sources
            income_analysis = analyze_income_sources(transactions)
            
            # Compile all analysis results
            analysis_results = {
                "financial_summary": summary,
                "spending_trends": trends,
                "recurring_transactions": recurring,
                "top_merchants": top_merchants,
                "income_analysis": income_analysis
            }
            
            # Generate high-level insights based on the analysis
            insights = self._extract_insights(analysis_results)
            
            # Update context for next agents in the pipeline
            context_updates = {
                "analysis_results": analysis_results,
                "financial_insights": insights
            }
            
            # Yield the results
            yield {
                "status": "success",
                "message": "Successfully analyzed transaction data",
                "financial_summary": summary,
                "insights": insights,
                "state_updates": {
                    "analysis_results": analysis_results,
                    "financial_insights": insights
                },
                "context_updates": context_updates
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transactions: {str(e)}")
            yield {
                "status": "error",
                "message": f"Failed to analyze transactions: {str(e)}"
            }
    
    def _extract_insights(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract key financial insights from analysis results.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            
        Returns:
            List of insight dictionaries with title and description
        """
        insights = []
        
        # Get components from analysis results
        summary = analysis_results.get("financial_summary", {})
        trends = analysis_results.get("spending_trends", {})
        recurring = analysis_results.get("recurring_transactions", [])
        top_merchants = analysis_results.get("top_merchants", [])
        income = analysis_results.get("income_analysis", {})
        
        # Insight: Overall financial health
        if "net_flow" in summary and "total_inflow" in summary and summary["total_inflow"] > 0:
            net_flow = summary["net_flow"]
            savings_rate = (net_flow / summary["total_inflow"]) * 100 if summary["total_inflow"] > 0 else 0
            
            if savings_rate > 20:
                health_status = "Excellent"
                description = f"Your savings rate is {savings_rate:.1f}%, which is excellent! You're saving a significant portion of your income."
            elif savings_rate > 10:
                health_status = "Good"
                description = f"Your savings rate is {savings_rate:.1f}%, which is good. You're saving more than the recommended 10% of your income."
            elif savings_rate > 0:
                health_status = "Fair"
                description = f"Your savings rate is {savings_rate:.1f}%, which is positive but could be improved. Consider increasing savings to at least 10% of income."
            else:
                health_status = "Needs Attention"
                description = f"Your spending exceeds your income by ${abs(net_flow):.2f}, resulting in a negative savings rate. Consider reducing expenses."
                
            insights.append({
                "title": "Financial Health",
                "type": "health",
                "status": health_status,
                "description": description,
                "savings_rate": float(savings_rate)
            })
        
        # Insight: Spending trend
        if "monthly_change_percentage" in trends and trends["monthly_change_percentage"]:
            changes = list(trends["monthly_change_percentage"].values())
            avg_change = sum(changes) / len(changes) if changes else 0
            
            if avg_change > 10:
                trend_status = "Increasing"
                description = f"Your monthly spending is increasing by an average of {avg_change:.1f}% month-over-month. Consider reviewing your budget."
            elif avg_change < -10:
                trend_status = "Decreasing"
                description = f"Your monthly spending is decreasing by an average of {abs(avg_change):.1f}% month-over-month. Great job reducing expenses!"
            else:
                trend_status = "Stable"
                description = f"Your monthly spending is relatively stable, changing by only {abs(avg_change):.1f}% on average."
                
            insights.append({
                "title": "Spending Trend",
                "type": "trend",
                "status": trend_status,
                "description": description,
                "average_change_percent": float(avg_change)
            })
        
        # Insight: Top expense category
        if "category_breakdown" in summary and summary["category_breakdown"]:
            categories = summary["category_breakdown"]["spending"]
            if categories:
                top_category = max(categories.items(), key=lambda x: x[1])
                percentage = summary["category_breakdown"]["percentage"].get(top_category[0], 0)
                
                insights.append({
                    "title": "Top Expense Category",
                    "type": "category",
                    "category": top_category[0],
                    "amount": float(top_category[1]),
                    "percentage": float(percentage),
                    "description": f"Your highest spending category is {top_category[0]}, accounting for ${top_category[1]:.2f} ({percentage:.1f}% of total expenses)."
                })
        
        # Insight: Recurring transactions
        if recurring:
            total_recurring = sum(item["amount"] * item["transaction_count"] for item in recurring)
            
            insights.append({
                "title": "Recurring Expenses",
                "type": "recurring",
                "count": len(recurring),
                "total_amount": float(total_recurring),
                "description": f"Found {len(recurring)} potential recurring expenses totaling ${abs(total_recurring):.2f}. Review these for subscription optimization opportunities."
            })
        
        # Insight: Top merchant
        if top_merchants:
            top = top_merchants[0]
            
            insights.append({
                "title": "Top Merchant",
                "type": "merchant",
                "merchant": top["merchant"],
                "amount": float(top["total_spent"]),
                "transaction_count": top["transaction_count"],
                "description": f"Your highest spending was at {top['merchant']}, totaling ${top['total_spent']:.2f} across {top['transaction_count']} transactions."
            })
        
        # Insight: Income stability
        if "monthly_income" in income and income["monthly_income"]:
            monthly_values = list(income["monthly_income"].values())
            avg_income = sum(monthly_values) / len(monthly_values) if monthly_values else 0
            variations = [abs((v - avg_income) / avg_income) * 100 for v in monthly_values] if avg_income > 0 else []
            avg_variation = sum(variations) / len(variations) if variations else 0
            
            if avg_variation < 10:
                stability = "Stable"
                description = f"Your income is very stable, with only {avg_variation:.1f}% average monthly variation."
            elif avg_variation < 25:
                stability = "Moderately Variable"
                description = f"Your income has moderate variability, with {avg_variation:.1f}% average monthly variation."
            else:
                stability = "Highly Variable"
                description = f"Your income is highly variable, with {avg_variation:.1f}% average monthly variation. Consider building a larger emergency fund."
                
            insights.append({
                "title": "Income Stability",
                "type": "income",
                "status": stability,
                "variation_percent": float(avg_variation),
                "description": description
            })
        
        return insights