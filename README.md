# Statements Analyzer

An intelligent financial statement analyzer using advanced NLP techniques and an agentic approach to extract, categorize, and analyze transaction data from statement PDFs.

## Features

- **PDF Statement Processing**: Extract transaction data from PDF statements
- **Smart Transaction Categorization**: Categorize transactions using zero-shot classification
- **Financial Insights**: Generate spending trends, recurring payments, and financial summaries
- **Agent-Based Architecture**: Modular, extensible system using autonomous agents

## Architecture

The system uses a multi-agent architecture where specialized agents work together in a pipeline:

1. **PDF Processing Agent**: Extracts raw transaction data from PDFs
2. **Data Processing Agent**: Cleans and standardizes the extracted data
3. **Categorization Agent**: Assigns categories to transactions using zero-shot learning
4. **Analysis Agent**: Generates financial insights and spending analysis

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/statements-analyzer.git
cd statements-analyzer

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py path/to/your/statement.pdf
```

Optional arguments:
- `--bank`: Specify the statement format (e.g., 'chase', 'bofa')
- `--output`: Directory to store analysis results

## Example Output

The analyzer will extract transaction data, categorize it, and generate insights:

```
===== FINANCIAL INSIGHTS =====

1. Financial Health
   Your savings rate is 15.2%, which is good. You're saving more than the recommended 10% of your income.

2. Spending Trend
   Your monthly spending is relatively stable, changing by only 3.5% on average.

3. Top Expense Category
   Your highest spending category is Dining, accounting for $523.45 (28.7% of total expenses).

4. Recurring Expenses
   Found 5 potential recurring expenses totaling $125.90. Review these for subscription optimization opportunities.

5. Income Stability
   Your income is very stable, with only 2.1% average monthly variation.
```

Full analysis results are saved to a JSON file for further processing or visualization.

## License

MIT

## Credits

Developed using open-source technologies including:
- Pandas for data processing
- Hugging Face Transformers for zero-shot classification
- pdfplumber for PDF extraction