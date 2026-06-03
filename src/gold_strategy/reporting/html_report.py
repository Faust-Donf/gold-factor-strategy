import pandas as pd
import os
from jinja2 import Template

def generate_html_report(metrics: dict, factor_selections: pd.DataFrame, plots: dict, out_path: str):
    """Generate HTML report."""
    
    template_str = """
    <html>
    <head><title>Gold Factor Strategy Report</title></head>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        table { border-collapse: collapse; width: 50%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        img { max-width: 800px; margin-bottom: 20px; }
    </style>
    <body>
        <h1>GLD Factor Strategy Report</h1>
        
        <h2>Backtest Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Strategy</th><th>Benchmark</th></tr>
            <tr><td>Ann. Return</td><td>{{ "%.2f"|format(metrics['Ann_Return'] * 100) }}%</td><td>{{ "%.2f"|format(metrics['BM_Return'] * 100) }}%</td></tr>
            <tr><td>Ann. Vol</td><td>{{ "%.2f"|format(metrics['Ann_Vol'] * 100) }}%</td><td>{{ "%.2f"|format(metrics['BM_Vol'] * 100) }}%</td></tr>
            <tr><td>Sharpe</td><td>{{ "%.2f"|format(metrics['Sharpe']) }}</td><td>{{ "%.2f"|format(metrics['BM_Sharpe']) }}</td></tr>
            <tr><td>Max DD</td><td>{{ "%.2f"|format(metrics['Max_DD'] * 100) }}%</td><td>{{ "%.2f"|format(metrics['BM_Max_DD'] * 100) }}%</td></tr>
            <tr><td>Win Rate</td><td>{{ "%.2f"|format(metrics['Win_Rate'] * 100) }}%</td><td>-</td></tr>
        </table>
        
        <h2>Selected Factors</h2>
        {{ factors_html }}
        
        <h2>Plots</h2>
        {% if 'equity' in plots %}
        <img src="{{ plots['equity'] }}" alt="Equity Curve">
        {% endif %}
        {% if 'drawdown' in plots %}
        <img src="{{ plots['drawdown'] }}" alt="Drawdown">
        {% endif %}
        
    </body>
    </html>
    """
    
    template = Template(template_str)
    
    factors_html = factor_selections.to_html(index=False) if not factor_selections.empty else "<p>No factor data</p>"
    
    html = template.render(
        metrics=metrics,
        factors_html=factors_html,
        plots=plots
    )
    
    with open(out_path, "w") as f:
        f.write(html)
