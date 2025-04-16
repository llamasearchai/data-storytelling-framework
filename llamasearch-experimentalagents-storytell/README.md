# Data Storytelling Framework

A modern Python framework for **automated data storytelling**: generate executive-ready narratives, actionable insights, and beautiful visualizations from your data using LLMs and best-in-class analytics. Perfect for business reporting, e-commerce analytics, marketing dashboards, and any organization seeking to turn data into decisions—**automatically**.

---

## 🚀 Features

- **Automated Narrative Generation**: Turn raw data into clear, compelling stories using OpenAI or your preferred LLM.
- **Insight Extraction**: Identify key drivers, trends, and actionable recommendations.
- **Beautiful Visualizations**: Generate interactive dashboards and charts (Plotly, Matplotlib, Seaborn) with theming support.
- **Executive Dashboards**: Combine narrative, metrics, and visuals in a single HTML dashboard for decision-makers.
- **Flexible Data Input**: Accepts CSV, Excel, JSON, Pandas DataFrames, or Python dicts.
- **Extensible Architecture**: Add new chart types, LLMs, or custom business logic easily.
- **CLI & Python API**: Use from the command line or integrate into your own Python workflows.
- **Robust Configuration**: .env, environment variable, and JSON config support.
- **Tested & Production-Ready**: Includes unit tests and real-world examples.

---

## 🌍 Real-World Use Cases (2024)

- **AI-Driven Business Reporting**: Automate monthly/quarterly business reviews with LLM-generated narratives and dashboards.
- **E-commerce Analytics**: Instantly surface what's driving sales, churn, or conversion—no analyst required.
- **Marketing Campaign Analysis**: Visualize attribution, experiment results, and ROI for every channel.
- **Executive Dashboards**: Deliver C-suite-ready insights, not just charts, with actionable recommendations.
- **GenAI for Data Teams**: Empower analysts and non-analysts alike to get stories, not just stats, from their data.
- **Trending**: As seen in 2024, companies are using AI storytelling to replace static BI dashboards and accelerate decision cycles ([see Gartner, Forrester, and McKinsey reports](https://www.gartner.com/en/newsroom/press-releases/2024-03-15-gartner-says-generative-ai-will-transform-business-intelligence)).

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/yourorg/data-storytelling-framework.git
cd data-storytelling-framework

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Quick Start (Python API)

```python
import pandas as pd
from llamasearch_experimentalagents_storytell.core.engine import StorytellingEngine

# Prepare your data (as DataFrame, dict, or file path)
data = pd.read_csv("sales_data.csv")

# Initialize the engine (set your OpenAI API key via env or argument)
engine = StorytellingEngine(openai_api_key="sk-...your-key...")

# Run the full pipeline
result = engine.run_pipeline(
    data=data,
    output_dir="./output",
    output_format="html",
    context="Monthly sales and marketing performance for Q2 2024. Highlight key drivers and risks.",
    title="Q2 2024 Sales & Marketing Executive Summary"
)

print(f"Narrative: {result['narrative'].summary}")
print(f"Dashboard: {result['dashboard_path']}")
```

---

## 🖥️ Command-Line Usage

```bash
python -m llamasearch_experimentalagents_storytell.cli \
  --data ./data/sales_data.csv \
  --output-dir ./output \
  --api-key sk-...your-key... \
  --title "Q2 2024 Sales & Marketing Executive Summary" \
  --context "Monthly sales and marketing performance for Q2 2024. Highlight key drivers and risks."
```

- Outputs: `narrative.json`, `dashboard.html`, and all visualizations in `./output`.
- Use `--narrative-only` to skip visualizations and dashboard.

---

## 📊 What You Get

- **NarrativeSummary**: Executive summary, key insights, recommendations, and full narrative text (JSON/Markdown).
- **Visualizations**: Interactive charts (attribution, experiment results, performance metrics, and more).
- **Dashboard**: A single HTML file combining narrative and visuals, ready for executives or clients.

---

## 🧩 Extending the Framework

- **Add new chart types**: Subclass or extend `VisualizationEngine`.
- **Plug in your own LLM**: Swap out the agent in `StorytellingEngine`.
- **Custom business logic**: Add new methods or hooks for your domain.
- **Configuration**: Use `.env`, environment variables, or JSON config files for all settings.

---

## 📝 Example: E-commerce Executive Dashboard

```python
from llamasearch_experimentalagents_storytell.core.engine import StorytellingEngine
import pandas as pd

data = pd.read_csv("ecommerce_q2_2024.csv")
engine = StorytellingEngine(openai_api_key="sk-...your-key...")
result = engine.run_pipeline(
    data=data,
    output_dir="./output",
    title="Q2 2024 E-commerce Executive Dashboard",
    context="Summarize key drivers of revenue, conversion, and customer growth."
)
print(result['dashboard_path'])
```

---

## 🔒 Security & Privacy
- Your data is processed locally; only the narrative generation step (if using OpenAI) sends data to the LLM API.
- You control what data is sent and can use mock agents for sensitive data.

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 🤝 Contributing

Pull requests, issues, and feature suggestions are welcome! Please see `CONTRIBUTING.md`.

---

## 📚 License

MIT License. See `LICENSE` for details.

---

## 💡 Why Data Storytelling?

In 2024, organizations are moving beyond static dashboards. Executives want **stories, not just stats**. This framework lets you:
- Automate business reviews and board reports
- Empower non-analysts to get actionable insights
- Accelerate decision cycles with AI-driven narratives
- Stand out with client-ready, C-suite-ready deliverables

**Turn your data into decisions—automatically.** 