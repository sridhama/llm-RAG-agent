# RAG Agent for Financial Document Analysis with LangChain

This Python repository utilizes the LangChain library and the concept of Retrieval Augmented Generation (RAG) to perform various tasks related to financial document analysis. It demonstrates the use of LangChain agents coupled with language models, vector databases, document loading, summarization, and sentiment analysis on a financial annual report.

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- OpenAI API key

### Setup

1. Create and activate a Python virtual environment:

   ```bash
   python -m venv llm_rag_env
   source llm_rag_env/bin/activate
   ```

2. Clone the repository:

   ```bash
   git clone https://github.com/sridhama/financial-rag-agent-llm
   cd financial-rag-agent-llm
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Obtain an OpenAI API key from [OpenAI Platform](https://platform.openai.com/account/api-keys). Replace `'YOUR_API_KEY'` in  `/src/agent.py` with your actual API key. 

5. Store financial report or any other pertinent document for passage retrieval in the `/documents` directory.

### Usage

1. **Load and Store Documents in VectorDB**: `/src/agent.py` starts by loading and storing a PDF document into a Vector Database (VectorDB). You can specify the path to the PDF document and its metadata, such as name and description.

    ```python
    # specify document path and metadata
    doc_info = {
        'path': '../documents/financial_annual_report_2022.pdf',
        'name': 'financial_annual_report_2022',
        'description': "a finance company's annual report documentation for the year 2022"
    }
    ```

    The script loads and splits the PDF document into pages. You can customize the number of pages to use.

2. **Initialize Models**: The script initializes various language models for different tasks:

   - A base Language Model (LLM) from OpenAI.
   - A financial summarization model using Pegasus [https://huggingface.co/human-centered-summarization/financial-summarization-pegasus].
   - A financial sentiment analysis model using FinBERT [https://huggingface.co/yiyanghkust/finbert-tone].

3. **Create Custom Tools**: Custom LangChain tools are defined as Python functions decorated with `@tool`. These tools can be used to extract financial summaries and sentiments from text.

    ```python
    @tool
    def financial_summary(text: str) -> str:
        """Returns the financial summary of the input text."""
        return summary_hf.predict(text)

    @tool
    def financial_sentiment(text: str) -> str:
        """Returns the financial sentiment of the input text."""
        return finbert_pipe(text)
    ```

4. **Initialize Agents**: Different agents are created for retrieval, retrieval + summary, and retrieval + summary + sentiment analysis. These agents use the defined tools and models to respond to queries.

5. **Results**: The script provides examples of how to use the agents to retrieve information, summarize it, and analyze sentiment.