# PerpleXive - AI Research Paper Assistant

A Gen AI-powered research paper assistant designed to streamline academic workflows, powered by Google Gemini 2.0 Flash. PerpleXive is a comprehensive Streamlit application that supports researchers with searching, analyzing, visualizing, and summarizing academic papers.

Features
🔎 Search Papers: Retrieve relevant papers from databases like Semantic Scholar, Arxiv, CORE, and OpenAlex using a smart search with fallback options. Papers are ranked by relevance with embeddings from Gemini, including detailed explanations for matches.

📄 Analyze Your Paper: Upload PDF papers for in-depth analysis, including text extraction, section segmentation (e.g., Abstract, Introduction), critiques, rubric-based scoring, rewriting suggestions, and a Q&A feature for specific queries.

📊 Visualize Paper: Generate graphical insights like word clouds, section length distributions, sentiment analysis, and readability scores using Matplotlib and WordCloud to interpret document structure and tone.

📝 Summarize Paper: Create professional summaries (250–350 words) of uploaded papers with key point explanations, downloadable as text files.

💬 Ask Questions: Engage with the app to ask targeted questions about uploaded papers for tailored feedback.

Built with robust tools like Google Gemini for content generation, FAISS for similarity search, and pdfplumber for PDF processing, PerpleXive offers error handling and a user-friendly interface with spinners and expanders. It’s an essential tool for researchers aiming to enhance productivity and research quality.

Screens of all features Perplexive can assist with:
![Screenshot 2025-04-24 at 2 05 53 PM](https://github.com/user-attachments/assets/87822a89-e353-46eb-a6bf-9ce3e289465c)
![Screenshot 2025-04-24 at 2 08 12 PM](https://github.com/user-attachments/assets/25479ae2-0d2f-4d7d-ae9b-5393061ec7e9)
![Screenshot 2025-04-24 at 2 07 18 PM](https://github.com/user-attachments/assets/3fb4c8e4-65cc-4e61-abbc-03337a10e1f9)
![Screenshot 2025-04-24 at 2 06 29 PM](https://github.com/user-attachments/assets/d9056b47-5b37-449f-9972-73f09ad04ead)


## Installation

1. Clone the repository
```bash
git clone https://github.com/aliasgar-saria/Perplexive---Gen-AI-Research-Paper-Analysis-AI-app.git
cd Perplexive---Gen-AI-Research-Paper-Analysis-AI-app
```

2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
cp .env
# Edit .env and add your Google Gemini API key
```

## Usage

Run the Streamlit app:
```bash
streamlit run Perplexive.py
```

## API Keys Required

- Google Gemini API key (Get it from [Google AI Studio](https://makersuite.google.com/app/apikey))


## Author

Aliasgar Saria
