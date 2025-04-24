import streamlit as st
import numpy as np
import faiss
import requests
import time
import google.generativeai as genai
import pdfplumber
import re
import io
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from textstat import textstat
from PIL import Image

# Streamlit app config

st.set_page_config(page_title="PerpleXive", page_icon="💡", layout="wide")
st.title("PerpleXive - Research Paper Assistant")
st.markdown("Search for academic papers, analyze or visualize your own research paper, or generate a professional summary. Powerd by Google Gemini 2.0 Flash.")

# Initialize session state for paper analysis
if 'paper_data' not in st.session_state:
    st.session_state.paper_data = {
        'full_text': None,
        'chunks': None,
        'chunk_embeddings': None,
        'faiss_index': None,
        'sections': None
    }


# Setting up Google Gemini API

from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: Google API Key not found. Please set it in the .env file.")
    exit(1)
genai.configure(api_key=GOOGLE_API_KEY)


# Utility functions

def get_gemini_embedding(text):
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(response['embedding'])
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return clean_text(text)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def split_into_sections(text):
    sections = {}
    section_titles = [
        "Abstract", "Introduction", "Literature Review",
        "Methodology", "Results", "Discussion", "Conclusion"
    ]
    pattern = '|'.join([f'(?P<{t.replace(" ", "_")}>{t})' for t in section_titles])
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    
    for i, match in enumerate(matches):
        section_name = match.lastgroup.replace('_', ' ')
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections[section_name] = section_text
    return sections


# Search Databases

def search_semantic_scholar(query, limit=20):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={limit}&fields=title,authors,url,abstract,year"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        time.sleep(1)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()['data']
        else:
            st.warning(f"Semantic Scholar error {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Semantic Scholar exception: {e}")
        return []

def search_arxiv(query, limit=20):
    base_url = 'http://export.arxiv.org/api/query?'
    params = f'search_query=all:{query}&start=0&max_results={limit}'
    try:
        response = requests.get(base_url + params)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            entries = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                url = entry.find('{http://www.w3.org/2005/Atom}id').text.strip()
                entries.append({
                    'title': title,
                    'abstract': summary,
                    'url': url,
                    'authors': [],
                    'year': ''
                })
            return entries
        else:
            st.warning(f"Arxiv error {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"Arxiv exception: {e}")
        return []

def search_core(query, limit=20):
    try:
        url = f"https://core.ac.uk:443/api-v2/search/{query}?page=1&pageSize={limit}&metadata=true&fulltext=false&citations=false&similar=false"
        core_api_key = "your_core_api_key_here"  # Replace with valid key
        headers = {
            "Authorization": f"Bearer {core_api_key}",
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            results = []
            for paper in data['data']:
                results.append({
                    'title': paper['title'],
                    'abstract': paper.get('description', ''),
                    'url': paper.get('downloadUrl', ''),
                    'authors': [],
                    'year': paper.get('year', '')
                })
            return results
        else:
            st.warning(f"CORE error {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"CORE exception: {e}")
        return []

def search_openalex(query, limit=20):
    try:
        url = f"https://api.openalex.org/works?filter=title.search:{query}&per-page={limit}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            results = []
            for work in data['results']:
                results.append({
                    'title': work['display_name'],
                    'abstract': work.get('abstract', ''),
                    'url': work['id'],
                    'authors': [],
                    'year': work.get('publication_year', '')
                })
            return results
        else:
            st.warning(f"OpenAlex error {response.status_code}")
            return []
    except Exception as e:
        st.warning(f"OpenAlex exception: {e}")
        return []

def smart_paper_search(query, limit=20):
    papers = search_semantic_scholar(query, limit)
    if papers:
        return papers, "Semantic Scholar"
    
    papers = search_arxiv(query, limit)
    if papers:
        return papers, "Arxiv"

    papers = search_core(query, limit)
    if papers:
        return papers, "CORE.ac.uk"
    
    papers = search_openalex(query, limit)
    if papers:
        return papers, "OpenAlex"

    return [], None

# Paper Search and Ranking

def explain_relevance(paper_title, paper_abstract, user_query):
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
You are an expert AI Research Assistant.

You have to:
1. Summarize the paper very clearly and simply in 6–8 lines (easy for beginners).
2. Highlight the key points and important findings of the paper.
3. Explain why this paper is relevant to the user's topic: "{user_query}".
4. Advise how the user can use this paper for their project or research (give actionable advice).
5. Suggest the best AI/ML models or methods they can explore based on this paper.
6. Discuss current trends in this research area and how this paper connects to them.

Here are the paper details:
Title: {paper_title}
Abstract: {paper_abstract}

Respond in clear Markdown format.
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating explanation: {e}"

def get_ranked_papers(user_query, top_k=5):
    papers, source = smart_paper_search(user_query)
    
    if not papers:
        st.warning(f"No papers found in {source or 'any database'}. Trying fallbacks...")
        papers = search_semantic_scholar(user_query)  # Get an API key to remove the Semantic rate limit error 429
        source = "Semantic Scholar"
        if not papers:
            papers = search_arxiv(user_query)
            source = "Arxiv"
        if not papers:
            papers = search_core(user_query)
            source = "CORE.ac.uk"
        if not papers:
            papers = search_openalex(user_query)
            source = "OpenAlex"
    
    if not papers:
        return []

    paper_texts = []
    for paper in papers:
        title = paper.get('title', 'No Title')
        abstract = paper.get('abstract', '')
        paper_texts.append(f"{title}. {abstract}")

    paper_embeddings = []
    for text in paper_texts:
        embedding = get_gemini_embedding(text)
        if embedding is not None:
            paper_embeddings.append(embedding)
    
    if not paper_embeddings:
        st.error("Failed to generate embeddings for papers.")
        return []

    paper_embeddings = np.array(paper_embeddings)

    query_embedding = get_gemini_embedding(user_query)
    if query_embedding is None:
        st.error("Failed to generate embedding for query.")
        return []
    query_embedding = query_embedding.reshape(1, -1)

    index = build_faiss_index(paper_embeddings)
    distances, indices = index.search(query_embedding, min(top_k, len(papers)))

    ranked_papers = []
    for i, idx in enumerate(indices[0]):
        if idx < len(papers):
            paper = papers[idx]
            similarity = 100 - distances[0][i]
            ranked_papers.append((paper, similarity, source))
    return ranked_papers


# Paper Analysis Functions (Needs improvements for later when API updates)

def build_analysis_prompt(pdf_text):
    few_shot_examples = """
Example 1:
---
Paper Summary:  
This paper proposes a novel convolutional neural network (CNN) architecture for early cancer detection from CT scans. The model outperforms traditional machine learning baselines by 12% on sensitivity and 10% on accuracy.

Key Points:  
- Custom lightweight CNN designed for low-resource environments.  
- Dataset: 30,000 labeled CT scan images.  
- Achieved 92% sensitivity and 89% accuracy.  

Strengths:  
- Innovative architecture optimized for edge devices.  
- Comprehensive dataset and evaluation.  

Weaknesses:  
- Lack of external validation on different datasets.  
- Limited explainability of model predictions.  

Advice for Improvement:  
- Include results on multiple datasets to improve generalization.  
- Add explainability techniques (e.g., Grad-CAM) to make results interpretable.

---

Example 2:
---
Paper Summary:  
The study explores reinforcement learning (RL) techniques for optimizing drone flight paths. It compares reward shaping methods across three environments: simulation, semi-realistic, and real-world.

Key Points:  
- Proposed a hybrid reward shaping mechanism.  
- Focused on robustness against real-world noise.  
- Achieved 25% faster convergence than standard DQN.  

Strengths:  
- Practical experiments on real drones.  
- Strong experimental setup and ablation studies.  

Weaknesses:  
- Only evaluated with quadcopters, not other drone types.  

Advice for Improvement:  
- Test across different drone types for better generalization.
- Explore additional RL algorithms beyond DQN.
"""
    return f"""
You are an AI Research Assistant.

Analyze the following research paper thoroughly:
{few_shot_examples}

Now analyze this paper:
{pdf_text}

Provide:
- Paper Summary
- Key Points
- Strengths
- Weaknesses
- Advice for Improvement
- Overall Quality Score
"""

def analyze_uploaded_paper(pdf_text):
    prompt = build_analysis_prompt(pdf_text)
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating analysis: {e}"

def score_paper_rubric(pdf_text):
    rubric_prompt = f"""
You are an academic reviewer.

Score this research paper based on the following rubric (each out of 10):
- Clarity of Writing
- Originality of Research
- Technical Rigor and Methodology
- Grammar and Language Style
- Structure and Organization

Be critical but fair. Here is the paper:
\"\"\"
{pdf_text}
\"\"\"
Please format your response clearly under headings.
"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(rubric_prompt)
        return response.text
    except Exception as e:
        return f"Error generating rubric score: {e}"

def analyze_section(name, content):
    prompt = f"""
You are an expert research paper editor.
Analyze the following '{name}' section for:
- Strengths
- Weaknesses
- Suggestions for rewriting if needed
- Grammar Score (out of 10)
- Technical Clarity Score (out of 10)

Content:
{content}

Give detailed reasoning.
"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing section: {e}"

def rewrite_section(name, content):
    prompt = f"""
You are an expert academic writing assistant.
Rewrite the '{name}' section below to improve:
- Grammar
- Academic Style
- Technical Depth
- Logical flow
- Add sample citations if needed (fake but plausible)

Rewrite professionally for a research paper.

Content:
{content}
"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error rewriting section: {e}"

def retrieve_relevant_chunks(query, chunk_texts, chunk_embeddings, index, top_k=3):
    query_embedding = get_gemini_embedding(query)
    if query_embedding is None:
        return ""
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunk_texts[i] for i in indices[0] if i < len(chunk_texts)]
    return "\n\n".join(retrieved_chunks)

def chat_about_paper(user_question, chunk_texts, chunk_embeddings, index):
    context = retrieve_relevant_chunks(user_question, chunk_texts, chunk_embeddings, index)
    if not context:
        return "No relevant content found or error in embedding generation."
    prompt = f"""
You are helping a researcher with their uploaded paper.

Context from their paper:
{context}

User's Question:
{user_question}

Answer very clearly, helpfully, and politely.
"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"


# Graphical analysis gunctions

def plot_word_cloud(text):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Frequency Distribution")
        return fig
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
        return None

def plot_section_lengths(sections):
    section_names = list(sections.keys())
    section_word_counts = [len(content.split()) for content in sections.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(section_names, section_word_counts, color='skyblue')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Sections')
    ax.set_title('Word Count Distribution Across Sections')
    return fig

def plot_sentiment_analysis(sections):
    section_names = list(sections.keys())
    sentiments = [TextBlob(content).sentiment.polarity for content in sections.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(section_names, sentiments, color='lightgreen')
    ax.set_xlabel('Sentiment Score (Range: -1 to 1)')
    ax.set_ylabel('Sections')
    ax.set_title('Sentiment Analysis of Sections')
    return fig

def plot_readability_scores(sections):
    section_names = list(sections.keys())
    readability_scores = [textstat.flesch_reading_ease(content) for content in sections.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(section_names, readability_scores, color='salmon')
    ax.set_xlabel('Flesch Reading Ease Score')
    ax.set_ylabel('Sections')
    ax.set_title('Readability Scores for Sections')
    return fig

def perform_graphical_analysis(sections, full_text):
    st.markdown("### Graphical Analysis Results")
    
    # Wordccloud
    with st.spinner("Generating word cloud..."):
        fig = plot_word_cloud(full_text)
        if fig:
            st.pyplot(fig)
            plt.close(fig)
    

    with st.spinner("Generating section lengths plot..."):
        fig = plot_section_lengths(sections)
        st.pyplot(fig)
        plt.close(fig)
    
    with st.spinner("Generating sentiment analysis plot..."):
        fig = plot_sentiment_analysis(sections)
        st.pyplot(fig)
        plt.close(fig)
    

    with st.spinner("Generating readability scores plot..."):
        fig = plot_readability_scores(sections)
        st.pyplot(fig)
        plt.close(fig)


# Summary generation functions

def generate_summary(full_text):
    prompt = f"""
You are a professional academic editor and scientific writer.

Your task is to generate a high-quality structured summary of the following research paper. 
The summary should be:
- Precise, clear, and highly professional
- Formal academic tone (no casual language)
- Between 250 and 350 words
- Well-organized into logical flow
- Emphasizing key points without unnecessary detail

Focus on summarizing:
1. The research objective and background
2. Methodologies and techniques used
3. Major findings and results
4. Important discussions and implications
5. Final conclusions and potential future work

Strictly avoid personal opinions or assumptions not grounded in the text.

Here is the paper content:

{full_text}
"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"

def explain_summary(summary_text):
    prompt = f"""
You are an expert academic assistant.

Provide a concise explanation of the following research paper summary, highlighting its key points in a clear and professional manner. The explanation should:
- Be 100–150 words
- Use a formal academic tone
- Identify 3–5 key points from the summary
- Explain the significance of these points for researchers or the field

Here is the summary:

{summary_text}

Respond in clear Markdown format.
"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating explanation: {e}"

# Streamlit GUI

def main():
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Slide to choose number of search results", min_value=1, max_value=10, value=5)
        st.markdown("### About")
        st.markdown("This app searches academic papers, analyzes, visualizes, and summarizes uploaded research papers. Powered by Google Gemini Flash 2.0.")
        st.markdown("### Made By")
        st.markdown("Aliasgar Saria")

    search_tab, analyze_tab, visualize_tab, summarize_tab = st.tabs(["🔎 Search Papers", "📄 Analyze Your Paper", "📊 Visualize Paper", "📝 Summarize Paper"])

    with search_tab:
        with st.form(key="query_form"):
            user_query = st.text_input("💬 Enter your research topic or question:", placeholder="e.g., Machine learning in healthcare")
            submit_button = st.form_submit_button("🔎 Search Papers")

        if submit_button and user_query:
            with st.spinner("Searching for papers..."):
                ranked_papers = get_ranked_papers(user_query, top_k=top_k)

            if not ranked_papers:
                st.error("No papers found. Try a different topic!")
            else:
                st.success(f"Found {len(ranked_papers)} papers!")
                for idx, (paper, similarity, source) in enumerate(ranked_papers, 1):
                    title = paper.get('title', 'No Title')
                    authors = ', '.join([author.get('name', '') for author in paper.get('authors', [])]) or "Unknown Authors"
                    year = paper.get('year', 'Unknown Year')
                    url = paper.get('url', '#')

                    with st.expander(f"📄 Paper {idx}: {title} ({similarity:.2f}% Match)"):
                        st.markdown(f"""
                        **Title:** {title}  
                        **Authors:** {authors}  
                        **Year:** {year}  
                        **Source:** {source}  
                        **Link:** [{url}]({url})  
                        **Match Score:** {similarity:.2f}%  
                        """)
                        
                        abstract = paper.get('abstract', '')
                        if abstract:
                            with st.spinner(f"Generating explanation for Paper {idx}..."):
                                explanation = explain_relevance(title, abstract, user_query)
                                st.markdown("### Why is this paper relevant?")
                                st.markdown(explanation, unsafe_allow_html=True)
                        else:
                            st.warning("No abstract available for this paper.")

    with analyze_tab:
        st.markdown("### Upload Your Research Paper")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file:
            with st.spinner("Processing your paper..."):
                # Extract and process text
                full_text = extract_text_from_pdf(uploaded_file)
                if not full_text:
                    st.error("Failed to extract text from the PDF.")
                else:
                    chunks = chunk_text(full_text)
                    chunk_embeddings = [get_gemini_embedding(chunk) for chunk in chunks]
                    chunk_embeddings = [emb for emb in chunk_embeddings if emb is not None]
                    
                    if not chunk_embeddings:
                        st.error("Failed to generate embeddings for paper content.")
                    else:
                        chunk_embeddings = np.array(chunk_embeddings)
                        faiss_index = build_faiss_index(chunk_embeddings)
                        sections = split_into_sections(full_text)
                        
                        # Store in session state
                        st.session_state.paper_data = {
                            'full_text': full_text,
                            'chunks': chunks,
                            'chunk_embeddings': chunk_embeddings,
                            'faiss_index': faiss_index,
                            'sections': sections
                        }

                        # Full paper analysis
                        st.markdown("### Full Paper Analysis")
                        with st.spinner("Analyzing your paper..."):
                            analysis = analyze_uploaded_paper(full_text)
                            st.markdown(analysis, unsafe_allow_html=True)

                        # Rubric scoring
                        st.markdown("### Paper Quality Score")
                        with st.spinner("Scoring your paper..."):
                            rubric = score_paper_rubric(full_text)
                            st.markdown(rubric, unsafe_allow_html=True)

                        # Display detected sections
                        if sections:
                            st.markdown("**Detected Sections:**")
                            for idx, sec in enumerate(sections.keys(), 1):
                                st.write(f"{idx}. {sec}")
                        else:
                            st.warning("No major sections detected in the paper.")

        if st.session_state.paper_data['full_text']:
            st.markdown("### Analyze and Rewrite Sections")
            if st.button("Analyze and Rewrite Sections"):
                sections = st.session_state.paper_data['sections']
                if not sections:
                    st.warning("No sections available for analysis.")
                else:
                    for name, content in sections.items():
                        if content.strip():
                            with st.expander(f"Analysis for {name}"):
                                with st.spinner(f"Analyzing {name}..."):
                                    section_analysis = analyze_section(name, content)
                                    st.markdown(section_analysis, unsafe_allow_html=True)
                                
                                if st.button(f"Rewrite {name}", key=f"rewrite_{name}"):
                                    with st.spinner(f"Rewriting {name}..."):
                                        rewritten = rewrite_section(name, content)
                                        st.markdown("### Rewritten Section")
                                        st.markdown(rewritten, unsafe_allow_html=True)
                        else:
                            st.warning(f"{name} section is empty.")

        if st.session_state.paper_data['full_text']:
            st.markdown("### Ask Questions About Your Paper")
            example_questions = [
                "Critique the clarity and focus of my research problem statement.",
                "Identify gaps or weaknesses in my literature review section.",
                "Analyze the strength of my methodology — any flaws or improvements?",
                "Evaluate if my results and analysis are convincing and well-supported.",
                "Suggest improvements for the conclusion to make it more powerful.",
                "Recommend recent papers or citations I should add.",
                "Check if my paper aligns with current trends in the field.",
                "Advise on making my abstract more concise and engaging.",
                "Point out any redundancy, irrelevant sections, or off-topic parts.",
                "Detect possible ethical concerns or biases in my study.",
                "Suggest formatting or structure improvements for better flow.",
                "Help me prepare potential reviewer questions for peer-review.",
                "Check if my research contributions are stated clearly enough.",
                "Critique the originality and innovation level of my work."
            ]
            st.markdown("**Example Questions:**")
            for q in example_questions:
                st.write(f"- {q}")

            with st.form(key="qa_form"):
                user_question = st.text_input("💬 Ask a question about your paper:", placeholder="e.g., What are the weaknesses in my methodology?")
                qa_submit = st.form_submit_button("Submit Question")

            if qa_submit and user_question:
                if st.session_state.paper_data['chunks'] is None:
                    st.error("No paper data available for Q&A.")
                else:
                    with st.spinner("Generating answer..."):
                        answer = chat_about_paper(
                            user_question,
                            st.session_state.paper_data['chunks'],
                            st.session_state.paper_data['chunk_embeddings'],
                            st.session_state.paper_data['faiss_index']
                        )
                        st.markdown("### Answer")
                        st.markdown(answer, unsafe_allow_html=True)

    with visualize_tab:
        st.markdown("### Visualize Your Research Paper")
        if st.session_state.paper_data['full_text']:
            st.markdown("Visualize insights from your uploaded paper, including word frequency, section lengths, sentiment, and readability.")
            if st.button("Generate Visualizations"):
                with st.spinner("Generating visualizations..."):
                    sections = st.session_state.paper_data['sections']
                    full_text = st.session_state.paper_data['full_text']
                    if not sections:
                        st.warning("No sections detected in the paper for visualization.")
                    else:
                        perform_graphical_analysis(sections, full_text)
        else:
            st.info("Please upload a paper in the 'Analyze Your Paper' tab to generate visualizations.")

    with summarize_tab:
        st.markdown("### Summarize Your Research Paper")
        if st.session_state.paper_data['full_text']:
            st.markdown("Generate a professional summary (250–350 words) of your uploaded paper, with an explanation and key points.")
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    full_text = st.session_state.paper_data['full_text']
                    summary = generate_summary(full_text)
                    st.markdown("### Paper Summary")
                    st.markdown(summary, unsafe_allow_html=True)


                    with st.spinner("Generating explanation..."):
                        explanation = explain_summary(summary)
                        st.markdown("### Explanation and Key Points")
                        st.markdown(explanation, unsafe_allow_html=True)

                    st.markdown("### Download Summary")
                    st.download_button(
                        label="Download Summary as Text File",
                        data=summary,
                        file_name="paper_summary.txt",
                        mime="text/plain"
                    )
        else:
            st.info("Please upload a paper in the 'Analyze Your Paper' tab to generate a summary.")


# App Running

if __name__ == "__main__":
    main()