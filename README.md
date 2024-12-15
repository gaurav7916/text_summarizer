## Text Summarizer (Multilingual)

**Text Summarizer (Multilingual)📝** is a web application that summarizes text, PDFs, and images in multiple languages using a T5 transformer model. The application is built with Streamlit, EasyOCR, and Hugging Face Transformers.

## Features:

- Summarize text input directly from TXT PDF, and image files.
- Detect and multiple languages support.
- Translate summarized text to English.
- Chat-like prompt system for refining summaries

## Demo:

Live demo on [Streamlit](https://textsummarizer-based-on-llm.streamlit.app/).

## Screenshot:

![App Screenshot](https://github.com/gaurav7916/textsummarizer/screenshot_textsummarizer.png)

### Installation and Local Setup:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/gaurav7916/llm-based-text-summarizer.git
   cd LLM-Based-Text-Summarizer
   ```

2. **Create and activate a virtual environment:**

   ```sh
   python -m venv myenv
   source myenv/bin/activate   # On Windows use `myenv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**

   ```sh
   streamlit run app.py
   ```


## Acknowledgments

- [Streamlit](https://streamlit.io)
- [Hugging Face](https://huggingface.co)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
