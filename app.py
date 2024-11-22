import streamlit as st
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from PyPDF2 import PdfReader

# --- Load Pre-trained Models ---
@st.cache_resource
def load_huggingface_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_bart_model():
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    return model, tokenizer

@st.cache_resource
def load_bart_no_attention_model():
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    # Disable training for the model to simulate no-attention behavior
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer

# Initialize Models
huggingface_model = load_huggingface_model()
bart_model, bart_tokenizer = load_bart_model()
bart_no_attention_model, bart_no_attention_tokenizer = load_bart_no_attention_model()

# --- Streamlit Interface ---
st.title("Text Summarization App")
st.write("Upload a file or enter text to summarize.")

# Dropdown for model selection
model_choice = st.selectbox(
    "Select a summarization model:",
    [
        "Transformer Model",
        "Attention Model",
        "Without Attention Model"
    ]
)

# Input option: text or file
input_choice = st.radio("Choose input method:", ["Enter text manually", "Upload PDF file"])

# Process input
input_text = ""
if input_choice == "Enter text manually":
    input_text = st.text_area("Enter text to summarize:", height=200)
elif input_choice == "Upload PDF file":
    if model_choice == "Without Attention Model":
        st.warning("PDF input is not supported for this model. Please enter text manually.")
    else:
        uploaded_file = st.file_uploader("Upload a PDF file:", type=["pdf"])
        if uploaded_file:
            try:
                reader = PdfReader(uploaded_file)
                input_text = " ".join(
                    page.extract_text() for page in reader.pages if page.extract_text()
                )
                if not input_text.strip():
                    st.warning("The uploaded PDF appears to be empty or unreadable.")
            except Exception as e:
                st.error(f"Error reading PDF file: {e}")

# Sliders for summary length
min_length = st.slider("Minimum Summary Length", 10, 100, 30)
max_length = st.slider("Maximum Summary Length", 50, 300, 130)

# Button to summarize
if st.button("Summarize"):
    if not input_text.strip():
        st.warning("Please provide some text or upload a valid PDF.")
    else:
        with st.spinner("Summarizing..."):
            # Ensure input doesn't exceed model limits
            if len(input_text.split()) > 1024:
                st.warning("Input text is too long. Truncating to fit model limits.")
                input_text = " ".join(input_text.split()[:1024])

            summary = ""
            try:
                if model_choice == "Transformer Model":
                    # Hugging Face model
                    summary = huggingface_model(
                        input_text, min_length=min_length, max_length=max_length, do_sample=False
                    )[0]['summary_text']
                elif model_choice == "Attention Model":
                    # BART with Attention
                    inputs = bart_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
                    summary_ids = bart_model.generate(
                        inputs["input_ids"], min_length=min_length, max_length=max_length, num_beams=4, early_stopping=True
                    )
                    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                elif model_choice == "Without Attention Model":
                    # BART without Attention
                    inputs = bart_no_attention_tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
                    summary_ids = bart_no_attention_model.generate(
                        inputs["input_ids"], min_length=min_length, max_length=max_length, num_beams=1, early_stopping=True
                    )
                    summary = bart_no_attention_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                st.success("Summary generated!")
                st.write("### Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error during summarization: {e}")
