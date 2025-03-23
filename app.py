import streamlit as st
import pandas as pd
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import time

# Page configuration
st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="wide"
)
import streamlit as st

# Remove sidebar top padding
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            padding-top: 0px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Short Welcome Message
st.sidebar.markdown("## 🌟 Welcome!")
# # Add an image at the top
st.sidebar.image("./robot.jpg", use_container_width=True)

# Project Short Description
st.sidebar.markdown("""
### 📌 About This Project  
🔍 **AI Research Paper Summarizer** – Generate clear, concise summaries of scientific papers instantly. Save time, grasp key insights, and stay ahead in research!
""")

st.sidebar.markdown("## 👥 Meet Our Team")

st.sidebar.markdown("""
👨‍💻 [Anurag Ghosh](https://github.com/Anurag-ghosh-12) &nbsp;&nbsp;  
👨‍💻 [Siddharth Sen](https://github.com/Sidhupaji-2004) &nbsp;&nbsp;  
👩‍💻 [Suchana Hazra](https://github.com/Suchana4Hazra) &nbsp;&nbsp;  
👨‍💻 [Uttam Mahata](https://github.com/Uttam-Mahata)  
""", unsafe_allow_html=True)

# # Add a demo GIF (optional)
# st.sidebar.markdown("## 🎥 Project Demo")
# st.sidebar.image("./demo.gif", use_container_width=True)

# Explore Project Button
st.sidebar.markdown("## 🔗 Explore More")

st.sidebar.markdown(
    '<a href="https://github.com/Suchana4Hazra/Research_Paper_Summarizer" target="_blank">'
    '<button style="background-color:#4CAF50; color:white; padding:10px 15px; border:none; border-radius:5px; cursor:pointer; font-size:16px;">'
    '🚀 Explore Project on GitHub</button></a>',
    unsafe_allow_html=True
)

# Use markdown with CSS to adjust the height

# Load and resize image
img = Image.open("robot2.webp")  # Ensure correct path
img = img.resize((img.width, int(img.height * 0.8)))  # Reduce height by 20%

# Display resized image
st.image(img, use_container_width=True)



# Optional: Add a title or welcome message below the image
st.title("🧠 AI Research Paper Summarizer")
st.markdown("Enter paper details to generate a summary using our optimized model.")

# Load model and tokenizer directly from Hugging Face with optimizations
@st.cache_resource
def load_model_and_tokenizer():
    try:
        model_name = "sshleifer/distilbart-cnn-12-6"  # Smaller, faster model

        # Show a spinner without exposing function internals
        with st.spinner("Loading summarization model (first load may take a minute)..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Convert to half precision for faster inference
            if torch.cuda.is_available():
                model = model.half().cuda()
            else:
                model = model

            # Create optimized pipeline
            summarizer = pipeline(
                "summarization", 
                model=model, 
                tokenizer=tokenizer, 
                device=0 if torch.cuda.is_available() else -1
            )

        return summarizer  # Spinner disappears once this completes

    except Exception as e:
        st.error("❌ Error loading model. Please try again.")
        return None



# Generate summary function with optimized parameters
def generate_summary(text, summarizer, max_length=96, min_length=20):
    try:
        # Start timing
        start_time = time.time()
        
        # Use the summarizer pipeline with optimized parameters
        summary = summarizer(
            text, 
            max_length=max_length, 
            min_length=min_length,
            do_sample=False,
            num_beams=2,  # Reduce beam search for speed
            early_stopping=True
        )
        
        # End timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        return summary[0]["summary_text"], processing_time
    
    except Exception as e:
        st.error(f"Error in summary generation: {e}")
        return "Failed to generate summary due to an error.", 0

# Load model at startup
try:
    # Check if model is already in session state
    if 'summarizer' not in st.session_state:
        summarizer = load_model_and_tokenizer()
        if summarizer is not None:
            st.success("Summarization model loaded successfully!")
            st.session_state['summarizer'] = summarizer
        else:
            st.error("Failed to load the summarization model.")
    else:
        st.success("Summarization model already loaded!")
except Exception as e:
    st.error(f"Error initializing model: {e}")

# Form for manual input of paper details
st.subheader("Enter Paper Details")

with st.form("paper_details_form"):
    # Input fields for all ArXiv paper attributes
    paper_id = st.text_input("ArXiv ID", placeholder="e.g., 2104.08663")
    submitter = st.text_input("Submitter", placeholder="e.g., John Doe")
    authors = st.text_input("Authors", placeholder="e.g., Author 1, Author 2, Author 3")
    title = st.text_input("Title", placeholder="Paper title")
    comments = st.text_input("Comments", placeholder="e.g., 12 pages, 5 figures")
    journal_ref = st.text_input("Journal Reference", placeholder="e.g., Nature Physics vol. 1, p. 23 (2022)")
    doi = st.text_input("DOI", placeholder="e.g., 10.1234/example.doi")
    report_no = st.text_input("Report Number", placeholder="e.g., REPORT-123")
    
    # Abstract input with larger text area
    abstract = st.text_area("Abstract", height=200, 
                           placeholder="Enter the paper abstract here. This is the main text that will be used for summarization.")
    
    categories = st.text_input("Categories", placeholder="e.g., cs.CL, cs.AI")
    
    # Summary options
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum Summary Length", 50, 150, 96)
    with col2:
        min_length = st.slider("Minimum Summary Length", 10, 50, 20)
    
    # Submit button
    submitted = st.form_submit_button("Generate Summary")

# Process form submission
if submitted:
    if abstract.strip():  # Check if abstract is provided (the most important field)
        # Create a data dictionary from inputs
        paper_data = {
            'id': paper_id,
            'submitter': submitter,
            'authors': authors,
            'title': title,
            'comments': comments,
            'journal-ref': journal_ref,
            'doi': doi,
            'report-no': report_no,
            'abstract': abstract,
            'categories': categories
        }
        
        # Display paper details in a nice format
        st.subheader("Paper Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**ID:** {paper_data['id']}")
            st.markdown(f"**Title:** {paper_data['title']}")
            st.markdown(f"**Authors:** {paper_data['authors']}")
            st.markdown(f"**Categories:** {paper_data['categories']}")
        
        with col2:
            st.markdown(f"**Submitter:** {paper_data['submitter']}")
            st.markdown(f"**Journal Ref:** {paper_data['journal-ref']}")
            st.markdown(f"**DOI:** {paper_data['doi']}")
            st.markdown(f"**Report No:** {paper_data['report-no']}")
        
        # Display the abstract
        with st.expander("Show Abstract", expanded=False):
            st.write(paper_data['abstract'])
        
        # Generate summary if model is loaded
        if 'summarizer' in st.session_state:
            with st.spinner("Generating summary..."):
                try:
                    # Get the summarizer
                    summarizer = st.session_state['summarizer']
                    
                    # Combine title and abstract for better context
                    input_text = f"{paper_data['title']} {paper_data['abstract']}" if paper_data['title'] else paper_data['abstract']
                    
                    # Generate summary using the model and tokenizer
                    summary, processing_time = generate_summary(
                        input_text, 
                        summarizer, 
                        max_length=max_length,
                        min_length=min_length
                    )
                    
                    # Display the summary
                    st.subheader("Generated Summary")
                    
                    # Create a styled container for the summary
                    st.markdown(
                                    f"""
                                    <div style="background-color: #2c3e50; padding: 20px; 
                                                border-radius: 10px; border-left: 5px solid #3498db; 
                                                color: white; font-weight: 500;">
                                        {summary}
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                    
                    # Show word count statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Abstract Word Count", len(abstract.split()))
                    with col2:
                        st.metric("Summary Word Count", len(summary.split()))
                    with col3:
                        if len(abstract.split()) > 0:
                            compression = round((1 - len(summary.split()) / len(abstract.split())) * 100)
                            st.metric("Compression Rate", f"{compression}%")
                    with col4:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    
                    # Allow downloading the summary
                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name=f"{paper_data['id'] if paper_data['id'] else 'arxiv'}_summary.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
                    st.info("Please check your inputs and try again.")
        else:
            st.error("Summarization model is not loaded. Please check your installation.")
    else:
        st.error("Abstract is required for generating a summary. Please enter the paper abstract.")

# Add some example data to help users
with st.expander("Need an example?"):
    st.markdown("""
    ### Example Paper Data
    
    **ID**: 2104.08663
    
    **Title**: Attention is All You Need
    
    **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
    
    **Abstract**: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.
    
    **Categories**: cs.CL, cs.AI, cs.LG
    
    (You can copy and paste this data into the form above to test the summarization)
    """)

    # Add a quick-fill button
    if st.button("Fill Example Data"):
        st.session_state['example_clicked'] = True

# If example button was clicked, fill the form (this happens after the page reloads)
if 'example_clicked' in st.session_state and st.session_state['example_clicked']:
    # Clear the flag
    st.session_state['example_clicked'] = False
    # Rerun with the example data in the session state
    st.session_state['paper_id'] = "2104.08663"
    st.session_state['title'] = "Attention is All You Need"
    st.session_state['authors'] = "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin"
    st.session_state['abstract'] = "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature."
    st.session_state['categories'] = "cs.CL, cs.AI, cs.LG"
    st.rerun()

# Add footer
st.markdown("---")
st.markdown("Research Paper Summarizer App | ©Gradient Geeks | Created with Streamlit 👑")
