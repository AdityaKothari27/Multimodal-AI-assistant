import streamlit as st
from PIL import Image
import io
import base64
import os
import tempfile
import time
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI with API key
genai.configure(api_key=api_key)

# Page configuration
st.set_page_config(
    page_title="Document Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history

if "document_content" not in st.session_state:
    st.session_state.document_content = None  # Store document text and images

if "document_name" not in st.session_state:
    st.session_state.document_name = None  # Store document name

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Function to extract text and images from PDF
def extract_from_pdf(file_path):
    document_content = []
    
    # Open the PDF
    doc = fitz.open(file_path)
    
    # Process each page
    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text()
        if text.strip():
            document_content.append({
                "type": "text",
                "content": text,
                "page": page_num + 1
            })
        
        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            document_content.append({
                "type": "image",
                "content": image,
                "page": page_num + 1,
                "index": img_index
            })
    
    return document_content

# Function to reset chat
def reset_chat():
    st.session_state.messages = []
    st.session_state.document_content = None
    st.session_state.document_name = None

# Function to display PDF
def display_pdf(file_bytes, file_name):
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="500px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"### Preview of {file_name}")
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to get response from Gemini
def get_gemini_response(query, context=None, image=None):
    try:
        # Configure the model
        generation_config = {
            "temperature": 0.4,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 2048,
        }
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Initialize the model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Create the prompt
        if context:
            prompt = f"""
            Based on the following information from the document, please answer the query: {query}
            
            Document information:
            {context}
            
            Please provide a concise and accurate answer based only on the information provided above.
            If the information doesn't contain the answer, please say "I don't have enough information to answer this question."
            """
            
            # Generate content (text only)
            response = model.generate_content(prompt)
            return response.text
        
        elif image:
            # Prepare the image
            image_data = encode_image(image)
            
            prompt = f"""
            Analyze this image from the document and answer the following question: {query}
            
            Extract all relevant information from the image that helps answer the question.
            If you can't find relevant information in the image, please say "I don't see information in this image that answers your question."
            """
            
            # Generate content with image
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": base64.b64decode(image_data)}
            ])
            return response.text
        
        else:
            return "I don't have enough information to answer this question."
            
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # If a new file is uploaded
        if st.session_state.document_name != uploaded_file.name:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                with st.spinner("Processing document... Please wait..."):
                    # Extract content from PDF
                    st.session_state.document_content = extract_from_pdf(temp_file_path)
                    st.session_state.document_name = uploaded_file.name
            
            st.success(f"Document processed: {uploaded_file.name}")
            
            # Add a welcome message to the chat
            if not st.session_state.messages:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"I've processed your document '{uploaded_file.name}'. What would you like to know about it?"
                })
        
        # Display the PDF preview
        display_pdf(uploaded_file.getvalue(), uploaded_file.name)
    
    # Reset button
    if st.button("Reset Chat 🗑️"):
        reset_chat()
        st.rerun()

# Main chat interface
st.title("Chat with your Document")
st.markdown("Upload a PDF and chat with it like you're talking to a helpful assistant!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask something about your document..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if st.session_state.document_content:
            with st.spinner("Thinking..."):
                # First try to find relevant text content
                relevant_text = ""
                for item in st.session_state.document_content:
                    if item["type"] == "text" and any(keyword in item["content"].lower() for keyword in prompt.lower().split()):
                        relevant_text += f"[Page {item['page']}]: {item['content']}\n\n"
                
                # If we found relevant text, use it
                if relevant_text:
                    result = get_gemini_response(prompt, context=relevant_text)
                else:
                    # If no relevant text, check if there's a question about images
                    image_related = any(word in prompt.lower() for word in ["image", "picture", "photo", "figure", "diagram", "chart", "graph", "illustration"])
                    
                    if image_related and any(item["type"] == "image" for item in st.session_state.document_content):
                        # Use the first few images
                        images = [item for item in st.session_state.document_content if item["type"] == "image"][:3]
                        image_results = []
                        
                        for img_item in images:
                            img_result = get_gemini_response(prompt, image=img_item["content"])
                            image_results.append(f"[Image on page {img_item['page']}]: {img_result}")
                        
                        result = "\n\n".join(image_results)
                    else:
                        # If no specific content found, use all text content (limited)
                        all_text = "\n\n".join([f"[Page {item['page']}]: {item['content']}" 
                                              for item in st.session_state.document_content 
                                              if item["type"] == "text"][:5])  # Limit to first 5 text blocks
                        
                        result = get_gemini_response(prompt, context=all_text)
        else:
            result = "Please upload a document first so I can help you with your questions."
        
        # Add cat-like personality to the response
    
        
        import random
        if not result.startswith("Error"):
            result = result
        
        # Display the response with typing effect
        full_response = ""
        lines = result.split('\n')
        for i, line in enumerate(lines):
            full_response += line
            if i < len(lines) - 1:
                full_response += '\n'
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.05)
        
        message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result})

