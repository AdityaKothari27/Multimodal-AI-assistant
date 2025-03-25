import streamlit as st
import google.generativeai as genai
from google.genai import types
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image
import io
import base64
import os
import tempfile
import time
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Set page configuration
st.set_page_config(
    page_title="Gemini Multi-Tool",
    page_icon="🤖",
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

# Initialize Google Genai client
@st.cache_resource
def get_client():
    return genai.Client(api_key=api_key)

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

# Function to get response from Gemini for chat
def get_gemini_chat_response(query, context=None, image=None):
    try:
        client = get_client()
        
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
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                
            
            )
            return response.text
        
        elif image:
            # Generate content with image
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    f"Analyze this image from the document and answer the following question: {query}\n\nExtract all relevant information from the image that helps answer the question.",
                    image
                ],
                
               
            )
            return response.text
        
        else:
            # General conversation without document context
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=query,
                
       
            )
            return response.text
            
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Function to generate image from text prompt
def generate_image_from_text(client, prompt):
    """Generate an image using Gemini model based on text prompt only"""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )
        
        # Return both text response and generated image
        text_response = None
        image_data = None
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                text_response = part.text
            elif part.inline_data is not None:
                image_data = part.inline_data.data
                
        return text_response, image_data
    
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None, None

# Function to edit image based on prompt
def edit_image(client, prompt, uploaded_image):
    """Edit an image using Gemini model based on prompt and input image"""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[prompt, uploaded_image],
            config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )
        
        # Return both text response and generated image
        text_response = None
        image_data = None
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                text_response = part.text
            elif part.inline_data is not None:
                image_data = part.inline_data.data
                
        return text_response, image_data
    
    except Exception as e:
        st.error(f"Error editing image: {str(e)}")
        return None, None

# Function to get image download link
def get_image_download_link(img_data, filename, text):
    """Generate a download link for the image"""
    b64 = base64.b64encode(img_data).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download {text}</a>'
    return href

# Sidebar for navigation
with st.sidebar:
    st.title("Gemini Multi-Tool")
    
    
    # Navigation
    app_mode = st.radio(
        "Choose Mode:",
        ["Image Generation", "Image Editing", "Chat"]
    )
    
    # Reset button
    if st.button("Reset"):
        reset_chat()
        st.session_state.document_content = None
        st.session_state.document_name = None
        st.rerun()

    st.write("Made by [Aditya Kothari](https://github.com/adityakothari27)")
    

# Main app logic based on selected mode
if app_mode == "Image Generation":
    st.header("Gemini Image Generation")
    st.write("Generate an image based on your text prompt.")
    
    # Prompt input
    prompt = st.text_area(
        "Enter your prompt for image generation", 
        placeholder="e.g., 'A realistic landscape with mountains and a lake at sunset'", 
        height=100
    )
    
    # Generate image button
    if prompt and st.button("Generate Image"):
        with st.spinner("Generating image..."):
            # Get client
            client = get_client()
            
            # Generate image
            text_response, generated_image_data = generate_image_from_text(client, prompt)
            
            # Display generated content
            if generated_image_data:
                st.subheader("Generated Image")
                generated_image = Image.open(io.BytesIO(generated_image_data))
                st.image(generated_image, use_container_width=True)
                
                # Download button
                st.markdown(
                    get_image_download_link(generated_image_data, "generated_image.png", "Generated Image"),
                    unsafe_allow_html=True
                )
            else:
                st.error("Failed to generate image")
            
            # Display model's text response if any
            if text_response:
                st.subheader("Model Response")
                st.write(text_response)

elif app_mode == "Image Editing":
    st.header("Gemini Image Editing")
    st.write("Upload an image and provide a prompt to edit/transform it.")
    
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image to edit", type=["jpg", "jpeg", "png"])
    
    # Prompt input
    prompt = st.text_area(
        "Enter your prompt for image editing", 
        placeholder="e.g., 'Add a llama next to me in this picture'", 
        height=100
    )
    
    # Process the image when both image and prompt are provided
    if uploaded_file is not None and prompt and st.button("Edit Image"):
        with st.spinner("Processing image..."):
            # Get client
            client = get_client()
            
            # Load and display original image
            original_image = Image.open(uploaded_file)
            
            # Create two columns for before/after display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(original_image, use_container_width=True)
            
            # Edit image
            text_response, edited_image_data = edit_image(client, prompt, original_image)
            
            # Display generated content
            with col2:
                if edited_image_data:
                    st.subheader("Edited Image")
                    edited_image = Image.open(io.BytesIO(edited_image_data))
                    st.image(edited_image, use_container_width=True)
                    
                    # Download button
                    st.markdown(
                        get_image_download_link(edited_image_data, "edited_image.png", "Edited Image"),
                        unsafe_allow_html=True
                    )
                else:
                    st.error("Failed to edit image")
            
            # Display model's text response if any
            if text_response:
                st.subheader("Model Response")
                st.write(text_response)

else:  # Document Chat
    st.header("Chat with Documents or Gemini")
    
    # Document upload in sidebar
    with st.sidebar:
        st.header("Upload Document (Optional)")
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
        else:
            if not st.session_state.messages:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Hi! I'm Gemini. Ask me anything or upload a document to chat about it."
                })
    
    # Main chat interface
    st.markdown("Chat with Gemini or upload a document to ask questions about it.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                # If a document is uploaded, try to find relevant content
                if st.session_state.document_content:
                    # First try to find relevant text content
                    relevant_text = ""
                    for item in st.session_state.document_content:
                        if item["type"] == "text" and any(keyword in item["content"].lower() for keyword in prompt.lower().split()):
                            relevant_text += f"[Page {item['page']}]: {item['content']}\n\n"
                    
                    # If we found relevant text, use it
                    if relevant_text:
                        result = get_gemini_chat_response(prompt, context=relevant_text)
                    else:
                        # If no relevant text, check if there's a question about images
                        image_related = any(word in prompt.lower() for word in ["image", "picture", "photo", "figure", "diagram", "chart", "graph", "illustration"])
                        
                        if image_related and any(item["type"] == "image" for item in st.session_state.document_content):
                            # Use the first few images
                            images = [item for item in st.session_state.document_content if item["type"] == "image"][:3]
                            image_results = []
                            
                            for img_item in images:
                                img_result = get_gemini_chat_response(prompt, image=img_item["content"])
                                image_results.append(f"[Image on page {img_item['page']}]: {img_result}")
                            
                            result = "\n\n".join(image_results)
                        else:
                            # If no specific content found, use all text content (limited)
                            all_text = "\n\n".join([f"[Page {item['page']}]: {item['content']}" 
                                                  for item in st.session_state.document_content 
                                                  if item["type"] == "text"][:5])  # Limit to first 5 text blocks
                            
                            result = get_gemini_chat_response(prompt, context=all_text)
                else:
                    # No document uploaded, use general conversation
                    result = get_gemini_chat_response(prompt)
            
            # Display the response with typing effect
            full_response = ""
            lines = result.split('\n')
            for i, line in enumerate(lines):
                full_response += line
                if i < len(lines) - 1:
                    full_response += '\n'
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)  # Speed up the typing effect
            
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": result})

if __name__ == "__main__":
    main = None  # The main app logic is run directly in the script