import streamlit as st
from PIL import Image
import io
import base64
import os
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
    page_title="Gemma-3 OCR",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Title and description in main area
# st.markdown("""
#     # <img src="data:image/png;base64,{}" width="50" style="vertical-align: -12px;"> Gemma-3 OCR
# """.format(base64.b64encode(open("./assets/gemma3.png", "rb").read()).decode()), unsafe_allow_html=True)

# Add clear button to top right
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract structured text from images using Gemma-3 Vision!</p>', unsafe_allow_html=True)
st.markdown("---")

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Move upload controls to sidebar
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        
        if st.button("Extract Text 🔍", type="primary"):
            with st.spinner("Processing image..."):
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
                        model_name="Gemma3-27B",
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    # Prepare the image
                    image_data = encode_image(image)
                    
                    # Create the prompt
                    prompt = """Analyze the text in the provided image. Extract all readable content
                    and present it in a structured Markdown format that is clear, concise, 
                    and well-organized. Ensure proper formatting (e.g., headings, lists, or
                    code blocks) as necessary to represent the content effectively."""
                    
                    # Generate content
                    response = model.generate_content(
                        [
                            prompt,
                            {"mime_type": f"image/jpeg", "data": base64.b64decode(image_data)}
                        ]
                    )
                    
                    st.session_state['ocr_result'] = response.text
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

# Main content area for results
if 'ocr_result' in st.session_state:
    st.markdown(st.session_state['ocr_result'])
else:
    st.info("Upload an image and click 'Extract Text' to see the results here.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Gemma-3 Vision Model")










# import streamlit as st
# import os
# import tempfile
# import gc
# import base64
# import time
# from langchain_google_genai import ChatGoogleGenerativeAI
# from src.agentic_rag.tools.custom_tool import DocumentSearchTool
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# api_token = os.getenv("GOOGLE_API_KEY")

# @st.cache_resource
# def load_llm():
#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_token)
#     return llm

# # ===========================
# #   Document QA Function
# # ===========================
# def get_document_qa_response(query, pdf_tool, llm):
#     """Get response from document search and Gemini LLM"""
#     # Search the document
#     search_results = pdf_tool._run(query)
    
#     # If no results, return a message
#     if not search_results or search_results == "Document is still being processed...":
#         return "I'm sorry, I couldn't find the information you're looking for in the document."
    
#     # Create prompt for the LLM
#     prompt = f"""
#     Based on the following information from the document, please answer the query: {query}
    
#     Document information:
#     {search_results}
    
#     Please provide a concise and accurate answer based only on the information provided above.
#     If the information doesn't contain the answer, please say "I don't have enough information to answer this question."
#     """
    
#     # Get response from LLM
#     response = llm.invoke(prompt)
#     return response.content

# # ===========================
# #   Streamlit Setup
# # ===========================
# if "messages" not in st.session_state:
#     st.session_state.messages = []  # Chat history

# if "pdf_tool" not in st.session_state:
#     st.session_state.pdf_tool = None  # Store the DocumentSearchTool

# def reset_chat():
#     st.session_state.messages = []
#     gc.collect()

# def display_pdf(file_bytes: bytes, file_name: str):
#     """Displays the uploaded PDF in an iframe."""
#     base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
#     pdf_display = f"""
#     <iframe 
#         src="data:application/pdf;base64,{base64_pdf}" 
#         width="100%" 
#         height="600px" 
#         type="application/pdf"
#     >
#     </iframe>
#     """
#     st.markdown(f"### Preview of {file_name}")
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # ===========================
# #   Sidebar
# # ===========================
# with st.sidebar:
#     st.header("Add Your PDF Document")
#     uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

#     if uploaded_file is not None:
#         # If there's a new file and we haven't set pdf_tool yet...
#         if st.session_state.pdf_tool is None:
#             with tempfile.TemporaryDirectory() as temp_dir:
#                 temp_file_path = os.path.join(temp_dir, uploaded_file.name)
#                 with open(temp_file_path, "wb") as f:
#                     f.write(uploaded_file.getvalue())

#                 with st.spinner("Indexing PDF... Please wait..."):
#                     st.session_state.pdf_tool = DocumentSearchTool(file_path=temp_file_path)
            
#             st.success("PDF indexed! Ready to chat.")

#         # Display the PDF in the sidebar
#         display_pdf(uploaded_file.getvalue(), uploaded_file.name)

#     st.button("Clear Chat", on_click=reset_chat)

# # ===========================
# #   Main Chat Interface
# # ===========================
# st.title("Document QA with Google Gemini")
# st.markdown("Upload a PDF document and ask questions about its content.")

# # Render existing conversation
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input
# prompt = st.chat_input("Ask a question about your PDF...")

# if prompt:
#     # 1. Show user message immediately
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # 2. Get the response
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
        
#         # Get the complete response first
#         with st.spinner("Thinking..."):
#             if st.session_state.pdf_tool:
#                 result = get_document_qa_response(prompt, st.session_state.pdf_tool, load_llm())
#             else:
#                 result = "Please upload a PDF document first."
        
#         # Split by lines first to preserve code blocks and other markdown
#         lines = result.split('\n')
#         for i, line in enumerate(lines):
#             full_response += line
#             if i < len(lines) - 1:  # Don't add newline to the last line
#                 full_response += '\n'
#             message_placeholder.markdown(full_response + "▌")
#             time.sleep(0.15)  # Adjust the speed as needed
        
#         # Show the final response without the cursor
#         message_placeholder.markdown(full_response)

#     # 3. Save assistant's message to session
#     st.session_state.messages.append({"role": "assistant", "content": result})