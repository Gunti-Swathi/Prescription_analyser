from __future__ import annotations
import os
import base64
from typing import List
from datetime import date, datetime
import requests
from PIL import Image
import shutil
from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain import globals
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain.chains import retrieval_qa
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
# Initialize Pinecone API Key and Google API Key

os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

os.environ["GOGGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
globals.set_debug(False)



class MedicationItem(BaseModel):
    name: str
    dosage: str
    frequency: str
    duration: str

class PrescriptionInformations(BaseModel):
    patient_name: str = Field(description="Patient's name")
    patient_age: int = Field(description="Patient's age")
    patient_gender: str = Field(description="Patient's gender")
    doctor_name: str = Field(description="Doctor's name")
    doctor_license: str = Field(description="Doctor's license number")
    prescription_date: datetime = Field(description="Date of the prescription")
    medications: List[MedicationItem] = []
    additional_notes: str = Field(description="Additional notes or instructions")

def load_images(inputs: dict) -> dict:
    image_paths = inputs["image_paths"]

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    images_base64 = [encode_image(image_path) for image_path in image_paths]
    return {"images": images_base64}

load_images_chain = TransformChain(
    input_variables=["image_paths"],
    output_variables=["images"],
    transform=load_images
)

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with images and prompt."""
    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-1.5-pro",temperature=0.4)
    image_urls = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img in inputs['images']]
    prompt = """
    # Medical Transcriptionist Task: Handwritten Prescription Analysis

You are an expert medical transcriptionist specializing in deciphering handwritten medical prescriptions with utmost accuracy. Your task is to carefully examine the provided prescription images and extract all relevant information, ensuring the highest level of precision and organization.

Please follow the format outlined below to present the transcribed information:

### Expected Output Format:

#### Example 1:

- **Patient's Full Name:**  John Doe  
- **Patient's Age:**  45 /45y  
- **Patient's Gender:**  M / Male  
- **Doctor's Full Name:**  Dr. Jane Smith  
- **Doctor's License Number:**  ABC123456  
- **Prescription Date:**  2023-04-01  

**Medications:**

- **Medication Name:**  Amoxicillin  
  - **Dosage:**  500 mg  
  - **Frequency:**  Twice a day  
  - **Duration:**  7 days  

- **Medication Name:**  Ibuprofen  
  - **Dosage:  200 mg  
  - **Frequency:  Every 4 hours as needed  
  - **Duration:  5 days  

Additional Notes: 
- Take medications with food.  
- Drink plenty of water.

#### Example 2:

- Patient's Full Name:  Jane Roe  
- Patient's Age:  60 / 60y  
- Patient's Gender:  F / Female  
- Doctor's Full Name:  Dr. John Doe  
- Doctor's License Number:  XYZ654321  
- Prescription Date:  2023-05-10  

**Medications:

- Medication Name:  Metformin  
  - Dosage: 850 mg  
  - Frequency:  Once a day  
  - Duration:  30 days  

**Additional Notes: 
- Monitor blood sugar levels daily.  
- Avoid sugary foods.

---

### Instructions:

1. **Patient Information:**  
   - Extract the **Patient's Full Name** (First and Last).  
   - Extract the **Patient's Age** and handle different formats (e.g., "42y", "42yrs", "42 years").  
   - Extract the **Patient's Gender** (e.g., Male/Female).  

2. **Doctor Information:**  
   - Extract the **Doctor's Full Name** (First and Last).  
   - Extract the **Doctor's License Number**.  

3. **Prescription Date:**  
   - Extract the **Prescription Date** in the **YYYY-MM-DD** format.

4. **Medications:**  
   - For each medication, extract:  
     - **Medication Name**  
     - **Dosage**  
     - **Frequency**  
     - **Duration**  

5. **Additional Notes:**  
   - Extract and enhance any additional instructions or notes provided.  
   - Organize the notes in **bullet points** for better clarity.  
   - Provide headings where applicable (e.g., "Medication Instructions", "Dietary Notes") and list sub-points under each category.  
   - Make sure the notes are **well-organized** with clear **tab spaces** and **bold text** to enhance readability.  

6. **Image Enhancement:**  
   - Before extracting the information, **enhance the image** if necessary, by adjusting brightness, contrast, or applying filters to improve clarity.

7. **Accuracy:**  
   - Ensure **all information** is accurately extracted. If any information is **illegible** or **missing**, mark it as "Not available."  
   - Do not guess or infer information that is not clearly visible.

---

### Prescription Images:  
{images_content}

Ensure that the transcription is presented in a clean, organized, and **easily readable** format using **bold text** for important sections and **proper tab spaces** for clear separation between categories.

    """
    msg = model.invoke(
    [HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "text", "text": parser.get_format_instructions()},
            *image_urls
        ]
    )],
       
    stop=None,  
    )
    return msg.content

def get_prescription_informations(image_paths: List[str]) -> dict:
    parser = JsonOutputParser(pydantic_object=PrescriptionInformations)
    vision_prompt = """
    Given the images, provide all available information including:
    - Patient's name, age, and gender
    - Doctor's name and license number
    - Prescription date
    - List of medications with name, dosage, frequency, and duration
    - Additional notes or instructions
    Note: If portions of the image are not clear then leave the values as empty. Do not make up the values.
    """
    vision_chain = load_images_chain | image_model | parser
    return vision_chain.invoke({'image_paths': image_paths, 'prompt': vision_prompt})


def display_prescription_details_in_table(final_result: dict):
    # Prepare patient details in a table format
    prescription_date = final_result.get('prescription_date', 'Not available')
    
    # Check if 'prescription_date' is a datetime object, if not try to parse it
    if isinstance(prescription_date, datetime):
        formatted_date = prescription_date.strftime('%Y-%m-%d')
    elif isinstance(prescription_date, str):
        try:
            # Try parsing the string to a datetime object
            formatted_date = datetime.strptime(prescription_date, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError:
            formatted_date = 'Not available'
    else:
        formatted_date = 'Not available'

    patient_info = {
        "Patient's Full Name": final_result.get('patient_name', 'Not available'),
        "Patient's Age": final_result.get('patient_age', 'Not available'),
        "Patient's Gender": final_result.get('patient_gender', 'Not available'),
        "Doctor's Full Name": final_result.get('doctor_name', 'Not available'),
        "Doctor's License Number": final_result.get('doctor_license', 'Not available'),
        "Prescription Date": formatted_date,
    }

    # Create a DataFrame for patient details
    patient_df = pd.DataFrame(list(patient_info.items()), columns=["Field", "Value"])

    # Prepare medications in a table format
    medications_info = []
    for med in final_result.get('medications', []):
        medications_info.append({
            "Medication Name": med['name'],
            "Dosage": med['dosage'],
            "Frequency": med['frequency'],
            "Duration": med['duration']
        })
    
    medications_df = pd.DataFrame(medications_info)

    # Prepare additional notes in a table format
    additional_notes = final_result.get('additional_notes', 'Not available')
    additional_notes_df = pd.DataFrame([{"Notes": additional_notes}])

    # Display tables
    st.subheader("Patient Information")
    st.table(patient_df)

    if not medications_df.empty:
        st.subheader("Medications")
        st.table(medications_df)

    if not additional_notes_df.empty:
        st.subheader("Additional Notes")
        st.table(additional_notes_df)


# Datapreprocessing
load_dotenv()
data1=CSVLoader("Medicine_Details.csv")
data_csv=data1.load()
def load_pdf_file(data):
    loader=DirectoryLoader(data,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents
data_pdf=load_pdf_file(data="DataResources/")
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
chunks_pdf=splitter.split_documents(data_pdf)
chunks_csv=splitter.split_documents(data_csv)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
index_name = "pharmabot"

pc= Pinecone(api_key=PINECONE_API_KEY)
index_name = "pharmabot"

pc.create_index(
    name=index_name,
    dimension=768, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)



# Connect to the existing Pinecone index
docs = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docs.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-pro',
    temperature=0
)

system_prompt = (
    "You are an assistant for question answering tasks. Use the following piece of retrieved context to respond to queries."
    "You should be able to answer questions related to the following:"
    "Search for detailed information about medicines, including their image, composition, uses, side effects, manufacturer, and reviews."
    "Provide only the relevant information requested and avoid generating unnecessary details. If you don’t know the answer, state that you don’t know. Use a maximum of three sentences and keep the response concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


def main():
    st.set_page_config(layout="wide")
    def encode_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    # Encode the local background image
    background_image_base64 = encode_image("Background.jpg")  # Provide your local image path

    # Background image with a glassy effect
    st.markdown(
        f"""
        <style>
            body {{
                background-image: url('data:image/jpeg;base64,{background_image_base64}');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .stApp {{
                background: rgba(255, 255, 255, 0);  /* Semi-transparent white background for the content */
                border-radius: 10px;
                padding: 20px;
                backdrop-filter: blur(8px); /* Frosted glass effect */
            }}
            h1 {{
                color: white;
                font-family: Arial, sans-serif;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    # Encode the local image
    logo_base64 = encode_image("pharmacy.png")

    # Center the title and logo together
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" alt="Medical Logo" width="60" style="margin-right: 15px;">
            <h1 style="color: #000000; font-family: Segoe UI Black, sans-serif; margin: 0;">PHARMABOT</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])  # Create two columns

    # Prescription processing section
    with col1:
        global parser
        parser = JsonOutputParser(pydantic_object=PrescriptionInformations)

        uploaded_file = st.file_uploader("Upload a Prescription image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            if "processed_image" not in st.session_state or st.session_state.processed_image != uploaded_file.name:
                # New image uploaded, process it
                st.session_state.processed_image = uploaded_file.name  # Mark the image as processed
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = uploaded_file.name.split('.')[0].replace(' ', '_')
                output_folder = os.path.join(".", f"Check_{filename}_{timestamp}")
                os.makedirs(output_folder, exist_ok=True)

                check_path = os.path.join(output_folder, uploaded_file.name)
                with open(check_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

               # Process and save the results in session state
                with st.spinner('Processing Prescription...'):
                   final_result = get_prescription_informations([check_path]) 

                st.session_state.processed_result = {
                    "image": uploaded_file,  # Save the uploaded image
                    "details": final_result,  # Save the extracted details
               }

        # Display processed results if available
        if "processed_result" in st.session_state:
            col3, col4 = st.columns([1, 2])  # Create two columns

            with col3:  # Left column for the result
               st.subheader("Prescription Image")
               st.image(st.session_state.processed_result["image"], caption='Uploaded Prescription Image.', use_column_width=True)

            with col4:  # Right column for the image
               st.subheader("Extracted Prescription Details")
               display_prescription_details_in_table(st.session_state.processed_result["details"])

    

    with col2:
        st.subheader("Ask about a medicine:")

        # Input box for user question
        question = st.text_input("", placeholder="Type your question here")

        # When user presses Enter, process the question and fetch the response
        if question:
            # Prepare the input query for the LangChain system
            input_query = {"input": question}

            # Invoke the LangChain retrieval chain to get the response
            response = rag_chain.invoke(input_query)

            # # Display the question
            # st.subheader("Your Question:")
            # st.write(question)

            # Display the response
            st.subheader("Response:")
            full_response = response["answer"]
            final_text = full_response
            image_url = next(
                (part for part in full_response.split() if part.startswith("http") and any(ext in part for ext in [".jpg", ".png", ".jpeg"])),
                None
            )
            if image_url:
                if(image_url[-1] == "."):
                    image_url = image_url[:-1]
                response = requests.get(image_url[:-1], stream=True)
                img = Image.open(response.raw)
                try:
                    st.image(img, caption="Medicine Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Failed to load image. Error: {str(e)}")
            else:
                st.markdown(final_text)

            # Show Clear Conversation button after response is shown
            if st.button("Clear Conversation"):
                
                st.experimental_rerun()





if __name__ == "__main__":
    main()