
## **Abstract**

The healthcare industry often faces significant challenges in interpreting handwritten prescriptions, primarily due to the illegible and inconsistent handwriting commonly used by healthcare professionals. This issue can lead to misinterpretation, resulting in delays or errors in medication understanding, ultimately affecting patient care and pharmacist efficiency. To address these challenges, we propose "PharmaBot," a comprehensive solution designed to automate the extraction, organization, and interpretation of handwritten prescription data.

PharmaBot leverages GEMINI optical character recognition (OCR) technology, with the assistance of the GEMINI vision bot, to extract pertinent information from handwritten prescriptions, converting unstructured content into a clear, structured format. The system organizes this extracted data into categories such as medication names, dosages, instructions, and other relevant details. Additionally, PharmaBot incorporates an intelligent chatbot powered by a rule-based approach using Retrieval-Augmented Generation (RAG) to enhance its ability to respond accurately to patient inquiries. The chatbot is capable of answering common questions, such as the purpose of specific medications, potential side effects, visual representations of drugs, and available alternatives with similar compositions.

The integration of RAG ensures that the chatbot generates reliable and contextually relevant responses by utilizing a combination of knowledge retrieval and natural language generation. This innovative system aims to streamline the prescription interpretation process, minimize human errors, and empower patients and pharmacists by providing a reliable, user-friendly interface for medication-related inquiries.


# **Introduction**

Welcome to the **Medical Prescription Classification System**! This project is an innovative blend of computer vision and large language model (LLM) techniques, designed to revolutionize how medical prescriptions are analyzed and interpreted. 

## **Problem Statement**
Medical prescriptions are a cornerstone of healthcare, yet their handwritten and often inconsistent formats pose significant challenges. Misinterpretation of prescriptions can lead to medication errors, affecting patient safety and treatment efficacy. Current solutions are either too rigid or lack the adaptability to handle the diversity of prescription formats.

## **Objective**
This project aims to:
1. **Digitize and Interpret Prescriptions**: Extract handwritten and printed text from prescription images with high accuracy.
2. **Drug Identification**: Recognize drug names within prescriptions and identify their intended uses.
3. **Contextual Insights via LLMs**: Provide a conversational interface that explains the purpose, usage, and potential side effects of the identified drugs using comprehensive medical knowledge.
4. **Empower Healthcare Systems**: Reduce errors, save time, and improve the overall efficiency of prescription processing.

## **Why This Project?**
Medical prescription analysis is a multifaceted challenge requiring robust solutions that can bridge the gap between raw image data and actionable insights. By combining state-of-the-art computer vision techniques with the contextual understanding of LLMs, this system provides a comprehensive tool for healthcare professionals and patients alike. 

Through this project, we aim to not only enhance the accuracy of prescription digitization but also enable informed decision-making, improving the quality of healthcare delivery worldwide.



## **Model Architecture of PharmaBot**

<img width="776" height="471" alt="image" src="https://github.com/user-attachments/assets/7fbc820f-cb62-4e11-b8c5-220998acd6a2" />


The **PharmaBot** system is designed to automate the extraction and organization of information from handwritten prescriptions and provide an intelligent, context-aware chatbot to answer user queries about medications. The architecture is composed of several key components that work in harmony to extract, process, store, retrieve, and generate relevant answers for medication-related queries.

### 1. **Image Preprocessing and Data Extraction**
   - **GEMINI Optical Character Recognition (OCR)** with **GEMINI Vision Bot**:
     - **Purpose**: The initial step is to extract textual information from handwritten prescription images. This is crucial since handwritten prescriptions can be challenging to read and interpret.
     - **Details**: Using **GEMINI OCR** technology and the **GEMINI vision bot**, prescription images are scanned and converted into machine-readable text. This includes the extraction of patient information, doctor details, medication names, dosages, and any special instructions.
     - **Workflow**: The images are encoded in base64 format and passed through the OCR model, which returns structured data containing the necessary prescription details.

### 2. **Data Structuring and Organization**
   - **Medication Item Class**:
     - **Purpose**: Organize and structure extracted medication data for easier processing.
     - **Details**: The `MedicationItem` class encapsulates key information about the medications prescribed, such as:
       - `name`: The name of the medication.
       - `dosage`: The prescribed dosage.
       - `frequency`: How often the medication should be taken.
       - `duration`: Duration for taking the medication.
     - **Impact**: This class ensures medication data is standardized, which simplifies subsequent data manipulation and retrieval.

   - **PrescriptionInformations Class**:
     - **Purpose**: Structure the entire prescription data, including both personal information and medication details.
     - **Details**: This class includes:
       - Patient and doctor information (e.g., `name`, `age`, `gender`).
       - The list of medications prescribed, each represented by `MedicationItem`.
       - Additional notes and instructions provided by the prescribing doctor.
     - **Impact**: Centralizing all prescription-related information in one container improves the system’s ability to process and query the data effectively.

### 3. **Embedding Generation and Data Ingestion**
   - **Ingestion of Medicine Information (CSV)**:
     - **Purpose**: Ingest detailed data about medicines, such as their name, composition, uses, side effects, and manufacturers.
     - **Details**: The data is typically stored in a `.csv` file, which is parsed and processed into smaller chunks. This chunking process ensures the data is manageable and efficient for embedding generation.
   
   - **Recursive Character Text Splitting**:
     - **Purpose**: Split large chunks of text into smaller, manageable pieces that the embedding model can handle efficiently.
     - **Details**: The recursive text-splitting method breaks down the large textual data into smaller segments to ensure that the semantic meaning is preserved and the embeddings are more accurately generated.
     - **Impact**: By breaking the data into smaller chunks, the system can create more efficient embeddings and facilitate more accurate retrieval during question answering.

   - **Embedding Generation with Google’s Generative AI Model (embedding-001)**:
     - **Purpose**: Convert the textual data into high-dimensional embeddings.
     - **Details**: Google's **Generative AI embeddings model (embedding-001)** is used to generate embeddings that capture the semantic meaning of each text chunk. These embeddings are high-dimensional numerical vectors representing the content of the text.
     - **Impact**: Embeddings enable efficient similarity search, allowing the system to quickly retrieve the most relevant information based on user queries.

   - **Storage in Pinecone Vector Database**:
     - **Purpose**: Store the generated embeddings in a vector database for efficient similarity search.
     - **Details**: **Pinecone** is used to store the embeddings, allowing the system to leverage its indexing and retrieval capabilities. Pinecone is hosted on AWS services to ensure high availability, low latency, and scalability.
     - **Impact**: This setup provides a powerful foundation for performing efficient searches, ensuring that the most relevant information is retrieved during query processing.

### 4. **Retrieval-Augmented Generative (RAG) Pipeline**
   - **Retriever for Similarity Search**:
     - **Purpose**: Retrieve the most relevant chunks of data from Pinecone based on user queries.
     - **Details**: The retriever employs a similarity search mechanism to find the top three most relevant text chunks based on the query. It uses the embeddings stored in Pinecone to match the query with the most similar documents.
     - **Impact**: This ensures that the system retrieves the most relevant context for answering specific questions about medications.

   - **Generative AI Model (gemini-1.5-pro)**:
     - **Purpose**: Generate accurate, context-aware answers based on the retrieved information.
     - **Details**: The **gemini-1.5-pro** model is configured for the question-answering task. It is set with a low-temperature value to generate deterministic and concise responses. The model is guided by a system prompt designed to focus specifically on medical-related information, including composition, uses, side effects, and alternatives.
     - **System Prompt**: The system prompt ensures that the AI model focuses on the medicine-related aspects and avoids generating irrelevant or extraneous details.
     - **Impact**: The generative AI model enhances the chatbot’s ability to produce accurate and meaningful answers based on the context retrieved from Pinecone.

   - **Creating a Cohesive Question-Answering Chain**:
     - **Purpose**: Integrate the retriever and generative AI model into a unified workflow.
     - **Details**: The retriever fetches the most relevant chunks, which are then passed to the generative model. The **create_stuff_documents_chain** method integrates these two components, forming a complete question-answering chain.
     - **Impact**: This integration ensures that the system first retrieves the necessary context and then generates a precise, context-aware response.

   - **RAG Pipeline Creation**:
     - **Purpose**: Combine the retriever and the question-answering chain into an end-to-end RAG pipeline.
     - **Details**: The **create_retrieval_chain** function is used to combine the retriever and the generative model into a retrieval-augmented generative pipeline. This pipeline ensures that queries are processed efficiently and accurately by first retrieving the relevant context from Pinecone and then generating answers using the generative AI model.
     - **Impact**: The RAG pipeline ensures that the system can provide highly relevant and precise answers to user queries.

### 5. **User Interface and Presentation**
   - **Displaying Prescription and Medication Information**:
     - **Purpose**: Present the prescription details and medication information in a user-friendly format.
     - **Details**: The extracted prescription data, including medication details, is displayed using **Pandas** DataFrames, which organize the data in a clear and readable tabular format.
     - **Impact**: The user interface simplifies the review of prescription information, making it easily accessible for healthcare professionals or patients.

### Process Overview in Sequential Steps
Step 1: Launch the Streamlit Application and upload the prescription image
<img width="627" height="365" alt="image" src="https://github.com/user-attachments/assets/679ce6db-7ce2-420d-8081-338680136bd7" />

Step 2: Extracted Information from the Prescription
<img width="620" height="613" alt="image" src="https://github.com/user-attachments/assets/1c57831f-a8b8-44f0-99fe-1cf1e3ed8ccd" />

Step 3: Sample questions aked to chatbot according to the prescription
<img width="564" height="424" alt="image" src="https://github.com/user-attachments/assets/4dfc208a-9d80-4cff-9ab4-5e10fece64e2" />
<img width="604" height="655" alt="image" src="https://github.com/user-attachments/assets/9c21ca78-4516-44d3-a9a1-fb01a3d2ed25" />
<img width="583" height="415" alt="image" src="https://github.com/user-attachments/assets/a8f09715-a21a-47b0-ae51-94ca0746d59c" />






### Summary of PharmaBot Architecture:
- **Data Extraction**: Uses **GEMINI OCR** and the **GEMINI vision bot** to extract textual data from prescription images.
- **Data Structuring**: Organizes the extracted data into structured formats using classes like `MedicationItem` and `PrescriptionInformations`.
- **Embedding Generation**: Generates high-dimensional embeddings using **Google’s Generative AI embeddings model** and stores them in **Pinecone**, a vector database.
- **RAG Pipeline**: Integrates the **Pinecone retriever** with the **gemini-1.5-pro generative AI model** to answer medication-related queries in an efficient, context-aware manner.
- **User Interface**: Presents the extracted prescription information in a tabular format using **Pandas** DataFrames.

This architecture combines several powerful technologies, including OCR, embeddings, vector search, and generative AI, to automate prescription interpretation and provide users with precise, reliable answers to their medication-related queries.


Read more here: https://app.readytensor.ai/publications/pharmabot-a-prescription-analyst-ircy6qc747ih


