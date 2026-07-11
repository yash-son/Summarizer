import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class MedicalSummarizer:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        
    def get_summary_chain(self):
        """Creates a chain for simplifying medical reports."""
        template = """
        You are a helpful medical assistant. Your task is to simplify complex radiology reports into patient-friendly language.
        
        Guidelines:
        1. Avoid medical jargon. If you must use a complex term, explain it simply.
        2. Focus on the 'Impression' and 'Findings'.
        3. Use a reassuring and clear tone.
        4. Break down the summary into: 'What was seen', 'What it means', and 'Next steps (if mentioned)'.
        
        Context (Relevant Report Chunks):
        {context}
        
        Original Report:
        {report_text}
        
        Patient-Friendly Summary:
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {"context": RunnablePassthrough(), "report_text": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    def summarize(self, report_text, context_docs):
        """Generates a summary based on the report and retrieved context."""
        context = "\n\n".join([doc.page_content for doc in context_docs])
        chain = self.get_summary_chain()
        # Note: In a real RAG, context might come from external medical knowledge.
        # Here we use retrieved report chunks to ensure the LLM stays grounded.
        return chain.invoke({"context": context, "report_text": report_text})

    def ask_question(self, question, report_text, context_docs):
        """Answers a specific question about the report."""
        template = """
        You are a medical assistant helping a patient understand their radiology report.
        
        Report:
        {report_text}
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the patient's question clearly and simply. If the information is not in the report, say you don't know.
        """
        prompt = ChatPromptTemplate.from_template(template)
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "report_text": report_text, "context": context})
