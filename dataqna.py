import streamlit as st
import transformers
from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

# Load pre-trained Tapex model and tokenizer
tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wikisql")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wikisql")

# Function to process the uploaded CSV file
def process_data(file):
    data = pd.read_csv(file)
    return data

# Function to answer user questions about the data using Tapex model
def answer_questions(data, question):
    encoding = tokenizer(table=data, query=question, return_tensors="pt")
    outputs = model.generate(**encoding)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer

# Function to display the uploaded data
def display_data(data):
    st.write("### Uploaded Data:")
    st.dataframe(data)

def main():
    st.title("Tapex-based CSV Data Explorer")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = process_data(uploaded_file)
        display_data(data)

        question = st.text_input("Ask a question about the data:")
        if st.button("Get Answer"):
            try:
                answer = answer_questions(data, question)
                st.write("#### Answer:")
                st.write(answer)
            except Exception as e:
                st.write(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
