import streamlit as st
import transformers
import torch

from huggingface_hub import login
login("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")

# Function to get the response back
def getLLMResponse(form_input, email_sender, email_recipient, email_style):
    try:
        # Load the model
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        # Template for building the PROMPT
        template = f"""
        Write an email with {email_style} style and includes topic: {form_input}.\n\nSender: {email_sender}\nRecipient: {email_recipient}\n\nEmail Text:
        """

        # Generate the response using LLM
        outputs = pipeline(
            template,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
         
        response = outputs[0]["generated_text"]
        return response

    except Exception as e:
        return f"An error occurred: {e}"

st.set_page_config(page_title="Generate Emails",
                   page_icon='📧',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Emails 📧")

form_input = st.text_area('Enter the email topic', height=275)

# Creating columns for the UI - To receive inputs from user
col1, col2, col3 = st.columns([10, 10, 5])
with col1:
    email_sender = st.text_input('Sender Name')
with col2:
    email_recipient = st.text_input('Recipient Name')
with col3:
    email_style = st.selectbox('Writing Style',
                               ('Formal', 'Appreciating', 'Not Satisfied', 'Neutral'),
                               index=0)

submit = st.button("Generate")

# When 'Generate' button is clicked, execute the below code
if submit:
    response = getLLMResponse(form_input, email_sender, email_recipient, email_style)
    st.write(response)
