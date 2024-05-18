"""Script with UI elements to interact with the PDF files"""
import re

import click
import gradio as gr

from tqdm import tqdm

from llmtools import openaiapi as oaiapi, pdf_utils
from db.pdfdb import PDFDB


@click.command()
@click.option(
    '--data_dir', '-f', help='Path to a directory containing PDFs',
    default=f'..\pdfs')
def main(data_dir):
    """ main """

    api_key = oaiapi.load_api_key('openai_api_key.txt')

    # Create database
    pdfdb = PDFDB('pdf_db', api_key)

    # Add PDFs from the directory to the database
    filenames = sorted(pdf_utils.find_pdfs_in_directory(data_dir))
    for filename in tqdm(filenames):
        pdfdb.add_pdf(filename)

    with open('json_prompt.txt') as f:
        base_prompt = f.read()

    def ask_question(question: str) -> str:
        """
        Function to get the pages with the closest matching context to answer
        the given question
        """

        # Get the top n closest pages for the question
        result = pdfdb.get_context_from_query(question, n_results=4)

        answers = []
        no_answer = None
        for res in result:
            prompt = base_prompt
            prompt += 'Question: ' + question
            prompt += res
            # print("Result:", res)
            page_number = re.search(r'\bPage Number: (\d+)\b', res).group(1)

            out = oaiapi.chat_completion(
                prompt=prompt,
                model='gpt-3.5-turbo-0125')

            # print(out)
            if 'There is no' in out:
                print(f'Not enough information on page {page_number}.')
                no_answer = out
                continue

            answers.append(out)

        if not answers:
            answers.append(no_answer)

        return '\n--\n'.join(answers)

    question = """What is the name of the company?"""
    question = ' '.join(question.split())

    with gr.Blocks() as demo:
        gr.Markdown('# PDF QA')
        with gr.Row():
            text_input = gr.Textbox(value=question)
        with gr.Row():
            question_button = gr.Button('Ask a question')

        gr.Markdown("# Results")
        with gr.Row():
            text_output = gr.Textbox()

        question_button.click(
            ask_question, inputs=text_input, outputs=text_output)

    demo.launch()


if __name__ == '__main__':
    main()

