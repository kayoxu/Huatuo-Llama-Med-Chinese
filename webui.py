import gradio as gr

import infer


def load_model(input):
    infer.main(load_8bit=True, base_model='decapoda-research/llama-7b-hf', lora_weights='./lora-llama-med',
               use_lora=True, prompt_template='med_template')

def submint_disease(input):
    return "Hello " + input + "!"


app = gr.Blocks()

diagnosis = gr.TextArea()

with app:
    gr.Button(value="加载模型", fn=load_model)

    gr.Interface(fn=submint_disease, inputs="text_area", outputs=diagnosis)

app.launch(server_name="127.0.0.1", inbrowser=True, share=False)
