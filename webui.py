import os
import sys
import json

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"

os.environ['modelInit'] = ''


def load_model():
    load_8bit = True
    base_model = 'decapoda-research/llama-7b-hf'
    lora_weights = './lora-llama-med'
    use_lora = True
    prompt_template = 'med_template'
    global prompter, tokenizer, model

    print('模型加载中')

    try:
        if '' == os.environ['modelInit']:
            prompter = Prompter(prompt_template)
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            if use_lora:
                print(f"using lora {lora_weights}")
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    torch_dtype=torch.float16,
                )
            # unwind broken decapoda-research config
            model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
            if not load_8bit:
                model.half()  # seems to fix bugs for some users.

            model.eval()

            if torch.__version__ >= "2" and sys.platform != "win32":
                model = torch.compile(model)
        print('模型加载成功')
        os.environ['modelInit'] = 'ok'
        return '模型加载成功'
    except:
        print('模型加载失败')
        os.environ['modelInit'] = ''
        return '模型加载失败'


def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompter.get_response(output)


def submint_disease(input):
    return evaluate(input)


app = gr.Blocks()

diagnosis = gr.TextArea()

with app:
    btn = gr.Button(value="加载模型")
    init_status = gr.Label(value='还没有加载模型' if '' == os.environ['modelInit'] else '模型加载成功')
    btn.click(load_model, outputs=init_status)

    gr.Interface(fn=submint_disease, inputs="text_area", outputs=diagnosis)

app.launch(server_name="127.0.0.1", inbrowser=True, share=True)
