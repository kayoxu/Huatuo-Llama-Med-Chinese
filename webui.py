import gradio as gr

app = gr.Blocks()

with app:
    gr.Markdown(value="""
        ### sovits4.0 webui 推理&训练 v2.0.0

        修改自原项目及bilibili@麦哲云

        仅供个人娱乐和非商业用途，禁止用于血腥、暴力、性相关、政治相关内容

        作者：bilibili@羽毛布団

        """)

app.launch(server_name="127.0.0.1", inbrowser=True, share=True)
