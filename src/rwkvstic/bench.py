import time
import torch
from rwkvstic.load import RWKV
def bechmark():
    # choose a file to load with the file picker dialog
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename( 
        initialdir = "./",
        title = "Select a File",
        filetypes = (
            ("pth files","*.pth"),
            ("all files","*.*")
        )
    )
    print(file_path)


    model = RWKV( 
        path=file_path,
    )


    context = '''
    The following is a conversation between a highly knowledgeable and intelligent AI assistant, called RWKV, and a human user, called User. In the following interactions, User and RWKV will converse in natural language, and RWKV will do its best to answer User’s questions. RWKV was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. The conversation begins.

    User: OK RWKV, I’m going to start by quizzing you with a few warm-up questions. Who is currently the president of the USA?

    RWKV: It’s Joe Biden; he was sworn in earlier this year.

    User: What year was the French Revolution?

    RWKV: It started in 1789, but it lasted 10 years until 1799.

    User: Can you guess who I might want to marry?

    RWKV: Only if you tell me more about yourself - what are your interests?

    User: Aha, I’m going to refrain from that for now. Now for a science question. What can you tell me about the Large Hadron Collider (LHC)?

    RWKV: It’s a large and very expensive piece of science equipment. If I understand correctly, it’s a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.
    '''

    time_slot = {}
    time_ref = time.time_ns()


    model.loadContext(context)
    for x in range(100):
        def record_time(name):
            if name not in time_slot:
                time_slot[name] = 1e20
            tt = (time.time_ns() - time_ref) / 1e9
            if tt < time_slot[name]:
                time_slot[name] = tt

        time_ref = time.time_ns()

        print(model.forward(number=100)["output"])

        record_time('Seconds/100 tokens')

        print(time_slot)
        time_slot = {}
