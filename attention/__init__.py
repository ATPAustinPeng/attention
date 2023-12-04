from flask import Flask
from .attention import get_completion
import json
import torch

app = Flask(__name__)

prompt = """
Given the following sentence: While the man hunted, the deer that was brown and graceful ran into the woods.

Q: Did the man hunt the deer? and Did the deer run through the woods?
A: 
""".strip()

result, tokenized, attn_m = get_completion(prompt)
print(type(result), result)
print(type(tokenized), len(tokenized), tokenized)
print(type(attn_m), attn_m.shape)
sparse = attn_m.to_sparse()


# @app.route("/attention/<int:query_id>/<int:question_id>")
# def attention_view(query_id=0, question_id=0):
#     with open(f"attention/data/q{query_id}_{question_id}_decoded", "rb") as f:
#         result = f.read().decode('utf-8')
    
#     with open(f"attention/data/q{query_id}_{question_id}_tokenized", "rb") as f:
#         tokenized = f.read().decode('utf-8')

#     with open(f"attention/data/q{query_id}_{question_id}_attention.pth", "rb") as f:
#         attn_m = torch.load(f, map_location=torch.device('cpu'))

@app.route("/attention")
def attention_view():
    sparse = attn_m.to_sparse()

    indices, values = sparse.indices(), sparse.values()
    return json.dumps({
        'tokens': tokenized,
        'attn_indices': indices.T.numpy().tolist(),
        'attn_values': values.numpy().tolist(),
    })