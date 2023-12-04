# import torch

def test(query_id=0, question_id=0):
    with open(f"attention/data/q{query_id}_{question_id}_decoded", "rb") as f:
        result = f.read().decode('utf-8')
    
    with open(f"attention/data/q{query_id}_{question_id}_tokenized", "rb") as f:
        tokenized = f.read().decode('utf-8')


    # print(result)
    print(tokenized)

    # with open(f"attention/data/q{query_id}_{question_id}_attention.pth", "rb") as f:
    #     attn_m = torch.load(f, map_location=torch.device('cpu'))


if __name__ == '__main__':
    test()