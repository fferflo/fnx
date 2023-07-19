import fnx, re, eval

def test_gpt():
    for name, builder in vars(fnx.pretrained.eleutherai).items():
        if re.match("gptneo_.*", name):
            perplexity = eval.text_generation(builder)
            assert perplexity < 100.0
