import fnx, re, eval

def test_gpt():
    for name, builder in vars(fnx.pretrained.openai).items():
        if re.match("gpt2_.*", name):
            perplexity = eval.text_generation(builder)
            assert perplexity < 100.0
