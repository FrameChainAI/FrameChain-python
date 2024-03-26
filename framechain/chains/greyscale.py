from framechain.chains.simple_chain import SimpleChain


@SimpleChain.from_func(standard_input_format="PIL", standard_output_format="PIL")
def greyscale(image: Image) -> Image:
    return image.convert("L")

