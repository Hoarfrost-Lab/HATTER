from transformers import RobertaTokenizer

class BioTokenizer(RobertaTokenizer):
  def __init__(self, ksize=1, stride=1, include_bos=False, include_eos=False, **kwargs):
    super().__init__(**kwargs)
    self.ksize = ksize
    self.stride = stride
    self.include_bos = include_bos
    self.include_eos = include_eos

  def tokenize(self, t, **kwargs):
    include_bos = self.include_bos if self.include_bos is not None else include_bos
    include_eos = self.include_eos if self.include_eos is not None else include_eos
    t = t.upper()
    if self.ksize == 1:
        toks = list(t)
#    else:
#        toks = [t[i:i + self.ksize] for i in range(0, len(t), self.stride) if len(t[i:i + self.ksize]) == self.ksize]
    if len(toks[-1]) < self.ksize:
        toks = toks[:-1]
    if include_bos:
        toks = ['S'] + toks
    if include_eos:
        toks = toks + ['/S']
    return toks

def load_tokenizer(model_name: str, model_max_length=128, padding=True, truncation=True) -> BioTokenizer:
    """Loads a pre-trained tokenizer for protein or DNA sequences.

    This function loads a tokenizer from the Transformers library based on the provided model name.
    It's likely that the `BioTokenizer` class is a custom class that modifies the tokenization
    behavior for your specific needs. The function retrieves the tokenizer from the `models` subdirectory
    and sets the vocabulary tokens (CLS, SEP, PAD, etc.) based on pre-defined values.

    **Note:** It's important to ensure that the `BioTokenizer` class appropriately handles the
             intended tokenization and maps the provided vocabulary tokens to the model's requirements.

    Args:
        model_name (str): The name of the pre-trained model to load the tokenizer for (e.g., "genebert").

    Returns:
        transformers.AutoTokenizer: The loaded tokenizer for the specified model.
    """

    tokenizer = BioTokenizer.from_pretrained(model_name, model_max_length=model_max_length, padding=padding, truncation=truncation) #genebert is of no use actually, since Biotokenizer class is overwritting tokenize function

    cls_token = "S"
    pad_token = "P"
    sep_token = "/S"
    unk_token = "N"
    mask_token = "M"
    G_token = "G"
    A_token = "A"
    C_token = "C"
    T_token = "T"

    token_ids = tokenizer.convert_tokens_to_ids([cls_token, pad_token, sep_token, unk_token, mask_token, G_token, A_token, C_token, T_token])

    tokenizer.cls_token_id = token_ids[0]
    tokenizer.pad_token_id = token_ids[1]
    tokenizer.sep_token_id = token_ids[2]
    tokenizer.unk_token_id = token_ids[3]
    tokenizer.mask_token_id = token_ids[4]
    tokenizer.bos_token_id = token_ids[0]
    tokenizer.eos_token_id = token_ids[1]

    return tokenizer

