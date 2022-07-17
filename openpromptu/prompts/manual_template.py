
from openpromptu import logging
from openpromptu.data_utils import InputExample
from typing import *


logger = logging.get_logger(__name__)

class ManualTemplate:
    """
    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """

    registered_inputflag_names = ['shortenable_ids']

    def __init__(self,
                #  tokenizer: PreTrainedTokenizer,
                 text: Optional[List[str]] = None,
                #  placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        # super().__init__(,
        #                  placeholder_mapping=placeholder_mapping)

        self.mixed_token_start = "{"
        self.mixed_token_end = "}"

        self.text = self.parse_text(text)

        # @abstractmethod
    def wrap_one_example(self,
                         example: InputExample) -> List[Dict]:
        r'''Given an input example which contains input text, which can be referenced
        by self.template.placeholder_mapping 's value.
        This function process the example into a list of dict,
        Each dict functions as a group, which has the sample properties, such as
        whether it's shortenable, whether it's the masked position, whether it's soft token, etc.
        Since a text will be tokenized in the subsequent processing procedure,
        these attributes are broadcasted along the tokenized sentence.

        Args:
            example (:obj:`InputExample`): An :py:class:`~openprompt.data_utils.data_utils.InputExample` object, which should have attributes that are able to be filled in the template.

        Returns:
            :obj:`List[Dict]`: A list of dict of the same length as self.text. e.g. ``[{"loss_ids": 0, "text": "It was"}, {"loss_ids": 1, "text": "<mask>"}, ]``
        '''

        if self.text is None:
            raise ValueError("template text has not been initialized")
        if isinstance(example, InputExample):
            text = self.incorporate_text_example(example)

            not_empty_keys = example.keys()
            # for placeholder_token in self.placeholder_mapping:
            #     not_empty_keys.remove(self.placeholder_mapping[placeholder_token]) # placeholder has been processed, remove
            not_empty_keys.remove('meta') # meta has been processed

            keys, values= ['text'], [text]
            for inputflag_name in self.registered_inputflag_names:
                keys.append(inputflag_name)
                v = None
                if hasattr(self, inputflag_name) and getattr(self, inputflag_name) is not None:
                    v = getattr(self, inputflag_name)
                elif hasattr(self, "get_default_"+inputflag_name):
                    v = getattr(self, "get_default_"+inputflag_name)()
                    setattr(self, inputflag_name, v) # cache
                else:
                    raise ValueError("""
                    Template's inputflag '{}' is registered but not initialize.
                    Try using template.{} = [...] to initialize
                    or create an method get_default_{}(self) in your template.
                    """.format(inputflag_name, inputflag_name, inputflag_name))

                if len(v) != len(text):
                    raise ValueError("Template: len({})={} doesn't match len(text)={}."\
                        .format(inputflag_name, len(v), len(text)))
                values.append(v)
            wrapped_parts_to_tokenize = []
            for piece in list(zip(*values)):
                wrapped_parts_to_tokenize.append(dict(zip(keys, piece)))

            wrapped_parts_not_tokenize = {key: getattr(example, key) for key in not_empty_keys}
            return [wrapped_parts_to_tokenize, wrapped_parts_not_tokenize]
        else:
            raise TypeError("InputExample")

    def incorporate_text_example(self,
                                 example: InputExample
                                ):
        text = self.text.copy()
        for i, d in enumerate(text):
            if 'meta' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x:x)(example.meta[d['meta']])
            elif 'mask' in d:
                text[i] = '<mask>'
            elif 'text' in d:
                text[i] = d["add_prefix_space"] + d['text']
            else:
                raise ValueError(f'can not parse {d}')
        return text

    def _check_template_format(self, ):
        r"""check whether the template format is correct.
        TODO: add more
        """
        mask_num = 0
        for i, d in enumerate(self.text):
            if 'mask' in d:
                mask_num += 1

        if mask_num==0:
            raise RuntimeError(f"'mask' position not found in the template: {self.text}. Please Check!")




    def parse_text(self, text: str) -> List[Dict]:
        parsed = []
        i = 0
        while i < len(text):
            d = {"add_prefix_space": ' ' if (i > 0 and text[i-1] == ' ') else ''}
            while i < len(text) and text[i] == ' ':
                d["add_prefix_space"] = ' '
                i = i + 1
            if i == len(text): break

            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d["text"] = text[i:j].rstrip(' ')
                i = j

            else:
                j = i + 1
                mixed_token_cnt = 1 # { {} {} } nested support
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        mixed_token_cnt -= 1
                        if mixed_token_cnt == 0: break
                    elif text[j] == self.mixed_token_start:
                        mixed_token_cnt += 1
                    j = j + 1
                if j == len(text):
                    raise ValueError(f"mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}")
                dict_str = '{'+text[i+1:j]+'}'
                try:
                    val = eval(dict_str)
                    if isinstance(val, set):
                        val = {k: None for k in val}
                    d.update(val)
                except:
                    import traceback
                    print(traceback.format_exc())
                    print(f"syntax error in {dict_str}")
                    exit()
                i = j + 1

            parsed.append(d)

        return parsed

    # def get_default_loss_ids(self) -> List[int]:
    #     '''Get the loss indices for the template using mask.
    #     e.g. when self.text is ``'{"placeholder": "text_a"}. {"meta": "word"} is {"mask"}.'``,
    #     output is ``[0, 0, 0, 0, 1, 0]``.

    #     Returns:
    #         :obj:`List[int]`: A list of integers in the range [0, 1]:

    #         - 1 for a masked tokens.
    #         - 0 for a sequence tokens.
    #     '''
    #     return [1 if 'mask' in d else 0 for d in self.text]

    def get_default_shortenable_ids(self) -> List[int]:
        """Every template needs shortenable_ids, denoting which part of the template can be trucate to fit
        the language model's ``max_seq_length``. Default: the input text is shortenable, while the template text and other
        special tokens are not shortenable.

        e.g. when self.text is ``'{"placeholder": "text_a"} {"placeholder": "text_b", "shortenable": False} {"meta": "word"} is {"mask"}.'``,
        output is ``[1, 0, 0, 0, 0, 0, 0]``.

        Returns:
            :obj:`List[int]`: A list of integers in the range ``[0, 1]``:

            - 1 for the input tokens.
            - 0 for the template sequence tokens.
        """
        idx = []
        for d in self.text:
            if 'shortenable' in d:
                idx.append(1 if d['shortenable'] else 0)
            else:
                idx.append(1 if 'placeholder' in d else 0)
        return idx