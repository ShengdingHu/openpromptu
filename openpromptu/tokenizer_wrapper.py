from collections import defaultdict
from typing import Callable, List, Mapping, Dict
import itertools
from .data_utils import InputExample
from math import ceil

def round_list(l: List[float], max_sum:int):
    r"""round a list of float e.g. [0.2,1.5, 4.5]
    to [1,2,4] # ceil and restrict the sum to `max_sum`
    used into balanced truncate.
    """
    s = 0
    for idx, i in enumerate(l):
        i = ceil(i)
        if s <= max_sum:
            s += i
            if s <= max_sum:
                l[idx] = i
            else:
                l[idx] = i - (s - max_sum)
        else:
            l[idx] = int(0)
    assert sum(l) == max_sum

class TokenizerWrapper:
    def __init__(self, max_seq_length, tokenizer, truncate_method, mask_token_func=lambda i: "<mask>"):
        self.max_seq_length=max_seq_length
        self.tokenizer=tokenizer
        self.num_special_tokens_to_add = len(tokenizer("")['input_ids'])
        # from IPython import embed; embed(header="Truega")
        self.truncate_method=truncate_method
        self.total_passed_sentences = 0
        self.num_truncated_sentences = 0
        self.mask_token_func = mask_token_func

        if truncate_method=='tail':
            self.truncate_fct = self.truncate_from_tail
        elif truncate_method=='head':
            self.truncate_fct = self.truncate_from_head
        elif truncate_method == 'balanced':
            self.truncate_fct = self.balanced_truncate
        else:
            raise NotImplementedError


    def merge_wrapped_example(self, wrapped_example, tgt_texts=None):
        ''' # TODO doens't consider the situation that input has two parts
        '''


        encoder_inputs = defaultdict(list)

        mask_count = 0
        for piece in wrapped_example:
            if piece['text'] == "<mask>":
                if tgt_texts is not None:
                    mask_text = tgt_texts[mask_count]
                else:
                    mask_text = self.mask_token_func(self.tokenizer, mask_count)
                encode_text = self.tokenizer.encode(mask_text, add_special_tokens=False, return_special_tokens_mask=True )
                mask_count += 1
            else:
                encode_text = self.tokenizer.encode(piece['text'], add_special_tokens=False, return_special_tokens_mask=True )
            encoder_inputs['input_ids'].append(encode_text)
            encoder_inputs['shortenable_ids'].append([piece['shortenable_ids']] * len(encode_text))


        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        decoded_inputs = self.tokenizer.decode(encoder_inputs['input_ids'], clean_up_tokenization_spaces=False)

        return decoded_inputs


    @staticmethod
    def balanced_truncate(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        '''truncate the inputs with balance, number of cut tokens is proportional to the part's length.
        '''
        shortenable_lens = [len(parts) if parts[0]==1 else 0
                                  for parts in input_dict['shortenable_ids']]
        total_shortenable_len = sum(shortenable_lens)
        num_tokens_to_truncate_each_part = [part_len/total_shortenable_len*num_tokens_to_truncate
                                                for part_len in shortenable_lens]
        round_list(num_tokens_to_truncate_each_part, num_tokens_to_truncate)

        truncated_example = defaultdict(list)
        for key in input_dict:
            parts = input_dict[key]
            for num_tokens_to_truncate_part, part in zip(num_tokens_to_truncate_each_part, parts):
                truncated_example[key].append(part[:len(part)-num_tokens_to_truncate_part])
        return truncated_example

    @staticmethod
    def truncate_from_tail(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        r"""truncate the inputs from the rear
        """
        truncated_example = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']

        for key in input_dict:
            parts = input_dict[key]
            to_trunc = num_tokens_to_truncate
            for i, part in enumerate(parts[::-1]):
                if len(part) == 0: # to prevent some part are empty after tokenization
                    continue
                if shortenable_ids[-1-i][0]==0: # ==0 means the part is not shortenable
                    continue
                parts[-1-i] = part[:-to_trunc] if to_trunc<len(part) else []
                to_trunc -= len(part)
                if to_trunc <= 0:
                    break
            truncated_example[key] = parts
        return truncated_example

    @staticmethod
    def truncate_from_head(input_dict: Dict,
                 num_tokens_to_truncate: int=0) -> Dict:
        r"""truncate the inputs from the head
        """
        truncated_example = defaultdict(list)
        shortenable_ids = input_dict['shortenable_ids']
        for key in input_dict:
            parts = input_dict[key]
            to_trunc = num_tokens_to_truncate
            for i, part in enumerate(parts):
                if shortenable_ids[i][0]==0: # ==0 means the part is not shortenable
                    continue
                parts[i] = part[:-to_trunc] if to_trunc<len(part) else []
                to_trunc -= len(part)
                if to_trunc <= 0:
                    break
            truncated_example[key] = parts
        return truncated_example

    @staticmethod
    def concate_parts(input_dict: Dict) -> Dict:
        for key in input_dict:
            input_dict[key] = list(itertools.chain(*input_dict[key]))
        return input_dict


    def truncate(self, encoder_inputs):
        total_tokens = sum([len(part) for part in encoder_inputs['input_ids']])
        num_specials = self.num_special_tokens_to_add
        # print("num_specials", num_specials)
        num_tokens_to_truncate = total_tokens - self.max_seq_length + num_specials
        self.total_passed_sentences+=1
        if num_tokens_to_truncate>0:
            self.num_truncated_sentences += 1
            if num_tokens_to_truncate > sum([len(x) for x in encoder_inputs['shortenable_ids']]):
                raise RuntimeError("num_tokens_to_truncate larger than number of shortenable tokens.")
            encoder_inputs = self.truncate_fct(input_dict=encoder_inputs,
                          num_tokens_to_truncate=num_tokens_to_truncate)
        return encoder_inputs

    def tokenizer_preprocessor(self, example):
        # source, target = example
        # from IPython import embed; embed(header="Trehre2")
        label = example['label']
        guid = example['idx']
        meta = dict(example)
        meta.pop("label")
        meta.pop("idx")



        # from IPython import embed; embed(header="Trehre2")

        e = InputExample(**{"meta": meta, 'label': label, 'guid': guid})

        if self.predict_with_generate:
            e = self.verbalizer.wrap_one_example(e)
        example_wrapped = self.template.wrap_one_example(e)
        encoded_sentence = self.tokenizer_wrapper.merge_wrapped_example(example_wrapped)
        print(encoded_sentence)
        if self.predict_with_generate:
            # return {"source": encoded_sentence, 'target': ', 'extra_fields':[]}
            return {"source": encoded_sentence, "label": label, 'target': '', 'extra_fields':{'dataset_name':self.name}}
        else:
            return {"source": encoded_sentence, "label": label, 'target': e.target_text, 'extra_fields':{'dataset_name':self.name}}



