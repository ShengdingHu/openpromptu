import copy
import json
import pickle
from typing import List, Optional, Union

class InputExample(object):
    """A raw input example consisting of segments of text,
    a label for classification task or a target sequence of generation task.
    Other desired information can be passed via meta.

    Args:
        guid (:obj:`str`, optional): A unique identifier of the example.
        text_a (:obj:`str`, optional): The placeholder for sequence of text.
        text_b (:obj:`str`, optional): A secend sequence of text, which is not always neccessary.
        label (:obj:`int`, optional): The label id of the example in classification task.
        tgt_text (:obj:`Union[str,List[str]]`, optional):  The target sequence of the example in a generation task..
        meta (:obj:`Dict`, optional): An optional dictionary to store arbitrary extra information for the example.
    """

    def __init__(self,
                 guid = None,
                 label = None,
                 tgt_text: Optional[Union[str,List[str]]] = None,
                 **kwargs
                ):

        self.guid = guid
        # self.text_a = text_a
        # self.text_b = text_b
        self.label = label
        self.meta = kwargs
        self.tgt_text = tgt_text

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        r"""Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def keys(self, keep_none=False):
        return [key for key in self.__dict__.keys() if getattr(self, key) is not None]

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)