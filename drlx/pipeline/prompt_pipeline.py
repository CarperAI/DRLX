from typing import Callable, Iterable, Any

from drlx.pipeline import  Pipeline

class PromptPipeline(Pipeline):
    """
    Base class for a pipeline that provides text prompts.
    """

    @classmethod
    def make_default_collate(self, prep : Callable):
        def collate(batch : Iterable[str]):
            return prep(batch)

        return collate
    
