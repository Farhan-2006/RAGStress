"""Conventional RAG baseline: identical retrieval and generator, no audit layer."""
import time

class VanillaRAG:
    def __init__(self,retriever,generator):
        self.retriever,self.generator=retriever,generator

    def run(self,query,method='hybrid',k=5):
        started=time.perf_counter()
        hits=self.retriever.retrieve(query,k,method)
        retrieval=time.perf_counter()-started
        stage=time.perf_counter()
        answer,mode,warning=self.generator.generate(query,hits)
        generation=time.perf_counter()-stage
        return {'query':query,'method':method,'k':k,'retrieved':hits,'generated_answer':answer,
                'generation_mode':mode,'warning':warning,'elapsed_seconds':time.perf_counter()-started,
                'timings':{'retrieval':retrieval,'generation':generation}}
