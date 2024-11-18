import fire

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

def train(nranks:int):
    with Run().context(RunConfig(nranks=nranks, root="/future/u/herumbshandilya/home/hyperbolic-colbert/colbert", experiment="hyperbolic-late-interaction-training", name="lr3e-06_warmup20k_maxsteps500k_bert-large-uncased_3")):
        config = ColBERTConfig(
            reranker=False,
            bsize=16,
            lr=3e-06,
            warmup=20_000,
            doc_maxlen=180,
            query_maxlen=32,
            dim=128,
            attend_to_mask_tokens=True,
            nway=2,
            accumsteps=1,
            similarity='hyperbolic',
            use_ib_negatives=False,
            ignore_scores=True,
            shuffle_triples = False,
            checkpoint="colbert-ir/colbertv2.0",
            maxsteps=500001,
            projection_dim=16,
            hyperbolic_maxnorm=None,
            amp=False,
            warmup_bert=None,
        )

        trainer = Trainer(
            triples="/future/u/herumbshandilya/home/ColBERT-new/round3C.train.json",
            queries="/future/u/herumbshandilya/home/ColBERT-new/queries.train.tsv",
            collection="/future/u/herumbshandilya/home/ColBERT-new/msmarco.psg.cleaned.collection.tsv",
            config=config,
        )

        checkpoint_path = trainer.train(checkpoint="bert-large-uncased")

        print(f"Saved checkpoint to {checkpoint_path}...")

if __name__ == '__main__':
    fire.Fire(train)

"""
PYTHONPATH=.:$PYTHONPATH CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train_colbert.py 4 | tee hyperbolic_bs16_lr3e-06_warmup20k_maxsteps500k_bert-large-uncased_64.log
"""