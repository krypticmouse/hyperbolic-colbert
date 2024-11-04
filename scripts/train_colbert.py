import fire

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

def train(nranks:int):
    with Run().context(RunConfig(nranks=nranks, root="/future/u/herumbshandilya/home/ColBERT/experiments", experiment="late-interaction-training", name="msmarco-colv2-bert-large.test")):
        config = ColBERTConfig(
            reranker=False,
            bsize=16,
            lr=3e-06,
            warmup=20_000,
            doc_maxlen=180,
            query_maxlen=32,
            dim=128,
            attend_to_mask_tokens=True,
            nway=64,
            accumsteps=1,
            similarity='hyperbolic',
            use_ib_negatives=True,
            ignore_scores=True,
            model_type = "encoder-only",
            shuffle_triples = False,
            amp_dtype="float16",
            checkpoint="bert-large-uncased",
            maxsteps=500001,
            projection_dim=100,
            hyperbolic_maxnorm=None,
        )

        trainer = Trainer(
            triples="/future/u/herumbshandilya/home/ColBERT/examples.64.json",
            queries="/future/u/herumbshandilya/home/ColBERT/queries.train.tsv",
            collection="/future/u/herumbshandilya/home/ColBERT/msmarco.psg.cleaned.collection.tsv",
            config=config,
        )

        checkpoint_path = trainer.train(checkpoint="bert-large-uncased")

        print(f"Saved checkpoint to {checkpoint_path}...")

if __name__ == '__main__':
    fire.Fire(train)

'''
CUDA_VISIBLE_DEVICES=0,1,2,3 colbert2.5_training_late_interaction_colv1.py 4 | late_interaction_2way_msmarco_colv1.txt

CUDA_VISIBLE_DEVICES=6,7 python colbert2.5_training_late_interaction_colv1.py 2 | tee late_interaction_2way_msmarco_colv1_mosaic_bert_2048.txt

CUDA_VISIBLE_DEVICES=4,5,6,7 python colbert2.5_training_late_interaction_colv1.py 4 | tee late_interaction_64way_msmarco_colv2_bert_large_lowlr.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python colbert2.5_training_late_interaction_colv1.py 8 | tee late_interaction_64way_msmarco_colv2_nomic_bert_2048.txt
'''