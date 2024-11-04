import os
import json
import gzip

from tqdm import tqdm
from ranx import Qrels
from random import sample
from loguru import logger
from typing import List, Tuple
from collections import defaultdict
from torch.utils.data import Dataset
from urllib.request import urlretrieve
from transformers import BertTokenizer, PreTrainedTokenizer

from src.utils import show_progress
from src.schemas import ValidationSet


MAX_QUERY_SUBSET = 1000


class MSMARCO(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer = None, **kwargs):
        self.file_path = None
        self.dataset_dir = "msmarco"

        self.query_max_length = kwargs.get("query_max_length", 64)
        self.passage_max_length = kwargs.get("passage_max_length", 128)

        self.tokenizer = tokenizer or BertTokenizer.from_pretrained("bert-base-uncased")
        self.dataset_path = "https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/msmarco-triplets.jsonl.gz"

        if not os.path.exists(self.dataset_dir):
            logger.info("No dataset found. Downloading...")
            os.makedirs(self.dataset_dir, exist_ok=True)

            self.file_path = self._download_and_extract()
        else:
            logger.debug("Found existing dataset directory.")
            self.file_path = os.path.join(self.dataset_dir, "msmarco-triplets.jsonl")

        self.data = self._load_data()

    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx) -> dict:
        query, positive, negative = self.data[idx]

        qt = self.tokenizer(
            query, 
            max_length=self.query_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        pt = self.tokenizer(
            positive, 
            max_length=self.passage_max_length,
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        nt = self.tokenizer(
            negative, 
            max_length=self.passage_max_length,
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            "query": query,
            "positive": positive,
            "negative": negative,
            "query_ids": qt["input_ids"].squeeze().detach().clone(),
            "query_mask": qt["attention_mask"].squeeze().detach().clone(),
            "query_type": qt["token_type_ids"].squeeze().detach().clone(),
            "positive_ids": pt["input_ids"].squeeze().detach().clone(),
            "positive_mask": pt["attention_mask"].squeeze().detach().clone(),
            "positive_type": pt["token_type_ids"].squeeze().detach().clone(),
            "negative_ids": nt["input_ids"].squeeze().detach().clone(),
            "negative_mask": nt["attention_mask"].squeeze().detach().clone(),
            "negative_type": nt["token_type_ids"].squeeze().detach().clone()
        }


    def _download_and_extract(self) -> str:
        output_path = os.path.join(self.dataset_dir, "msmarco-triplets.jsonl.gz")
        urlretrieve(self.dataset_path, output_path, show_progress)

        logger.info("Download complete.")
        logger.info(f"Extracting dataset from {self.dataset_dir}/msmarco-triplets.jsonl.gz")
        
        with gzip.open(os.path.join(self.dataset_dir, "msmarco-triplets.jsonl.gz"), "rb") as f:
            with open(os.path.join(self.dataset_dir, "msmarco-triplets.jsonl"), "wb") as out:
                out.write(f.read())

        logger.info("Extraction complete.")

        os.remove(os.path.join(self.dataset_dir, "msmarco-triplets.jsonl.gz"))
        return os.path.join(self.dataset_dir, "msmarco-triplets.jsonl")
    

    def _load_data(self) -> List[List[str]]:
        data = []

        with open(self.file_path, "r") as f:
            for line in tqdm(f, desc="Loading data"):
                row = json.loads(line)

                query = row["query"]
                positive = row["pos"]
                negative = row["neg"]

                for positive_passage in positive:
                    for negative_passage in negative:
                        data.append([query, positive_passage, negative_passage])

        return data


def prepare_dev_dataset() -> Tuple[ValidationSet, ValidationSet]:
    passages = []
    id2passage = {}
    passage2id = {}

    with open("msmarco/collection.tsv", "r") as f:
        for line in tqdm(f, desc="Loading passages"):
            passage_id, passage = line.strip().split("\t")
            passages.append(passage)
            id2passage[int(passage_id)] = passage
            passage2id[passage] = int(passage_id)


    queries = []
    query2id = {}
    qrels = defaultdict(dict)

    with open("msmarco/qrels.dev.tsv", "r") as f:
        for line in tqdm(f, desc="Loading dev qrels"):
            query_id, _, passage_id, relevance = line.strip().split()
            query_id, passage_id = int(query_id), int(passage_id)

            qrels[str(query_id)][str(passage_id)] = int(relevance)

    with open("msmarco/queries.dev.tsv", "r") as f:
        for line in tqdm(f, desc="Loading dev queries"):
            query_id, query = line.strip().split("\t")
            query_id = int(query_id)

            queries.append(query)
            query2id[query] = query_id

    # Create subset
    query_subset = sample(queries, min(MAX_QUERY_SUBSET, len(queries)))
    query2id_subset = {query: query2id[query] for query in query_subset}
    qid_set = set(query2id_subset.values())

    qrels_subset = {str(qid): qrels[str(qid)] for qid in qid_set}
    passage_subset = set()

    for qid in qid_set:
        passage_subset.update(id2passage[int(pid)] for pid in qrels[str(qid)])

    random_passages = sample(passages, min(15000, len(passages)))
    passage_subset = list(set(passage_subset).union(random_passages))
    passage2id_subset = {passage: passage2id[passage] for passage in passage_subset}

    val_complete = ValidationSet(
        queries=queries,
        passages=passages,
        query2id=query2id,
        passage2id=passage2id,
        qrels=Qrels(qrels)
    )
    val_subset = ValidationSet(
        queries=query_subset,
        passages=passage_subset,
        query2id=query2id_subset,
        passage2id=passage2id_subset,
        qrels=Qrels(qrels_subset)
    )

    return val_complete, val_subset
    