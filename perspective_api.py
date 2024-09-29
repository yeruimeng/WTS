import json
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Iterable, Tuple
import itertools
import random

from googleapiclient import discovery
from googleapiclient.errors import HttpError
from tqdm.auto import tqdm

PERSPECTIVE_API_KEY = "KEY"
PERSPECTIVE_API_ATTRIBUTES = ["TOXICITY"]

def batchify(iterable, n=1):
    iterator = iter(iterable)
    return iter(lambda: list(itertools.islice(iterator, n)), [])

class PerspectiveAPI:
    def __init__(self, api_key: str = PERSPECTIVE_API_KEY, rate_limit: int = 25):
        self.service = self._make_service(api_key)
        self.last_request_time = -1
        self.rate_limit = rate_limit
        self.next_uid = 0

    def request(self, texts: Union[str, List[str]], max_retries: int = 5) -> List[Tuple[Optional[Dict[str, Any]], Optional[HttpError]]]:
        if isinstance(texts, str):
            texts = [texts]

        assert len(texts) <= self.rate_limit
        
        for attempt in range(max_retries):
            try:
                time_since_last_request = time.time() - self.last_request_time
                if time_since_last_request < 1:
                    time.sleep(1 - time_since_last_request)
                self.last_request_time = time.time()

                responses = {str(uid): None for uid in range(self.next_uid, self.next_uid + len(texts))}
                self.next_uid += len(texts)

                def response_callback(request_id, response, exception):
                    nonlocal responses
                    responses[request_id] = (response, exception)

                batch_request = self.service.new_batch_http_request()
                for uid, text in zip(responses.keys(), texts):
                    batch_request.add(self._make_request(text, self.service), callback=response_callback, request_id=uid)
                batch_request.execute()

                return list(responses.values())

            except HttpError as e:
                if e.resp.status == 429:  
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Quota exceeded. Waiting for {wait_time:.2f} seconds before retrying.")
                    time.sleep(wait_time)
                else:
                    raise

        print("Max retries reached. Some requests may have failed.")
        return [(None, "Max retries reached")] * len(texts)

    def request_bulk(self, corpus: Iterable[Tuple[str, str]], output_file: Union[str, Path], pbar: tqdm = None):
        output_file = Path(output_file)
        if not pbar:
            pbar = tqdm(dynamic_ncols=True)
        pbar.set_description(f'Perspective API')

        num_failures = 0
        with output_file.open('a') as f:
            for batch in batchify(corpus, self.rate_limit):
                request_ids, texts = zip(*batch)

                for j, (response, exception) in enumerate(self.request(texts)):
                    response_dict = {
                        'request_id': request_ids[j],
                        'response': response,
                        'error': str(exception) if exception else None
                    }

                    json.dump(response_dict, f)
                    f.write('\n')

                    if exception:
                        num_failures += 1

                pbar.update(len(batch))
                pbar.set_postfix(failures=num_failures, rate_limit=self.rate_limit)

    @staticmethod
    def _make_service(api_key: str):
        return discovery.build('commentanalyzer', 'v1alpha1', developerKey=api_key, discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1")

    @staticmethod
    def _make_request(text: str, service):
        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {attr: {} for attr in PERSPECTIVE_API_ATTRIBUTES},
        }
        return service.comments().analyze(body=analyze_request)

    @staticmethod
    def _extract_toxicity(response: dict) -> Optional[dict]:
        try:
            return {"toxicity": response['attributeScores']['TOXICITY']['summaryScore']['value']}
        except KeyError:
            return None

class PerspectiveWorker:
    SENTINEL = 'STOP'

    def __init__(self, out_file: Path, total: int, rate_limit: int):
        self.enabled = rate_limit > 0
        if not self.enabled:
            print("Disabling Perspective API (rps is 0)")
            return

        self.requests_handled = set()
        for response in self.load_cache(out_file):
            self.requests_handled.add(response['request_id'])
        total -= len(self.requests_handled)

        self.task_queue = mp.Queue()
        self.process = mp.Process(target=self.perspective_worker,
                                  args=(self.task_queue, out_file, total, rate_limit))
        self.process.start()

    def __call__(self, request_id: str, text: str):
        if not self.enabled:
            return

        if request_id not in self.requests_handled:
            self.task_queue.put((request_id, text))

    def stop(self):
        if not self.enabled:
            return

        print("Waiting for Perspective to finish...")
        self.task_queue.put(self.SENTINEL)
        self.process.join()

    @classmethod
    def perspective_worker(cls, queue: mp.Queue, responses_file: Path, total: int, rate_limit: int):
        queue_iter = iter(queue.get, cls.SENTINEL)
        api = PerspectiveAPI(rate_limit=rate_limit)
        pbar = tqdm(total=total, dynamic_ncols=True, position=1)
        api.request_bulk(queue_iter, output_file=responses_file, pbar=pbar)

    @staticmethod
    def load_cache(file_path: Path):
        if file_path.exists():
            with file_path.open() as f:
                for line in f:
                    yield json.loads(line)

def process_jsonl(input_file: str, output_file: str, cache_file: str, rate_limit: int):
    worker = PerspectiveWorker(Path(cache_file), total=sum(1 for _ in open(input_file)) * 2, rate_limit=rate_limit)

    with open(input_file, 'r') as infile:
        for i, line in enumerate(tqdm(infile)):
            data = json.loads(line)
            prompt_text = data['prompt']['text']
            continuation_text = data['continuation']['text']

            worker(f"prompt_{i}", prompt_text)
            worker(f"continuation_{i}", continuation_text)

    worker.stop()

    # process results
    cache = {}
    for response in PerspectiveWorker.load_cache(Path(cache_file)):
        if response['error'] is None and response['response']:
            cache[response['request_id']] = PerspectiveAPI._extract_toxicity(response['response'])

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for i, line in enumerate(infile):
            data = json.loads(line)
            result = {
                "prompt": {
                    "text": data['prompt']['text'],
                    "toxicity": cache.get(f"prompt_{i}")
                },
                "continuation": {
                    "text": data['continuation']['text'],
                    "toxicity": cache.get(f"continuation_{i}")
                }
            }
            json.dump(result, outfile)
            outfile.write('\n')

def main():
    base_input_path = "./input"
    base_output_path = "./output"
    rate_limit = 1  

    for i in range(0, 25):  
        input_file = "./input"
        output_file = "./output"
        cache_file = "./cache"

        print(f"Processing file: ft_weak{i:02d}")
        process_jsonl(input_file, output_file, cache_file, rate_limit)
        print(f"Completed processing ft_weak{i:02d}")

        
        time.sleep(60)  

if __name__ == "__main__":
    main()