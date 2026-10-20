"""只读 Lucene 子进程。使用已有检索环境，不创建或修改索引。"""
import json
import sys


def main():
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(sys.argv[1])
    searcher.set_bm25(1.2, 0.75)
    print("STRUCKFLOW:" + json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            hits = searcher.search(request["query"], request["limit"])
            response = {"hits": [{"id": h.docid, "bm25": float(h.score)} for h in hits]}
        except Exception as exc:
            response = {"error": type(exc).__name__}
        print("STRUCKFLOW:" + json.dumps(response), flush=True)
    searcher.close()


if __name__ == "__main__":
    main()
