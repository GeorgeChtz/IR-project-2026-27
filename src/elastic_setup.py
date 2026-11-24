import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from elasticsearch import Elasticsearch, helpers, NotFoundError
from src.preprocessing import parse_cf_documents
from src.inverted_index import InvertedIndex

# --- Ρυθμίσεις ---
ES_URL = "http://localhost:9200"
IDX_DOCS = "cf_docs"               # index με τα κείμενα
IDX_INVERTED = "cf_inverted"       # index με το ανεστραμμένο ευρετήριο
IDX_STATS = "cf_stats"             # index με N, avgdl, doc_lengths και doc_norms

es = Elasticsearch(ES_URL)


def recreate_index(name, mappings=None):
    """Σβήνει το index αν υπάρχει και το φτιάχνει από την αρχή."""
    if es.indices.exists(index=name):
        es.indices.delete(index=name)
    es.indices.create(index=name, mappings=mappings)


def store_docs(docs):
    """Γράφει τα κείμενα στον ES. Κάθε dict της λίστας docs = ένα document."""
    recreate_index(IDX_DOCS)
    actions = (
        {
            "_index": IDX_DOCS,
            "_id": d["doc_id"],   # ΔΙΟΡΘΩΣΗ: το κλειδί από τον parser είναι "doc_id"
            "_source": d,
        }
        for d in docs
    )
    helpers.bulk(es, actions)


def store_inverted(inv_index: InvertedIndex):
    """Γράφει το ανεστραμμένο ευρετήριο (ένα document ανά όρο)."""
    recreate_index(IDX_INVERTED, mappings={
        "properties": {
            "df": {"type": "integer"},
            "postings": {"type": "object", "enabled": False},
        }
    })
    actions = (
        {
            "_index": IDX_INVERTED,
            "_id": term,
            "_source": {
                "df": data["df"],
                # ΔΙΟΡΘΩΣΗ: το data["postings"] είναι dict {doc_id: tf}
                "postings": [{"doc_id": d, "tf": tf} for d, tf in data["postings"].items()],
            },
        }
        for term, data in inv_index.index.items()
    )
    helpers.bulk(es, actions)


def store_stats(inv_index: InvertedIndex):
    """Γράφει τα καθολικά στατιστικά που χρειάζονται BM25 και VSM."""
    recreate_index(IDX_STATS, mappings={
        "properties": {
            "N": {"type": "integer"},
            "avgdl": {"type": "float"},
            "doc_lengths": {"type": "object", "enabled": False},
            "doc_norms": {"type": "object", "enabled": False},
        }
    })
    es.index(index=IDX_STATS, id="stats", document={
        "N": inv_index.total_docs,
        "avgdl": inv_index.avg_doc_length,
        "doc_lengths": inv_index.doc_lengths,   # Απαραίτητο για BM25
        "doc_norms": inv_index.doc_norms,       # Απαραίτητο για VSM (Cosine Similarity)
    })


def build_all(docs, inv_index: InvertedIndex):
    """Εκτελεί όλη την εισαγωγή δεδομένων και ευρετηρίου στον Elasticsearch."""
    store_docs(docs)
    store_inverted(inv_index)
    store_stats(inv_index)
    es.indices.refresh(index="_all")


# ------------------------------------------------------------
# Συναρτήσεις ανάγνωσης: θα τις καλούν τα μοντέλα και το RAG
# ------------------------------------------------------------

def get_postings(term):
    """Δίνει (df, [(doc_id, tf), ...]) για έναν όρο."""
    try:
        src = es.get(index=IDX_INVERTED, id=term)["_source"]
    except NotFoundError:
        return 0, []
    return src["df"], [(p["doc_id"], p["tf"]) for p in src["postings"]]


def get_stats():
    """Δίνει λεξικό με N, avgdl, doc_lengths και doc_norms."""
    return es.get(index=IDX_STATS, id="stats")["_source"]


def get_doc(doc_id):
    """Δίνει το dict ενός κειμένου από το doc_id του."""
    return es.get(index=IDX_DOCS, id=str(doc_id))["_source"]


def get_all_docs():
    """Φέρνει όλα τα αποθηκευμένα κείμενα από τον ES (για τα μοντέλα βιβλιοθηκών & RAG)."""
    results = helpers.scan(es, index=IDX_DOCS, query={"query": {"match_all": {}}})
    docs = [hit["_source"] for hit in results]
    # Ταξινομούμε με βάση το αριθμητικό doc_id για σταθερή σειρά
    docs.sort(key=lambda d: int(d["doc_id"]))
    return docs


if __name__ == "__main__":
    print("1. Ανάγνωση εγγράφων από τα τοπικά αρχεία...")
    docs = parse_cf_documents()
   
    print("2. Κατασκευή Custom Inverted Index...")
    inv_index = InvertedIndex()
    inv_index.build(docs)
   
    print("3. Αποθήκευση συλλογής και ευρετηρίου στην Elasticsearch...")
    build_all(docs, inv_index)
   
    stats = get_stats()
    all_docs = get_all_docs()
    print(f"Ολοκληρώθηκε! Αποθηκεύτηκαν {len(all_docs)} έγγραφα (N={stats['N']}, avgdl={stats['avgdl']:.2f}).")