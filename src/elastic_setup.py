# ============================================================
# elastic_setup.py
# Αποθηκεύει κείμενα + ευρετήριο στον Elasticsearch
# και δίνει συναρτήσεις για να τα διαβάζουμε πίσω.
# ============================================================

# Elasticsearch: ο client που μιλά με τον server
# helpers: έχει το bulk (εισαγωγή πολλών documents μαζί)
# NotFoundError: το σφάλμα όταν ζητάμε κάτι που δεν υπάρχει
from elasticsearch import Elasticsearch, helpers, NotFoundError

# --- Ρυθμίσεις: τις αλλάζουμε μόνο εδώ ---
ES_URL = "http://localhost:9200"   # πού τρέχει ο ES (το container σου)
IDX_DOCS = "cf_docs"               # index με τα κείμενα
IDX_INVERTED = "cf_inverted"       # index με το ανεστραμμένο ευρετήριο
IDX_STATS = "cf_stats"             # index με τα N και avgdl

# Ο client. Εδώ ΔΕΝ γίνεται ακόμα επικοινωνία, μόνο ετοιμάζεται.
es = Elasticsearch(ES_URL)


def recreate_index(name, mappings=None):
    """Σβήνει το index αν υπάρχει και το φτιάχνει από την αρχή.
    Έτσι, αν τρέξεις το script 2 φορές, δεν μένουν παλιά δεδομένα."""
    if es.indices.exists(index=name):      # υπάρχει ήδη;
        es.indices.delete(index=name)      # ναι -> σβήστο
    # mappings=None -> ο ES μαντεύει μόνος του τους τύπους των πεδίων
    es.indices.create(index=name, mappings=mappings)


def store_docs(docs):
    """Γράφει τα κείμενα. Κάθε dict της λίστας docs = ένα document."""
    recreate_index(IDX_DOCS)
    # Generator: για κάθε κείμενο d φτιάχνει μια "εντολή" για το bulk
    actions = (
        {
            "_index": IDX_DOCS,   # σε ποιο index θα γραφτεί
            "_id": d["rn"],       # το id του document (το RN, το ίδιο με το cfquery)
            "_source": d,         # το περιεχόμενο: ολόκληρο το dict του κειμένου
        }
        for d in docs
    )
    helpers.bulk(es, actions)     # τα στέλνει όλα μαζεμένα, πολύ πιο γρήγορα από ένα-ένα


def store_inverted(inverted):
    """Γράφει το ανεστραμμένο ευρετήριο: ένα document ανά όρο."""
    # mapping: λέμε στον ES πώς να αποθηκεύσει τα πεδία
    recreate_index(IDX_INVERTED, mappings={
        "properties": {
            "df": {"type": "integer"},            # σε πόσα κείμενα εμφανίζεται ο όρος
            # enabled=False: αποθήκευσέ το αλλά ΜΗΝ το ευρετηριάσεις
            # (το ευρετήριο είναι δικό μας, όχι του ES)
            "postings": {"type": "object", "enabled": False},
        }
    })
    actions = (
        {
            "_index": IDX_INVERTED,
            "_id": term,                          # ο ίδιος ο όρος είναι το id -> lookup με get
            "_source": {
                "df": len(postings),              # df = πόσα ζεύγη (doc, tf) έχει ο όρος
                # μετατρέπουμε τα ζεύγη (doc_id, tf) σε dicts, ώστε να είναι JSON
                "postings": [{"doc_id": d, "tf": tf} for d, tf in postings],
            },
        }
        for term, postings in inverted.items()    # για κάθε όρο του ευρετηρίου
    )
    helpers.bulk(es, actions)


def store_stats(docs):
    """Γράφει τα καθολικά στατιστικά που χρειάζεται το BM25."""
    N = len(docs)                                 # συνολικός αριθμός κειμένων
    avgdl = sum(d["length"] for d in docs) / N    # μέσο μήκος κειμένου
    recreate_index(IDX_STATS)
    # ένα μόνο document, με σταθερό id "stats"
    es.index(index=IDX_STATS, id="stats", document={"N": N, "avgdl": avgdl})


def build_all(docs, inverted):
    """Κάνει τα τρία παραπάνω με τη σειρά."""
    store_docs(docs)
    store_inverted(inverted)
    store_stats(docs)
    # refresh: κάνει τα νέα δεδομένα ορατά αμέσως
    # (αλλιώς ο ES μπορεί να αργήσει ~1 δευτερόλεπτο)
    es.indices.refresh(index="_all")


# ------------------------------------------------------------
# Συναρτήσεις ανάγνωσης: θα τις καλούν τα μοντέλα και το RAG
# ------------------------------------------------------------

def get_postings(term):
    """Δίνει (df, [(doc_id, tf), ...]) για έναν όρο.
    Αν ο όρος δεν υπάρχει στο ευρετήριο, δίνει (0, [])."""
    try:
        # get με _id = όρος: άμεσο lookup, σαν dict
        src = es.get(index=IDX_INVERTED, id=term)["_source"]
    except NotFoundError:
        return 0, []                              # άγνωστος όρος: όχι σφάλμα, απλά κενό
    # ξαναγυρνάμε από dicts σε ζεύγη (doc_id, tf)
    return src["df"], [(p["doc_id"], p["tf"]) for p in src["postings"]]


def get_stats():
    """Δίνει (N, avgdl)."""
    src = es.get(index=IDX_STATS, id="stats")["_source"]
    return src["N"], src["avgdl"]


def get_doc(rn):
    """Δίνει το dict ενός κειμένου από το RN του."""
    return es.get(index=IDX_DOCS, id=rn)["_source"]

