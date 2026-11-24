import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
from collections import Counter, defaultdict
from src.preprocessing import preprocess_text, parse_cf_documents


class InvertedIndex:
    """
    Custom υλοποίηση Ανεστραμμένου Ευρετηρίου (Inverted Index)
    που αποθηκεύει όλες τις απαραίτητες στατιστικές πληροφορίες
    για τα μοντέλα Okapi BM25 και Vector Space Model (TF-IDF).
    """

    def __init__(self):
        # Λεξικό: term -> {"df": int, "postings": {doc_id: tf}}
        self.index = {}
        # Λεξικό: doc_id -> μήκος εγγράφου (πλήθος επεξεργασμένων tokens)
        self.doc_lengths = {}
        # Λεξικό: doc_id -> ευκλείδεια νόρμα του TF-IDF διανύσματος (για Cosine Similarity)
        self.doc_norms = {}
        # Συνολικό πλήθος εγγράφων (N)
        self.total_docs = 0
        # Μέσο μήκος εγγράφου (avgdl) για το BM25
        self.avg_doc_length = 0.0

    def build(self, documents: list[dict]):
        """
        Κατασκευάζει το ανεστραμμένο ευρετήριο από μια λίστα εγγράφων.
        Κάθε έγγραφο έχει τη μορφή: {'doc_id': str, 'title': str, 'text': str, ...}
        """
        self.total_docs = len(documents)
        temp_postings = defaultdict(dict)
        total_length = 0

        # Βήμα 1: Πέρασμα από όλα τα έγγραφα για συλλογή tf, df και μήκους εγγράφων
        for doc in documents:
            doc_id = doc["doc_id"]
            tokens = preprocess_text(doc["text"])
            
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            total_length += doc_len

            # Μετράμε πόσες φορές εμφανίζεται κάθε όρος στο τρέχον έγγραφο (tf)
            term_counts = Counter(tokens)
            for term, tf in term_counts.items():
                temp_postings[term][doc_id] = tf

        # Υπολογισμός μέσου μήκους εγγράφου (avgdl)
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0.0

        # Βήμα 2: Δόμηση του τελικού ευρετηρίου με df και postings
        for term, postings in temp_postings.items():
            self.index[term] = {
                "df": len(postings),
                "postings": postings
            }

        # Βήμα 3: Προϋπολογισμός των TF-IDF διανυσματικών νορμών (||d||) για το VSM
        self._compute_doc_norms()

    def _compute_doc_norms(self):
        """
        Υπολογίζει την ευκλείδεια νόρμα κάθε εγγράφου στον διανυσματικό χώρο TF-IDF:
        ||d|| = sqrt( sum( (w_{t,d})^2 ) ), όπου w_{t,d} = (1 + log(tf)) * log(N / df)
        """
        squared_sums = defaultdict(float)

        for term, data in self.index.items():
            df = data["df"]
            idf = math.log(self.total_docs / df) if df > 0 else 0.0 #βαρος καθε λεξης

            for doc_id, tf in data["postings"].items():
                # Λογαριθμικό TF (sublinear TF scaling): 1 + log(tf)
                tf_weight = 1.0 + math.log(tf) if tf > 0 else 0.0 
                """Εφαρμόζει λογαριθμική εξομάλυνση (sublinear TF scaling). 
                Αν μια λέξη εμφανίζεται 100 φορές σε ένα έγγραφο, δεν είναι 100 φορές πιο σημαντική από το 
                αν εμφανιζόταν 1 φορά — ο λογάριθμος βάζει «φρένο» στην υπερβολική επανάληψη."""
                tfidf = tf_weight * idf
                squared_sums[doc_id] += tfidf ** 2

        for doc_id in self.doc_lengths:
            self.doc_norms[doc_id] = math.sqrt(squared_sums[doc_id])

    def get_term_info(self, term: str) -> dict | None:
        """Επιστρέφει το df και τα postings ενός όρου (ή None αν δεν υπάρχει)."""
        return self.index.get(term)


if __name__ == "__main__":
    # Δοκιμή κατασκευής του ευρετηρίου τοπικά
    docs = parse_cf_documents()
    inv_index = InvertedIndex()
    inv_index.build(docs)

    print(f"Σύνολο εγγράφων (N): {inv_index.total_docs}")
    print(f"Μοναδικοί όροι στο λεξιλόγιο: {len(inv_index.index)}")
    print(f"Μέσο μήκος εγγράφου (avgdl): {inv_index.avg_doc_length:.2f} όροι")

    # Δοκιμαστική αναζήτηση ενός όρου (μετά από stemming: 'fibrosis' -> 'fibrosi')
    sample_term = "fibrosi"
    info = inv_index.get_term_info(sample_term)
    if info:
        print(f"Όρος '{sample_term}': εμφανίζεται σε {info['df']} έγγραφα (df).")
        first_3 = list(info['postings'].items())[:3]
        print(f"Πρώτα 3 postings (doc_id, tf): {first_3}")