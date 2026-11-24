import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Κατεβάζουμε τη λίστα των αγγλικών stopwords (τρέχει μόνο την 1η φορά)
nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def preprocess_text(text: str) -> list[str]:
    """
    Εφαρμόζει τη βασική ροή προεπεξεργασίας IR:
    1. Lowercasing (πεζά γράμματα)
    2. Αφαίρεση σημείων στίξης και αριθμών (κρατάμε μόνο γράμματα)
    3. Tokenization (διαχωρισμός σε λέξεις)
    4. Αφαίρεση Stopwords
    5. Stemming (Porter Stemmer)
    """
    text = text.lower()
    # Κρατάμε μόνο αλφαβητικούς χαρακτήρες (a-z)
    tokens = re.findall(r'\b[a-z]+\b', text)
    # Αφαιρούμε stopwords και λέξεις 1 γράμματος, και κάνουμε stemming
    cleaned_tokens = [
        STEMMER.stem(token) 
        for token in tokens 
        if token not in STOPWORDS and len(token) > 1 #ελεγχος για λέξεις μεγαλύτερες του 1 χαρακτήρα και αν ανηκουν στα stopwords (ετοιμες λίστες με βασικες λεςξεις που δεν προσφέρουν πληροφορία )
    ]
    return cleaned_tokens

def parse_cf_documents(data_dir: str = "data") -> list[dict]:
    """
    Διαβάζει τα αρχεία cf74-cf79 της συλλογής Cystic Fibrosis
    και επιστρέφει λίστα με λεξικά: {'doc_id': str, 'title': str, 'text': str}
    """
    doc_files = ['cf74', 'cf75', 'cf76', 'cf77', 'cf78', 'cf79']
    documents = []

    for filename in doc_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Προσοχή: Το αρχείο {filepath} δεν βρέθηκε!")
            continue

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            #ολα τα παραπανω ειναι για να ανοιγουν το αρχειο και να διαβαζουν το περιεχομενο του αρχειου σωστα για το καθε λειτουργικο

        # Κάθε εγγραφή στη συλλογή C.F. ξεκινάει με 'PN '
        records = re.split(r'\n(?=PN\s)', content.strip()) #δημιουργουνται καινουργια "υποαρχεια" για καθε εγγραφη που ξεκιναει με PN και χωριζονται με το regex.

        for record in records: #ψαχνει καθε PN και βρισκει το RN και το doc_id για να τα αποθηκευσει στο λεξικο documents
            rn_match = re.search(r'^RN\s+(\d+)', record, re.MULTILINE)
            if not rn_match:
                continue
            
            # Το doc_id ως ακέραιος σε string μορφή (π.χ. '00001' -> '1') για εύκολη αντιστοίχιση με το λυσάρι
            doc_id = str(int(rn_match.group(1)))

            # Εξαγωγή Τίτλου (TI) - μπορεί να πιάνει πολλές γραμμές μέχρι το επόμενο tag 2 κεφαλαίων γραμμάτων
            ti_match = re.search(r'^TI\s+(.*?)(?=\n[A-Z]{2}\s|\Z)', record, re.MULTILINE | re.DOTALL)
            title = re.sub(r'\s+', ' ', ti_match.group(1)).strip() if ti_match else "" #αντικαθιστά πολλαπλά κενά με ένα και αφαιρεί περιττά κενά στην αρχή και στο τέλος

            # Εξαγωγή Περίληψης (AB ή EX) (ομοιως με τον τίτλο, μπορεί να πιάνει πολλές γραμμές)
            ab_match = re.search(r'^(?:AB|EX)\s+(.*?)(?=\n[A-Z]{2}\s|\Z)', record, re.MULTILINE | re.DOTALL)
            abstract = re.sub(r'\s+', ' ', ab_match.group(1)).strip() if ab_match else ""

            full_text = f"{title} {abstract}".strip()
            processed_tokens = preprocess_text(full_text)
            
            documents.append({
                "doc_id": doc_id,
                "title": title,
                "text": full_text,
                "length": len(processed_tokens)
            })

    return documents

def parse_cf_queries(data_dir: str = "data") -> tuple[dict, dict]:
    """
    Διαβάζει το αρχείο cfquery και επιστρέφει:
    - queries: {query_id: query_text}
    - qrels (relevance judgments): {query_id: {doc_id: relevance_score}}
    """
    filepath = os.path.join(data_dir, 'cfquery')
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    records = re.split(r'\n(?=QN\s)', content.strip())
    queries = {}
    qrels = {}

    for record in records:
        qn_match = re.search(r'^QN\s+(\d+)', record, re.MULTILINE)
        if not qn_match:
            continue
        qid = str(int(qn_match.group(1)))

        qu_match = re.search(r'^QU\s+(.*?)(?=\n[A-Z]{2}\s|\Z)', record, re.MULTILINE | re.DOTALL)
        query_text = re.sub(r'\s+', ' ', qu_match.group(1)).strip() if qu_match else ""
        queries[qid] = query_text

        # Εξαγωγή Relevant Documents (RD)
        rd_match = re.search(r'^RD\s+(.*?)(?=\n[A-Z]{2}\s|\Z)', record, re.MULTILINE | re.DOTALL)
        qrels[qid] = {}
        if rd_match:
            # Κάθε καταχώρηση είναι ζευγάρι: doc_id και 4-ψήφιο score (π.χ. 139 1222)
            pairs = re.findall(r'(\d+)\s+(\d{4})', rd_match.group(1))
            for doc_num, score_str in pairs:
                doc_id = str(int(doc_num))
                # Αθροίζουμε τα 4 ψηφία των 4 κριτών (εύρος 0-8) ως βαθμό σχετικότητας
                rel_score = sum(int(digit) for digit in score_str)
                if rel_score > 0:
                    qrels[qid][doc_id] = rel_score

    return queries, qrels

# Γρήγορος έλεγχος αν τρέξουμε το αρχείο απευθείας
if __name__ == "__main__":
    docs = parse_cf_documents()
    queries, qrels = parse_cf_queries()
    print(f"Φορτώθηκαν {len(docs)} έγγραφα και {len(queries)} ερωτήματα.")
    if docs:
        print("Παράδειγμα εγγράφου 1:", docs[0]['title'])
        print("Tokens εγγράφου 1 (πρώτα 10):", preprocess_text(docs[0]['text'])[:10])