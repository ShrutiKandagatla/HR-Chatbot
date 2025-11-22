import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import re

class HRChatbot:
    """
    HRChatbot with:
    - FAQ retrieval (TF-IDF + fuzzy)
    - Rule-based handlers
    - Short-term memory (remembers last intent/entity for 1 follow-up turn)
    """

    def __init__(self, faq_path='data/faqs.csv', emp_path='data/mock_employee_data.csv',
                 similarity_threshold=0.45):
        # Load FAQ data
        self.faq_df = pd.read_csv(faq_path)
        self.questions = self.faq_df['question'].astype(str).tolist()
        self.answers = self.faq_df['answer'].astype(str).tolist()
        self.categories = self.faq_df['category'].astype(str).tolist()

        # TF-IDF training
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.q_vectors = self.vectorizer.fit_transform(self.questions)
        self.threshold = similarity_threshold

        # Load employee database (CSV)
        self.emp_db = pd.read_csv(emp_path)
        self.emp_db['employee_id'] = self.emp_db['employee_id'].str.upper()

        # Short-term context (for one-turn follow-ups)
        self.last_intent = None       # e.g., 'ask_leave', 'ask_details', 'ask_payslip'
        self.last_entity = None       # e.g., employee id or other entity to carry over

    # -----------------------------
    # Text preprocessing
    # -----------------------------
    def preprocess(self, text):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return text

    # -----------------------------
    # Extract employee id (EMP\d+)
    # -----------------------------
    def extract_employee_id(self, query):
        match = re.search(r"(EMP\d+)", query.upper())
        return match.group(1) if match else None

    # -----------------------------
    # Employee CSV lookup
    # -----------------------------
    def get_employee(self, emp_id):
        if emp_id is None:
            return None
        emp_id = emp_id.upper().strip()
        row = self.emp_db[self.emp_db['employee_id'] == emp_id]
        return row.iloc[0] if not row.empty else None

    # -----------------------------
    # Responses: leave balance & details
    # -----------------------------
    def leave_balance(self, emp_id):
        emp = self.get_employee(emp_id)
        if emp is None:
            return f"❌ Employee ID **{emp_id}** not found."
        return (
            f"### Leave Balance for {emp['name']} ({emp['employee_id']})\n"
            f"- **Paid Leaves:** {emp['paid_leaves']}\n"
            f"- **Sick Leaves:** {emp['sick_leaves']}\n"
            f"- **Department:** {emp['department']}"
        )

    def employee_details(self, emp_id):
        emp = self.get_employee(emp_id)
        if emp is None:
            return f"❌ Employee ID **{emp_id}** not found."
        return (
            f"### Employee Details\n"
            f"- **Name:** {emp['name']}\n"
            f"- **Employee ID:** {emp['employee_id']}\n"
            f"- **Department:** {emp['department']}\n"
            f"- **Role:** {emp['role']}\n"
            f"- **Location:** {emp['location']}\n"
            f"- **Paid Leaves:** {emp['paid_leaves']}\n"
            f"- **Sick Leaves:** {emp['sick_leaves']}"
        )

    # -----------------------------
    # Rule-based handlers
    # -----------------------------
    def rule_based(self, query):
        q = query.lower()
        emp_id = self.extract_employee_id(query)

        # Leave queries
        if "leave" in q or "leaves" in q or "leave balance" in q:
            # store intent and ask for id if not provided
            if emp_id:
                # immediate answer if id present
                return self.leave_balance(emp_id)
            # store intent so follow-up emp id can be used
            self.last_intent = 'ask_leave'
            return "Please provide your Employee ID to check leave balance. Example: `EMP10234`"

        # Employee details/profile
        if "details" in q or "profile" in q or ("employee" in q and "details" in q):
            if emp_id:
                return self.employee_details(emp_id)
            self.last_intent = 'ask_details'
            return "Please provide the Employee ID to fetch details. Example: `EMP56789`"

        # Payslip / payroll
        if "payslip" in q or "salary" in q or "payroll" in q:
            # not an emp-specific action generally; still set last_intent if required
            self.last_intent = None
            return "You can download your payslip from **Payroll → Payslips → Select month → Download** in the portal."

        # Bank update
        if "bank" in q and ("update" in q or "change" in q):
            self.last_intent = None
            return "To update bank details: Go to **Profile → Bank Details → Edit**, enter new account details and submit. Changes will be verified."

        # Generic fallback for short queries (like "EMP10234" or "10234")
        # If user typed only an ID and last_intent exists, we handle that in retrieve()
        return None

    # -----------------------------
    # Main pipeline with short-term memory
    # -----------------------------
    def retrieve(self, query):
        """
        Retrieve an answer. This pipeline:
        1. Checks if query is just an employee ID follow-up and last_intent exists -> fulfills it.
        2. If not, runs rule-based intent detection (which may set last_intent).
        3. If still unresolved, tries TF-IDF FAQ retrieval + fuzzy fallback.
        """

        # Normalize
        raw_query = query.strip()
        query_p = self.preprocess(raw_query)

        # ---- 1) If the user submitted only an employee id (or short id), and we have a last_intent -> fulfill ----
        emp_id_candidate = self.extract_employee_id(raw_query)
        if emp_id_candidate and self.last_intent:
            intent = self.last_intent
            # clear intent after using
            self.last_intent = None

            if intent == 'ask_leave':
                return self.leave_balance(emp_id_candidate)
            if intent == 'ask_details':
                return self.employee_details(emp_id_candidate)
            # no matching intent: fall through

        # ---- 2) Run rule-based handlers (these may set last_intent) ----
        rule_resp = self.rule_based(raw_query)
        if rule_resp:
            # If rule-based returned a prompt asking for more info (like ID), it already set last_intent.
            return rule_resp

        # ---- 3) TF-IDF retrieval + fuzzy fallback for FAQ answers ----
        qv = self.vectorizer.transform([query_p])
        sims = cosine_similarity(qv, self.q_vectors).flatten()
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        fuzzy_scores = [fuzz.token_set_ratio(raw_query.lower(), q.lower()) / 100 for q in self.questions]
        best_fuzzy_score = max(fuzzy_scores) if fuzzy_scores else 0.0

        if best_score >= self.threshold or best_fuzzy_score >= 0.75:
            # reset context (we answered directly from FAQ)
            self.last_intent = None
            return self.answers[best_idx]

        # ---- 4) If nothing matched, give friendly fallback ----
        return "I couldn't find an exact answer. You can try:\n- `Check leaves for EMP10234`\n- `Show employee details EMP56789`\n- `How to download payslip?`"

# Quick test when running the module directly
if __name__ == '__main__':
    bot = HRChatbot()
    tests = [
        "How many leaves do I have?",
        "EMP10234",               # follow-up should trigger leave answer
        "Show details for EMP90877",
        "How to update bank details?",
        "What is HRA?",
    ]
    for q in tests:
        print("Q:", q)
        print("A:", bot.retrieve(q))
        print("---")
