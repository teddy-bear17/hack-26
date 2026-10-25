import os
import re
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    HAS_EMBEDDINGS = True
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_EMBEDDINGS = False


class TransactionCategorizer:
    def __init__(self):
        self.threshold = 0.60 if HAS_EMBEDDINGS else 0.45
        self.ref_df = None
        self.model = None
        self.ref_embeddings = None
        self.vectorizer = None
        self.X_ref = None
        self._init_reference()

    def _normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ''
        t = text.lower().replace('ё', 'е')
        t = re.sub(r'№\s*[\w\-/\.]+', ' ', t)
        t = re.sub(r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b', ' ', t)
        t = re.sub(r'\b\d+[.,]?\d*\s*%', ' ', t)
        t = re.sub(r'\b\d+[.,]\d+\b', ' ', t)
        t = re.sub(r'\b\d+\b', ' ', t)
        t = re.sub(r'[^\w\s]', ' ', t)
        return re.sub(r'\s+', ' ', t).strip()

    def _rule_category(self, text: str):
        if not isinstance(text, str):
            return None
        t = text.lower()
        
        # Жесткие правила. Проверяются ДО нейросети.
        rules = [
            (r'налог|налогов|енп|казначейств|фнс', 'Налоги и взносы в бюджет'),
            (r'стройматериал|ремонт|стройк|бетон|цемент|кирпич|арматур|доск|брус', 'Стройматериалы и ремонт'),
            (r'заработн|аванс|зарплат|перечислени\w*\s+на\s+счет\s+сотрудник', 'Зарплата и авансы сотрудникам'),
            (r'арендн\w*\s+плат|аренд\w*\s+земельн|аренд\w*\s+дол|аренд\w*\s+нежил|аренд\w*\s+помещ|арендодател|аренд', 'Аренда имущества'),
            (r'закупк|поставщик|оплат\w*\s+за\s+товар|оплат\w*\s+за\s+продукц', 'Оплата поставщикам за товар и продукцию'),
            (r'перевод\w*\s+собственн\w*\s+средств|пополнени\w*\s+счета\s+собственн', 'Переводы между собственными счетами'),
            (r'сопровождени\w*\s+1с|\b1с\b', 'Подписки на ПО и сервисы'),
            (r'комисси\w*.*(ведение|расчетн|счет)|ведение\s+счета|расчетно-кассов', 'Банковские комиссии и обслуживание'),
            (r'овердрафт|процент\w*.*пользовани|кредит', 'Проценты по кредитам и овердрафту'),
            (r'договор\w*\s+гпх|самозанят|физическ\w*\s+лицо', 'Выплаты по ГПХ и самозанятым'),
            (r'электроэнерг|энергосбыт|теплоснабж|водоснабж|водоканал|газоснабж|коммунальн', 'Коммунальные услуги и ресурсы'),
            (r'услуг\w*\s+связи|доступ\w*\s+в\s+интернет|\bинтернет\b|телефон', 'Связь и интернет'),
            (r'бухгалтер', 'Бухгалтерские услуги'),
            (r'транспортно-экспедиц|транспортн\w*\s+услуг|грузоперевоз', 'Транспортно-экспедиционные услуги'),
            (r'канцтовар|хозтовар|бытов\w*\s+хим', 'Канцтовары и хозтовары'),
            (r'взнос\s+наличн|торговая\s+выручка', 'Прочие поступления / инкассация'),
        ]
        for pat, cat in rules:
            if re.search(pat, t):
                return cat
        return None

    def _init_reference(self):
        emb_path = "data/category_embeddings.npy"
        texts_path = "data/category_texts_for_embeddings.csv"
        csv_path = "data/ai_generated_categories.csv"

        # Вариант 1: Предвычисленные векторы
        if HAS_EMBEDDINGS and os.path.exists(emb_path) and os.path.exists(texts_path):
            try:
                print("[Categorizer] Загрузка предвычисленных векторов...")
                self.ref_embeddings = np.load(emb_path)
                self.ref_df = pd.read_csv(texts_path, sep=';', encoding='utf-8-sig')

                if len(self.ref_df) != len(self.ref_embeddings):
                    raise ValueError("Несовпадение текстов и векторов")
                
                # Убираем префикс AI: если он там остался
                self.ref_df['category_ru'] = self.ref_df['category_ru'].str.replace('AI: ', '', regex=False)

                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print(f"[Categorizer] Готово. Эталонов: {len(self.ref_df)}")
                return
            except Exception as e:
                print(f"[Categorizer] Ошибка векторов: {e}. Fallback на TF-IDF.")
                self.ref_embeddings = None

        # Вариант 2: TF-IDF
        if not os.path.exists(csv_path):
            csv_path = "sample_data/top_50_categories.csv"

        if not os.path.exists(csv_path):
            self.ref_df = pd.DataFrame({'category_ru': ['Прочее'], 'norm': ['пусто'], 'representative_purpose': ['пусто']})
            self.vectorizer = TfidfVectorizer()
            self.X_ref = self.vectorizer.fit_transform(['пусто'])
            return

        print(f"[Categorizer] Загрузка CSV: {csv_path}")
        ref = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')

        garbage = r'^AI:\s*(Группа\s*\d+|\(\d+%\)\s*\(\d+%\)|\(\d+%\)|\d+(\s|$))'
        mask = ref['category_ru'].astype(str).str.match(garbage, na=False)
        if mask.sum() > 0:
            ref = ref[~mask]

        ref = ref[ref['category_ru'].astype(str).str.len() > 3].reset_index(drop=True)
        
        # Убираем префикс AI:
        ref['category_ru'] = ref['category_ru'].str.replace('AI: ', '', regex=False)
        
        ref['norm'] = ref['representative_purpose'].apply(self._normalize)
        self.ref_df = ref

        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.X_ref = self.vectorizer.fit_transform(self.ref_df['norm'])

    def predict(self, df: pd.DataFrame, purpose_col: str = 'payment_purpose') -> pd.DataFrame:
        df = df.copy()
        if purpose_col not in df.columns:
            df[purpose_col] = 'Неизвестно'

        df['norm'] = df[purpose_col].apply(self._normalize)
        texts = [t if t else 'пусто' for t in df['norm'].tolist()]

        if self.model is not None and self.ref_embeddings is not None:
            query_emb = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            ref_tensor = torch.tensor(self.ref_embeddings, device=query_emb.device, dtype=query_emb.dtype)
            scores = util.cos_sim(query_emb, ref_tensor)
            best_scores, best_idx = scores.max(dim=1)
            df['category_ml'] = [self.ref_df.iloc[i]['category_ru'] for i in best_idx.cpu().numpy()]
            df['match_score'] = best_scores.cpu().numpy()
        elif self.vectorizer is not None:
            X_q = self.vectorizer.transform(texts)
            sim = cosine_similarity(X_q, self.X_ref)
            df['category_ml'] = [self.ref_df.iloc[i]['category_ru'] for i in sim.argmax(axis=1)]
            df['match_score'] = sim.max(axis=1)
        else:
            df['category_ml'] = 'Прочее'
            df['match_score'] = 0.0

        df['category_rule'] = df[purpose_col].apply(self._rule_category)

        final = []
        for _, row in df.iterrows():
            score = row['match_score']
            rule = row['category_rule']
            ml = row['category_ml']

            if score >= 0.85:
                final.append(ml)
            elif pd.notna(rule):
                final.append(rule)
            elif score >= self.threshold:
                final.append(ml)
            else:
                final.append('Прочее')

        df['category_ru'] = final
        return df

    def predict_single(self, text: str) -> str:
        rule = self._rule_category(text)
        norm = self._normalize(text)
        if not norm:
            return 'Прочее'

        # Если правило сработало, отдаем приоритет ему (для налогов и стройки это критично)
        if rule:
            return rule

        if self.model is not None and self.ref_embeddings is not None:
            query_emb = self.model.encode([norm], convert_to_tensor=True, show_progress_bar=False)
            ref_tensor = torch.tensor(self.ref_embeddings, device=query_emb.device, dtype=query_emb.dtype)
            scores = util.cos_sim(query_emb, ref_tensor)[0]
            best_score = scores.max().item()
            best_cat = self.ref_df.iloc[scores.argmax().item()]['category_ru']

            if best_score >= self.threshold:
                return best_cat
        elif self.vectorizer is not None:
            X_q = self.vectorizer.transform([norm])
            sim = cosine_similarity(X_q, self.X_ref)[0]
            if sim.max() >= self.threshold:
                return self.ref_df.iloc[sim.argmax()]['category_ru']

        return 'Прочее'