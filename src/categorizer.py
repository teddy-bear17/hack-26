import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TransactionCategorizer:
  def __init__(self, ref_path='sample_data/top_50_categories.csv'):
    self.ref_path = ref_path
    self.threshold = 0.45
    self.ref_df = None
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
    t = re.sub(r'\s+', ' ', t).strip()
    return t

  def _rule_category(self, text: str):
    if not isinstance(text, str):
      return None
    t = text.lower()
    rules = [
        (r'перевод\w*\s+собственн\w*\s+средств|пополнени\w*\s+счета\s+собственн', 'Переводы между собственными счетами'),
        (r'сопровождени\w*\s+1с|1с\b', 'Подписки на ПО и сервисы'),
        (r'предоплат\w*.*договор\w*\s+поставк|доплат\w*\s+по\s+упд', 'Оплата поставщикам за товар и продукцию'),
        (r'арендн\w*\s+плат|аренд\w*\s+земельн|аренд\w*\s+дол', 'Аренда имущества'),
        (r'аренд\w*\s+нежил|аренд\w*\s+помещ|арендодател', 'Аренда имущества'),
        (r'^енп$|единый\s+налогов\w*\s+платеж|единый\s+налогов\w*\s+платёж', 'Налоги и взносы в бюджет'),
        (r'казначейств|поручени\w*\s+налогового\s+органа|взыскани\w*\s+задолженности\s+по\s+енп', 'Налоги и взносы в бюджет'),
        (r'комисси\w*.*(ведение|расчетн|счет)|ведение\s+счета|расчетно-кассов', 'Банковские комиссии и обслуживание'),
        (r'овердрафт|процент\w*.*пользовани', 'Проценты по кредитам и овердрафту'),
        (r'заработн|аванс\s+по\s+заработн|аванс\s+по\s+зар|перечислени\w*\s+на\s+счет\s+сотрудник', 'Зарплата и авансы сотрудникам'),
        (r'договор\w*\s+гпх|самозанят|физическ\w*\s+лицо', 'Выплаты по ГПХ и самозанятым'),
        (r'электроэнерг|энергосбыт|теплоснабж|водоснабж|водоканал|газоснабж|коммунальн', 'Коммунальные услуги и ресурсы'),
        (r'услуг\w*\s+связи|доступ\w*\s+в\s+интернет|\bинтернет\b|телефон', 'Связь и интернет'),
        (r'бухгалтер', 'Бухгалтерские услуги'),
        (r'транспортно-экспедиц|транспортн\w*\s+услуг|грузоперевоз', 'Транспортно-экспедиционные услуги'),
        (r'канцтовар|хозтовар|бытов\w*\s+хим', 'Канцтовары и хозтовары'),
        (r'оплат\w*\s+за\s+[а-я]', 'Оплата поставщикам за товар и продукцию'),
        (r'взнос\s+наличн|торговая\s+выручка', 'Прочие поступления / инкассация'),
    ]
    for pat, cat in rules:
      if re.search(pat, t):
        return cat
    return None

  def _init_reference(self):
    try:
      ref = pd.read_csv(self.ref_path, encoding='utf-8-sig')
      ref = ref[ref['cluster'] != -1].reset_index(drop=True)
      ref['norm'] = ref['representative_purpose'].apply(self._normalize)
      self.ref_df = ref
      self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, sublinear_tf=True)
      self.X_ref = self.vectorizer.fit_transform(self.ref_df['norm'])
    except Exception:
      # Защита, если файл не найден
      self.ref_df = pd.DataFrame(columns=['category_ru', 'norm'])
      self.vectorizer = TfidfVectorizer()
      self.X_ref = self.vectorizer.fit_transform(["заглушка"])
      self.ref_df = pd.DataFrame({'category_ru': ['Прочее'], 'norm': ['заглушка']})

  def predict(self, df: pd.DataFrame, purpose_col: str = 'payment_purpose') -> pd.DataFrame:
    df = df.copy()
    if purpose_col not in df.columns:
      df[purpose_col] = "Неизвестно"
        
    df['norm'] = df[purpose_col].apply(self._normalize)
    
    # ЗАЩИТА ОТ ПУСТОГО СПРАВОЧНИКА
    if self.ref_df is None or self.ref_df.empty or self.X_ref is None or self.X_ref.shape[0] == 0:
      df['category_tfidf'] = 'Прочее'
      df['match_score'] = 0.0
    else:
      X_main = self.vectorizer.transform(df['norm'])
      sim = cosine_similarity(X_main, self.X_ref)
      best_idx = sim.argmax(axis=1)
      best_score = sim.max(axis=1)
        
      # Безопасное извлечение
      df['category_tfidf'] = [self.ref_df.iloc[idx]['category_ru'] if idx < len(self.ref_df) else 'Прочее' for idx in best_idx]
      df['match_score'] = best_score
        
    df['category_rule'] = df[purpose_col].apply(self._rule_category)
    
    df['category_ru'] = np.where(
      df['category_rule'].notna(), df['category_rule'],
      np.where(df['match_score'] >= self.threshold, df['category_tfidf'], 'Прочее / неклассифицированное')
    )
    return df

  def predict_single(self, text: str) -> str:
    norm_text = self._normalize(text)
    rule_cat = self._rule_category(text)
    if rule_cat:
      return rule_cat
    if self.X_ref is not None and self.vectorizer is not None and not self.ref_df.empty:
      vec = self.vectorizer.transform([norm_text])
      sim = cosine_similarity(vec, self.X_ref)
      best_score = sim.max()
      best_idx = sim.argmax()
      if best_score >= self.threshold:
        return self.ref_df.iloc[best_idx]['category_ru']
    words = [w for w in norm_text.split() if len(w) > 4]
    if words:
      return f"Новая: {words[0].capitalize()}"
    return "Прочее / неклассифицированное"