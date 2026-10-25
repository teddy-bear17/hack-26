import time
import random
from datetime import datetime, timedelta

# --- КАСТОМНЫЕ ИСКЛЮЧЕНИЯ ---
class IntegrationError(Exception):
    """Базовая ошибка интеграции."""
    pass

class TimeoutError(IntegrationError):
    """Сервер не ответил вовремя."""
    pass

class AuthError(IntegrationError):
    """Пользователь не авторизован в системе (протух токен)."""
    pass

class BankAPI:
    """Имитация внутреннего банковского API (DirectBank / Open API)."""
    
    @staticmethod
    def get_pending_payments(company_inn: str) -> dict:
        time.sleep(random.uniform(0.1, 0.3))
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        
        return {
            "status": "success",
            "source": "Bank API",
            "data": [
                {"id": "P-1044", "date": tomorrow, "amount": 150000, "direction": "Списание", "purpose": "Оплата аренды"},
                {"id": "P-1045", "date": tomorrow, "amount": 45000, "direction": "Списание", "purpose": "Услуги связи"}
            ]
        }

class ERP1C:
    """Имитация учетной системы (1С:Предприятие / МойСклад)."""
    
    @staticmethod
    def get_unpaid_invoices(company_inn: str, token: str = None) -> dict:
        if not token:
            raise AuthError("Токен авторизации 1С отсутствует или просрочен.")
            
        time.sleep(random.uniform(0.5, 1.0))
        
        # 20% шанс, что сервер 1С упадет по таймауту
        if random.random() < 0.2:
            raise TimeoutError("Превышено время ожидания ответа от 1С (5 сек).")
            
        in_3_days = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
        in_7_days = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        return {
            "status": "success",
            "source": "1C ERP",
            "data": [
                {"invoice_id": "INV-2026-88", "expected_date": in_3_days, "amount": 320000, "type": "Поступление"},
                {"invoice_id": "INV-2026-89", "expected_date": in_7_days, "amount": 185000, "type": "Поступление"}
            ]
        }

class FNSRegistry:
    """Имитация открытых государственных реестров (ФНС, ЕФРСБ)."""
    
    @staticmethod
    def check_counterparties(inn_list: list) -> dict:
        time.sleep(random.uniform(0.1, 0.4))
        
        results = {}
        high_risk = {
            '0006094950': {"status": "CRITICAL", "reason": "Находится в стадии банкротства"},
            '0062343230': {"status": "WARNING", "reason": "Задолженность по налогам > 500 тыс. руб."},
            '0068275202': {"status": "OK", "reason": "Действующее предприятие"}
        }
        
        for inn in inn_list:
            clean_inn = str(inn).strip()
            results[clean_inn] = high_risk.get(clean_inn, {"status": "OK", "reason": ""})
            
        return {
            "status": "success",
            "source": "FNS/EFRSB API",
            "data": results
        }