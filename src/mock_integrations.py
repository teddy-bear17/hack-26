import time
import random
from datetime import datetime, timedelta

class BankAPI:
    """Имитация внутреннего банковского API (DirectBank / Open API)."""
    
    @staticmethod
    def get_pending_payments(company_inn: str) -> dict:
        """Возвращает созданные, но еще не исполненные платежные поручения."""
        # Имитация сетевой задержки
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
    def get_unpaid_invoices(company_inn: str) -> dict:
        """Возвращает выставленные, но не оплаченные счета."""
        time.sleep(random.uniform(0.2, 0.5))
        
        in_3_days = (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')
        in_7_days = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Шанс "отвала" API для демонстрации отказоустойчивости
        if random.random() < 0.05:
            return {"status": "error", "message": "1C Server Timeout 504"}
            
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
        """Массовая проверка ИНН контрагентов на риски (долги, банкротство)."""
        time.sleep(random.uniform(0.1, 0.4))
        
        results = {}
        # Захардкоженные ИНН из вашего датасета для демо
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