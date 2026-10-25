"""
Генератор AI-выводов на основе метрик модели.
Только для вкладки "Отчет ML" (доступна админу).
"""

import numpy as np


def generate_ai_conclusion(feature_importances, features, mae, mape, accuracy):
    top_idx = int(np.argmax(feature_importances))
    top_feature = features[top_idx]
    top_share = float(feature_importances[top_idx]) * 100

    second_idx = None
    for i in np.argsort(feature_importances)[::-1]:
        if i != top_idx:
            second_idx = int(i)
            break
    second_feature = features[second_idx] if second_idx is not None else None
    second_share = float(feature_importances[second_idx]) * 100 if second_idx is not None else 0

    lines = [
        f"Наибольший вклад в прогноз вносит фактор «{top_feature}» ({top_share:.1f}%).",
    ]
    if second_feature:
        lines.append(f"На втором месте — «{second_feature}» ({second_share:.1f}%).")

    if 'месяц' in top_feature.lower():
        lines.append(
            "Подтверждается сильная зависимость от календарных платежей: "
            "налоги, зарплаты, аренда сконцентрированы на определенных числах. "
            "Модель выучила эти паттерны."
        )
    elif 'недел' in top_feature.lower():
        lines.append(
            "Основной сигнал — недельная сезонность. "
            "Платежи сконцентрированы в рабочие дни, в выходные — затишье."
        )
    elif 'баланс' in top_feature.lower() or 'среднее' in top_feature.lower():
        lines.append(
            "Модель опирается на инерцию денежного потока: "
            "текущий баланс коррелирует с завтрашним. "
            "Надежно для краткосрочного прогноза, менее — на горизонте больше 2-3 недель."
        )
    else:
        lines.append("Обнаружена сложная нелинейная зависимость.")

    lines.append("")
    lines.append("**Качество прогноза:**")
    lines.append(f"- MAE: **{mae:,.0f} руб.**")
    lines.append(f"- MAPE: **{mape:.1f}%**")
    lines.append(f"- Доля дней с ошибкой < 10%: **{accuracy:.1f}%**")

    if accuracy >= 80:
        q = "высокая. Модель надежна для краткосрочного планирования."
    elif accuracy >= 60:
        q = "приемлемая. Улавливает тренды, но есть волатильность."
    else:
        q = "низкая. Рекомендуется загрузить больше истории (60-90 дней)."

    lines.append(f"Интерпретация: надежность {q}")
    return "\n".join(lines)