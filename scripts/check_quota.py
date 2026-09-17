from kagglesdk.kaggle_client import KaggleClient
from kagglesdk.models.types.model_proxy_api_service import ApiGetModelProxyQuotasRequest

try:
    client = KaggleClient()
    resp = client.models.model_proxy_api_client.get_model_proxy_quotas(ApiGetModelProxyQuotasRequest())
    print("\n=== KAGGLE MODEL PROXY API QUOTA ===")
    for q in resp.quota_balances:
        period = "DAILY (Ngày)" if "DAILY" in str(q.refill_period) else "MONTHLY (Tháng)"
        remaining = q.total_quota_allowed - q.quota_used
        print(f"  • {period:16}: Đã dùng ${q.quota_used:.4f} / Hạn mức ${q.total_quota_allowed:.2f} (Còn lại: ${remaining:.4f})")
except Exception as e:
    print(f"Không lấy được Model Quota: {e}")
