# %%
# Syntax reference: kaggle_benchmarks_reference.md
import os
import json
import time
from dotenv import load_dotenv
load_dotenv()
import kaggle_benchmarks as kbench

# 5 pilot samples from CustomTools-VI
PILOT_SAMPLES = [
  {
    "id": "custom_006401",
    "query": "Please giúp mình tìm quán phở bò ở Cà Mau, giá dưới 50.000 đồng, điểm đánh giá từ 2,5 trở lên, đồng thời đặt 2 phần bún bò Huế giao đến phường Cầu Giấy, không cay, giao trong tối đa 30 phút nhé. Mình cần hoàn tất phần chuẩn bị còn thiếu.",
    "gold_tools": ["vi_search_restaurants", "vi_order_food"],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "vi_order_food",
          "description": "Đặt món ăn giao đến địa chỉ tại Việt Nam.",
          "parameters": {
            "type": "object",
            "properties": {
              "dish": {"type": "string", "description": "Món ăn cần đặt"},
              "quantity": {"type": "integer", "description": "Số phần cần đặt"},
              "delivery_location": {"type": "string", "description": "Địa điểm giao món"},
              "spicy": {"type": "boolean", "description": "Có chọn vị cay hay không"},
              "max_delivery_minutes": {"type": "integer", "description": "Thời gian giao tối đa bằng phút"}
            },
            "required": ["dish", "quantity", "delivery_location"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "vi_search_restaurants",
          "description": "Tìm quán ăn theo món, địa điểm và mức giá.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "Địa điểm cần tìm"},
              "dish": {"type": "string", "description": "Món ăn cần tìm"},
              "max_price_vnd": {"type": "integer", "description": "Mức giá tối đa bằng VND"},
              "min_rating": {"type": "number", "description": "Điểm đánh giá tối thiểu"}
            },
            "required": ["location", "dish"]
          }
        }
      }
    ]
  },
  {
    "id": "custom_006402",
    "query": "Phiền bạn đặt 4 phần mì Quảng giao đến phường Ninh Kiều, không cay, giao trong tối đa 30 phút, đồng thời tìm quán bánh xèo ở Sài Gòn, giá dưới 250k giúp tôi. Tôi muốn kiểm tra kỹ trước khi quyết định.",
    "gold_tools": ["vi_order_food", "vi_search_restaurants"],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "vi_order_food",
          "description": "Đặt món ăn giao đến địa chỉ tại Việt Nam.",
          "parameters": {
            "type": "object",
            "properties": {
              "dish": {"type": "string", "description": "Món ăn cần đặt"},
              "quantity": {"type": "integer", "description": "Số phần cần đặt"},
              "delivery_location": {"type": "string", "description": "Địa điểm giao món"},
              "spicy": {"type": "boolean", "description": "Có chọn vị cay hay không"},
              "max_delivery_minutes": {"type": "integer", "description": "Thời gian giao tối đa bằng phút"}
            },
            "required": ["dish", "quantity", "delivery_location"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "vi_search_restaurants",
          "description": "Tìm quán ăn theo món, địa điểm và mức giá.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "Địa điểm cần tìm"},
              "dish": {"type": "string", "description": "Món ăn cần tìm"},
              "max_price_vnd": {"type": "integer", "description": "Mức giá tối đa bằng VND"},
              "min_rating": {"type": "number", "description": "Điểm đánh giá tối thiểu"}
            },
            "required": ["location", "dish"]
          }
        }
      }
    ]
  },
  {
    "id": "custom_006403",
    "query": "Tôi muốn quy đổi 500 USD sang VND và tìm mã giảm giá Shopee ngành thời trang.",
    "gold_tools": ["vi_convert_currency", "vi_search_discount_codes"],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "vi_convert_currency",
          "description": "Quy đổi số tiền giữa các loại tiền tệ.",
          "parameters": {
            "type": "object",
            "properties": {
              "amount": {"type": "number", "description": "Số tiền cần đổi"},
              "from_currency": {"type": "string", "enum": ["VND", "USD", "EUR", "JPY"]},
              "to_currency": {"type": "string", "enum": ["VND", "USD", "EUR", "JPY"]}
            },
            "required": ["amount", "from_currency", "to_currency"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "vi_search_discount_codes",
          "description": "Tìm mã giảm giá cho sàn hoặc sản phẩm.",
          "parameters": {
            "type": "object",
            "properties": {
              "platform": {"type": "string", "enum": ["Shopee", "Lazada", "Tiki"]},
              "category": {"type": "string", "description": "Danh mục sản phẩm"}
            },
            "required": ["platform"]
          }
        }
      }
    ]
  },
  {
    "id": "custom_006404",
    "query": "Tìm khách sạn tại Đà Nẵng nhận phòng ngày 2026-09-20 cho 2 đêm giá dưới 1 triệu/đêm, và tìm địa điểm tham quan văn hóa tại Đà Nẵng.",
    "gold_tools": ["vi_search_accommodations", "vi_search_attractions"],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "vi_search_accommodations",
          "description": "Tìm nơi lưu trú theo địa điểm và ngân sách.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "Địa điểm lưu trú"},
              "check_in_date": {"type": "string", "description": "Ngày nhận phòng"},
              "nights": {"type": "integer", "description": "Số đêm lưu trú"},
              "max_price_vnd": {"type": "integer", "description": "Giá tối đa mỗi đêm"}
            },
            "required": ["location", "check_in_date", "nights"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "vi_search_attractions",
          "description": "Tìm địa điểm tham quan tại Việt Nam.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "Tỉnh hoặc thành phố"},
              "category": {"type": "string", "enum": ["thiên nhiên", "lịch sử", "văn hóa", "giải trí"]}
            },
            "required": ["location"]
          }
        }
      }
    ]
  },
  {
    "id": "custom_006405",
    "query": "Tính thuế thu nhập cá nhân cho lương gộp 35 triệu với 1 người phụ thuộc, sau đó tìm sự kiện âm nhạc tại Hà Nội ngày 2026-09-25.",
    "gold_tools": ["vi_calculate_personal_income_tax", "vi_search_events"],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "vi_calculate_personal_income_tax",
          "description": "Ước tính thuế thu nhập cá nhân hàng tháng.",
          "parameters": {
            "type": "object",
            "properties": {
              "gross_income_vnd": {"type": "integer", "description": "Thu nhập gộp hàng tháng"},
              "dependents": {"type": "integer", "description": "Số người phụ thuộc"}
            },
            "required": ["gross_income_vnd", "dependents"]
          }
        }
      },
      {
        "type": "function",
        "function": {
          "name": "vi_search_events",
          "description": "Tìm sự kiện giải trí theo địa điểm và ngày.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "Địa điểm tổ chức"},
              "event_type": {"type": "string", "description": "Loại sự kiện"},
              "date": {"type": "string", "description": "Ngày diễn ra"}
            },
            "required": ["location", "date"]
          }
        }
      }
    ]
  }
]

# %%
@kbench.task(name="tool-calling-vi-pilot", description="Pilot benchmark on 5 Vietnamese Tool Calling samples")
def tool_calling_vi_pilot(llm) -> dict:
    correct_calls = 0
    total_calls = len(PILOT_SAMPLES)
    details = []

    for item in PILOT_SAMPLES:
        t0 = time.time()
        extra_kwargs = {}
        if "gpt" in str(llm.model).lower() or "luna" in str(llm.model).lower():
            extra_kwargs["reasoning_effort"] = "none"
        try:
            resp = llm.client.chat.completions.create(
                model=llm.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là trợ lý AI hữu ích. Nếu người dùng yêu cầu nhiều tác vụ, hãy gọi TẤT CẢ các hàm phù hợp cùng một lúc."
                    },
                    {"role": "user", "content": item["query"]}
                ],
                tools=item["tools"],
                tool_choice="auto",
                **extra_kwargs
            )
            elapsed_ms = (time.time() - t0) * 1000
            called_tools = [
                tc.function.name for tc in (resp.choices[0].message.tool_calls or [])
            ]
            # Verify all gold tools were called
            gold = set(item["gold_tools"])
            pred = set(called_tools)
            is_correct = (gold == pred)
            if is_correct:
                correct_calls += 1

            details.append({
                "id": item["id"],
                "gold": item["gold_tools"],
                "predicted": called_tools,
                "is_correct": is_correct,
                "latency_ms": round(elapsed_ms, 1)
            })
        except Exception as e:
            details.append({
                "id": item["id"],
                "error": str(e)
            })

    accuracy = correct_calls / total_calls
    kbench.assertions.assert_true(accuracy >= 0.6, expectation=f"Accuracy should be >= 60%, got {accuracy * 100:.1f}%")

    return {
        "model": llm.model,
        "total_samples": total_calls,
        "correct_tool_selection": correct_calls,
        "accuracy": accuracy,
        "details": details
    }

# %%
tool_calling_vi_pilot.run(kbench.llm)
