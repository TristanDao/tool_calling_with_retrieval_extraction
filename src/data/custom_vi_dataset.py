"""Generate and validate the reproducible CustomTools-VI v1 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

DOMAIN_NAMES = {
    "food": "Ẩm thực & Đặc sản",
    "travel": "Du lịch & Địa danh",
    "finance": "Tài chính & Ngân hàng",
    "education": "Giáo dục Việt Nam",
    "health": "Y tế & Sức khỏe",
    "ecommerce": "Thương mại điện tử",
    "transport": "Giao thông & Di chuyển",
    "public_services": "Hành chính công",
    "entertainment": "Giải trí & Lịch",
    "agriculture": "Nông nghiệp & Thời tiết",
}

RELATED_DOMAINS = {
    "food": ("travel", "ecommerce"),
    "travel": ("transport", "food"),
    "finance": ("public_services", "ecommerce"),
    "education": ("public_services", "health"),
    "health": ("public_services", "ecommerce"),
    "ecommerce": ("finance", "transport"),
    "transport": ("travel", "public_services"),
    "public_services": ("finance", "transport"),
    "entertainment": ("travel", "food"),
    "agriculture": ("finance", "travel"),
}

PROVINCES = [
    "Hà Nội",
    "Huế",
    "Lai Châu",
    "Điện Biên",
    "Sơn La",
    "Lạng Sơn",
    "Quảng Ninh",
    "Thanh Hóa",
    "Nghệ An",
    "Hà Tĩnh",
    "Cao Bằng",
    "Tuyên Quang",
    "Lào Cai",
    "Thái Nguyên",
    "Phú Thọ",
    "Bắc Ninh",
    "Hưng Yên",
    "Hải Phòng",
    "Ninh Bình",
    "Quảng Trị",
    "Đà Nẵng",
    "Quảng Ngãi",
    "Gia Lai",
    "Khánh Hòa",
    "Lâm Đồng",
    "Đắk Lắk",
    "TP. Hồ Chí Minh",
    "Đồng Nai",
    "Tây Ninh",
    "Cần Thơ",
    "Vĩnh Long",
    "Đồng Tháp",
    "Cà Mau",
    "An Giang",
]

LOCATION_SURFACES = {
    "Hà Nội": ["Hà Nội", "HN", "thủ đô"],
    "TP. Hồ Chí Minh": ["TP.HCM", "Sài Gòn", "SG"],
    "Đà Nẵng": ["Đà Nẵng", "ĐN"],
    "Cần Thơ": ["Cần Thơ", "Tây Đô"],
    "Huế": ["Huế", "cố đô Huế"],
}

DISTRICTS = [
    "phường Bến Thành",
    "phường Cầu Giấy",
    "phường Hải Châu",
    "phường Ninh Kiều",
    "phường Thuận Hóa",
    "khu vực trung tâm",
]
FOODS = [
    "phở bò",
    "bún bò Huế",
    "cơm tấm",
    "mì Quảng",
    "bánh xèo",
    "cao lầu",
    "bún chả",
    "hủ tiếu",
    "bánh canh",
    "lẩu mắm",
]
CUISINES = ["món Việt", "món chay", "đồ ăn miền Trung", "đặc sản địa phương", "hải sản"]
AIRLINES = ["Vietnam Airlines", "Vietjet Air", "Bamboo Airways", "Vietravel Airlines"]
BANKS = ["Vietcombank", "BIDV", "VietinBank", "Agribank", "Techcombank", "MB Bank"]
UNIVERSITIES = [
    "Đại học Bách khoa Hà Nội",
    "Đại học Kinh tế Quốc dân",
    "Đại học Y Dược TP.HCM",
    "Đại học Cần Thơ",
    "Đại học Huế",
]
SUBJECTS = ["Toán", "Ngữ văn", "Tiếng Anh", "Vật lý", "Hóa học", "Tin học"]
HOSPITALS = [
    "Bệnh viện Bạch Mai",
    "Bệnh viện Chợ Rẫy",
    "Bệnh viện Trung ương Huế",
    "Bệnh viện Đại học Y Dược TP.HCM",
]
MARKETPLACES = ["Shopee", "Lazada", "Tiki", "Sendo"]
CARRIERS = ["GHN", "GHTK", "Viettel Post", "VNPost", "J&T Express"]
RIDE_APPS = ["Grab", "be", "Xanh SM"]
CROPS = ["cà phê", "sầu riêng", "lúa", "thanh long", "hồ tiêu", "xoài", "vải thiều"]
EVENTS = ["liveshow", "hội sách", "triển lãm", "lễ hội ẩm thực", "giải chạy", "đêm nhạc"]
ZODIACS = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]


def param(
    schema_type: str,
    description: str,
    *,
    enum: list[str] | None = None,
    fmt: str | None = None,
    identifier: bool = False,
) -> dict[str, Any]:
    value: dict[str, Any] = {"type": schema_type, "description": description}
    if enum is not None:
        value["enum"] = enum
    if fmt is not None:
        value["format"] = fmt
    if identifier:
        value["x-identifier"] = True
    return value


def tool(
    name: str,
    description: str,
    domain: str,
    split: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "feature_group": DOMAIN_NAMES[domain],
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "x-domain": domain,
        "x-tool-split": split,
    }


def build_tools() -> list[dict[str, Any]]:
    p = param
    definitions = [
        tool(
            "vi_search_restaurants",
            "Tìm quán ăn theo món, địa điểm và mức giá.",
            "food",
            "seen",
            {
                "location": p("string", "Địa điểm cần tìm"),
                "dish": p("string", "Món ăn cần tìm"),
                "max_price_vnd": p("integer", "Mức giá tối đa bằng VND"),
            },
            ["location", "dish"],
        ),
        tool(
            "vi_order_food",
            "Đặt món ăn giao đến địa chỉ tại Việt Nam.",
            "food",
            "seen",
            {
                "dish": p("string", "Món ăn cần đặt"),
                "quantity": p("integer", "Số phần cần đặt"),
                "delivery_location": p("string", "Địa điểm giao món"),
                "spicy": p("boolean", "Có chọn vị cay hay không"),
            },
            ["dish", "quantity", "delivery_location"],
        ),
        tool(
            "vi_search_recipes",
            "Tìm công thức món Việt theo món và độ khó.",
            "food",
            "dev_unseen",
            {
                "dish": p("string", "Tên món ăn"),
                "difficulty": p("string", "Độ khó", enum=["dễ", "trung bình", "khó"]),
                "max_minutes": p("integer", "Thời gian nấu tối đa bằng phút"),
            },
            ["dish"],
        ),
        tool(
            "vi_estimate_meal_cost",
            "Ước tính tổng chi phí cho một bữa ăn.",
            "food",
            "test_unseen",
            {
                "people": p("integer", "Số người dùng bữa"),
                "meal_type": p("string", "Loại bữa ăn", enum=["bình dân", "trung cấp", "cao cấp"]),
                "location": p("string", "Địa điểm dùng bữa"),
            },
            ["people", "meal_type", "location"],
        ),
        tool(
            "vi_search_domestic_flights",
            "Tìm chuyến bay nội địa Việt Nam.",
            "travel",
            "seen",
            {
                "origin": p("string", "Điểm khởi hành"),
                "destination": p("string", "Điểm đến"),
                "departure_date": p("string", "Ngày khởi hành", fmt="date"),
                "airline": p("string", "Hãng hàng không ưu tiên"),
            },
            ["origin", "destination", "departure_date"],
        ),
        tool(
            "vi_search_accommodations",
            "Tìm nơi lưu trú theo địa điểm và ngân sách.",
            "travel",
            "seen",
            {
                "location": p("string", "Địa điểm lưu trú"),
                "check_in_date": p("string", "Ngày nhận phòng", fmt="date"),
                "nights": p("integer", "Số đêm lưu trú"),
                "max_price_vnd": p("integer", "Giá tối đa mỗi đêm bằng VND"),
            },
            ["location", "check_in_date", "nights"],
        ),
        tool(
            "vi_search_intercity_buses",
            "Tìm chuyến xe khách liên tỉnh.",
            "travel",
            "dev_unseen",
            {
                "origin": p("string", "Điểm đi"),
                "destination": p("string", "Điểm đến"),
                "departure_date": p("string", "Ngày đi", fmt="date"),
                "seat_type": p("string", "Loại ghế", enum=["ghế ngồi", "giường nằm", "limousine"]),
            },
            ["origin", "destination", "departure_date"],
        ),
        tool(
            "vi_search_attractions",
            "Tìm địa điểm tham quan tại Việt Nam.",
            "travel",
            "test_unseen",
            {
                "location": p("string", "Tỉnh hoặc thành phố"),
                "category": p(
                    "string",
                    "Loại địa điểm",
                    enum=["thiên nhiên", "lịch sử", "văn hóa", "giải trí"],
                ),
                "family_friendly_only": p(
                    "boolean", "Có chỉ lấy địa điểm phù hợp với gia đình hay không"
                ),
            },
            ["location"],
        ),
        tool(
            "vi_convert_currency",
            "Quy đổi số tiền giữa các loại tiền tệ.",
            "finance",
            "seen",
            {
                "amount": p("number", "Số tiền cần đổi"),
                "from_currency": p(
                    "string", "Mã tiền tệ nguồn", enum=["VND", "USD", "EUR", "JPY", "KRW"]
                ),
                "to_currency": p(
                    "string", "Mã tiền tệ đích", enum=["VND", "USD", "EUR", "JPY", "KRW"]
                ),
            },
            ["amount", "from_currency", "to_currency"],
        ),
        tool(
            "vi_calculate_loan_payment",
            "Tính khoản thanh toán vay định kỳ.",
            "finance",
            "seen",
            {
                "principal_vnd": p("integer", "Số tiền vay bằng VND"),
                "annual_rate_percent": p("number", "Lãi suất năm theo phần trăm"),
                "term_months": p("integer", "Thời hạn vay bằng tháng"),
            },
            ["principal_vnd", "annual_rate_percent", "term_months"],
        ),
        tool(
            "vi_calculate_savings_interest",
            "Tính tiền lãi tiết kiệm dự kiến.",
            "finance",
            "dev_unseen",
            {
                "deposit_vnd": p("integer", "Số tiền gửi bằng VND"),
                "annual_rate_percent": p("number", "Lãi suất năm theo phần trăm"),
                "term_months": p("integer", "Kỳ hạn gửi bằng tháng"),
                "interest_type": p("string", "Hình thức nhận lãi", enum=["cuối kỳ", "hàng tháng"]),
            },
            ["deposit_vnd", "annual_rate_percent", "term_months"],
        ),
        tool(
            "vi_calculate_personal_income_tax",
            "Ước tính thuế thu nhập cá nhân hàng tháng.",
            "finance",
            "test_unseen",
            {
                "gross_income_vnd": p("integer", "Thu nhập gộp hàng tháng bằng VND"),
                "dependents": p("integer", "Số người phụ thuộc"),
                "insurance_vnd": p("integer", "Khoản bảo hiểm được khấu trừ bằng VND"),
            },
            ["gross_income_vnd", "dependents"],
        ),
        tool(
            "vi_lookup_admission_scores",
            "Tra cứu điểm chuẩn tuyển sinh đại học.",
            "education",
            "seen",
            {
                "university": p("string", "Tên trường đại học"),
                "major": p("string", "Tên ngành học"),
                "year": p("integer", "Năm tuyển sinh"),
            },
            ["university", "year"],
        ),
        tool(
            "vi_search_tutors",
            "Tìm gia sư theo môn học và địa điểm.",
            "education",
            "seen",
            {
                "subject": p("string", "Môn học cần tìm gia sư"),
                "location": p("string", "Địa điểm học"),
                "grade": p("integer", "Lớp của học sinh"),
                "online": p("boolean", "Có học trực tuyến hay không"),
            },
            ["subject", "location"],
        ),
        tool(
            "vi_lookup_exam_schedule",
            "Tra cứu lịch thi tại Việt Nam.",
            "education",
            "dev_unseen",
            {
                "exam": p("string", "Tên kỳ thi", enum=["THPT", "ĐGNL", "IELTS"]),
                "location": p("string", "Địa điểm thi"),
                "year": p("integer", "Năm tổ chức"),
            },
            ["exam", "year"],
        ),
        tool(
            "vi_search_scholarships",
            "Tìm học bổng phù hợp với người học.",
            "education",
            "test_unseen",
            {
                "study_level": p("string", "Bậc học", enum=["đại học", "thạc sĩ", "tiến sĩ"]),
                "field": p("string", "Lĩnh vực học tập"),
                "destination_country": p("string", "Quốc gia du học"),
            },
            ["study_level", "field"],
        ),
        tool(
            "vi_search_medical_facilities",
            "Tìm cơ sở y tế theo chuyên khoa và địa điểm.",
            "health",
            "seen",
            {
                "specialty": p("string", "Chuyên khoa cần khám"),
                "location": p("string", "Địa điểm cần tìm"),
                "accepts_bhyt_only": p("boolean", "Có chỉ lấy cơ sở tiếp nhận BHYT hay không"),
            },
            ["specialty", "location"],
        ),
        tool(
            "vi_book_medical_appointment",
            "Đặt lịch khám tại cơ sở y tế.",
            "health",
            "seen",
            {
                "facility": p("string", "Tên cơ sở y tế"),
                "specialty": p("string", "Chuyên khoa"),
                "appointment_date": p("string", "Ngày khám", fmt="date"),
                "time_period": p("string", "Buổi khám", enum=["sáng", "chiều"]),
            },
            ["facility", "specialty", "appointment_date"],
        ),
        tool(
            "vi_lookup_medicine_information",
            "Tra cứu thông tin sử dụng thuốc phổ biến.",
            "health",
            "dev_unseen",
            {
                "medicine": p("string", "Tên thuốc"),
                "information_type": p(
                    "string",
                    "Loại thông tin",
                    enum=["công dụng", "cách dùng", "tác dụng phụ", "chống chỉ định"],
                ),
            },
            ["medicine"],
        ),
        tool(
            "vi_search_pharmacies",
            "Tìm nhà thuốc theo địa điểm và thời gian mở cửa.",
            "health",
            "test_unseen",
            {
                "location": p("string", "Địa điểm cần tìm"),
                "open_now_only": p("boolean", "Có chỉ lấy nhà thuốc đang mở cửa hay không"),
                "medicine": p("string", "Tên thuốc cần tìm"),
            },
            ["location"],
        ),
        tool(
            "vi_search_products",
            "Tìm sản phẩm trên sàn thương mại điện tử.",
            "ecommerce",
            "seen",
            {
                "product": p("string", "Sản phẩm cần tìm"),
                "platform": p(
                    "string", "Sàn thương mại điện tử", enum=["Shopee", "Lazada", "Tiki", "Sendo"]
                ),
                "max_price_vnd": p("integer", "Giá tối đa bằng VND"),
                "official_store_only": p("boolean", "Có chỉ lấy gian hàng chính hãng hay không"),
            },
            ["product", "platform"],
        ),
        tool(
            "vi_track_shipment",
            "Tra cứu trạng thái giao hàng theo mã vận đơn.",
            "ecommerce",
            "seen",
            {
                "tracking_code": p("string", "Mã vận đơn", identifier=True),
                "carrier": p(
                    "string",
                    "Đơn vị vận chuyển",
                    enum=["GHN", "GHTK", "Viettel Post", "VNPost", "J&T Express"],
                ),
            },
            ["tracking_code"],
        ),
        tool(
            "vi_compare_prices",
            "So sánh giá một sản phẩm giữa các sàn.",
            "ecommerce",
            "dev_unseen",
            {
                "product": p("string", "Sản phẩm cần so sánh"),
                "condition": p("string", "Tình trạng sản phẩm", enum=["mới", "đã qua sử dụng"]),
                "include_shipping": p("boolean", "Có tính phí vận chuyển hay không"),
            },
            ["product"],
        ),
        tool(
            "vi_search_discount_codes",
            "Tìm mã giảm giá cho sàn hoặc sản phẩm.",
            "ecommerce",
            "test_unseen",
            {
                "platform": p(
                    "string", "Sàn thương mại điện tử", enum=["Shopee", "Lazada", "Tiki", "Sendo"]
                ),
                "category": p("string", "Danh mục sản phẩm"),
                "minimum_discount_percent": p("integer", "Mức giảm tối thiểu theo phần trăm"),
            },
            ["platform"],
        ),
        tool(
            "vi_estimate_ride_fare",
            "Ước tính cước chuyến xe công nghệ.",
            "transport",
            "seen",
            {
                "pickup": p("string", "Điểm đón"),
                "destination": p("string", "Điểm đến"),
                "provider": p("string", "Ứng dụng gọi xe", enum=["Grab", "be", "Xanh SM"]),
                "vehicle_type": p("string", "Loại phương tiện", enum=["xe máy", "ô tô"]),
            },
            ["pickup", "destination"],
        ),
        tool(
            "vi_search_public_transport",
            "Tìm tuyến giao thông công cộng phù hợp.",
            "transport",
            "seen",
            {
                "origin": p("string", "Điểm đi"),
                "destination": p("string", "Điểm đến"),
                "mode": p("string", "Loại phương tiện", enum=["xe buýt", "metro", "kết hợp"]),
                "departure_time": p("string", "Giờ khởi hành", fmt="time"),
            },
            ["origin", "destination"],
        ),
        tool(
            "vi_lookup_traffic_fines",
            "Tra cứu phạt nguội theo biển số xe.",
            "transport",
            "dev_unseen",
            {
                "license_plate": p("string", "Biển số xe", identifier=True),
                "vehicle_type": p("string", "Loại phương tiện", enum=["ô tô", "xe máy"]),
            },
            ["license_plate", "vehicle_type"],
        ),
        tool(
            "vi_calculate_route_distance",
            "Tính khoảng cách và thời gian dự kiến giữa hai địa điểm.",
            "transport",
            "test_unseen",
            {
                "origin": p("string", "Điểm đi"),
                "destination": p("string", "Điểm đến"),
                "travel_mode": p(
                    "string", "Phương thức di chuyển", enum=["ô tô", "xe máy", "đi bộ"]
                ),
            },
            ["origin", "destination"],
        ),
        tool(
            "vi_search_admin_procedures",
            "Tra cứu thủ tục hành chính và hồ sơ cần chuẩn bị.",
            "public_services",
            "seen",
            {
                "procedure": p("string", "Tên thủ tục hành chính"),
                "location": p("string", "Tỉnh hoặc thành phố thực hiện"),
                "online_only": p("boolean", "Có chỉ lấy thủ tục thực hiện trực tuyến hay không"),
            },
            ["procedure", "location"],
        ),
        tool(
            "vi_lookup_application_status",
            "Tra cứu tình trạng xử lý hồ sơ hành chính.",
            "public_services",
            "seen",
            {
                "application_code": p("string", "Mã hồ sơ", identifier=True),
                "procedure": p("string", "Loại thủ tục"),
            },
            ["application_code"],
        ),
        tool(
            "vi_lookup_land_price_reference",
            "Tra cứu giá đất tham khảo theo khu vực.",
            "public_services",
            "dev_unseen",
            {
                "location": p("string", "Khu vực cần tra cứu"),
                "land_type": p(
                    "string", "Loại đất", enum=["đất ở", "đất nông nghiệp", "đất thương mại"]
                ),
                "year": p("integer", "Năm áp dụng"),
            },
            ["location", "land_type", "year"],
        ),
        tool(
            "vi_calculate_registration_fee",
            "Ước tính lệ phí trước bạ cho tài sản.",
            "public_services",
            "test_unseen",
            {
                "asset_type": p("string", "Loại tài sản", enum=["ô tô", "xe máy", "nhà đất"]),
                "asset_value_vnd": p("integer", "Giá trị tài sản bằng VND"),
                "location": p("string", "Nơi đăng ký"),
            },
            ["asset_type", "asset_value_vnd", "location"],
        ),
        tool(
            "vi_convert_lunar_date",
            "Chuyển đổi giữa ngày dương lịch và âm lịch.",
            "entertainment",
            "seen",
            {
                "date": p("string", "Ngày cần chuyển đổi", fmt="date"),
                "direction": p(
                    "string", "Chiều chuyển đổi", enum=["dương sang âm", "âm sang dương"]
                ),
            },
            ["date", "direction"],
        ),
        tool(
            "vi_search_events",
            "Tìm sự kiện giải trí theo địa điểm và ngày.",
            "entertainment",
            "seen",
            {
                "location": p("string", "Địa điểm tổ chức"),
                "event_type": p("string", "Loại sự kiện"),
                "date": p("string", "Ngày diễn ra", fmt="date"),
                "max_price_vnd": p("integer", "Giá vé tối đa bằng VND"),
            },
            ["location", "date"],
        ),
        tool(
            "vi_search_karaoke_songs",
            "Tìm bài hát karaoke và mã bài hát.",
            "entertainment",
            "dev_unseen",
            {
                "title": p("string", "Tên bài hát"),
                "singer": p("string", "Tên ca sĩ"),
                "karaoke_system": p(
                    "string", "Hệ thống karaoke", enum=["Arirang", "California", "VietKTV"]
                ),
            },
            ["title"],
        ),
        tool(
            "vi_lookup_zodiac_information",
            "Tra cứu thông tin tham khảo về con giáp theo năm sinh.",
            "entertainment",
            "test_unseen",
            {
                "birth_year": p("integer", "Năm sinh"),
                "zodiac": p("string", "Con giáp", enum=ZODIACS),
                "topic": p("string", "Chủ đề", enum=["tổng quan", "công việc", "tình cảm"]),
            },
            ["birth_year"],
        ),
        tool(
            "vi_get_weather_forecast",
            "Lấy dự báo thời tiết cho địa điểm tại Việt Nam.",
            "agriculture",
            "seen",
            {
                "location": p("string", "Tỉnh hoặc thành phố"),
                "date": p("string", "Ngày cần dự báo", fmt="date"),
                "include_rain_probability": p("boolean", "Có bao gồm xác suất mưa hay không"),
            },
            ["location", "date"],
        ),
        tool(
            "vi_recommend_crop_calendar",
            "Gợi ý lịch mùa vụ theo cây trồng và khu vực.",
            "agriculture",
            "seen",
            {
                "crop": p("string", "Loại cây trồng"),
                "location": p("string", "Khu vực canh tác"),
                "season": p("string", "Mùa vụ", enum=["đông xuân", "hè thu", "thu đông"]),
            },
            ["crop", "location"],
        ),
        tool(
            "vi_lookup_agricultural_prices",
            "Tra cứu giá nông sản tham khảo.",
            "agriculture",
            "dev_unseen",
            {
                "product": p("string", "Nông sản cần tra giá"),
                "location": p("string", "Khu vực thị trường"),
                "date": p("string", "Ngày cần tra cứu", fmt="date"),
                "unit": p("string", "Đơn vị tính", enum=["kg", "tấn"]),
            },
            ["product", "location"],
        ),
        tool(
            "vi_search_agricultural_suppliers",
            "Tìm nhà cung cấp vật tư nông nghiệp.",
            "agriculture",
            "test_unseen",
            {
                "product": p("string", "Vật tư cần tìm"),
                "location": p("string", "Khu vực cần tìm"),
                "certified_only": p("boolean", "Có chỉ lấy nhà cung cấp có chứng nhận hay không"),
            },
            ["product", "location"],
        ),
    ]
    numeric_extensions = {
        "vi_search_restaurants": ("min_rating", p("number", "Điểm đánh giá tối thiểu")),
        "vi_order_food": ("max_delivery_minutes", p("integer", "Thời gian giao tối đa bằng phút")),
        "vi_search_recipes": ("servings", p("integer", "Số khẩu phần")),
        "vi_estimate_meal_cost": ("drinks_per_person", p("integer", "Số đồ uống mỗi người")),
        "vi_search_domestic_flights": ("passengers", p("integer", "Số hành khách")),
        "vi_search_accommodations": ("guests", p("integer", "Số khách lưu trú")),
        "vi_search_intercity_buses": ("passengers", p("integer", "Số hành khách")),
        "vi_search_attractions": ("max_distance_km", p("number", "Khoảng cách tối đa bằng km")),
        "vi_convert_currency": ("fee_percent", p("number", "Phí quy đổi theo phần trăm")),
        "vi_calculate_loan_payment": ("down_payment_vnd", p("integer", "Khoản trả trước bằng VND")),
        "vi_calculate_savings_interest": (
            "monthly_contribution_vnd",
            p("integer", "Khoản gửi thêm hàng tháng bằng VND"),
        ),
        "vi_calculate_personal_income_tax": (
            "charity_vnd",
            p("integer", "Khoản từ thiện được khấu trừ bằng VND"),
        ),
        "vi_lookup_admission_scores": (
            "minimum_score",
            p("number", "Mức điểm tối thiểu cần đối chiếu"),
        ),
        "vi_search_tutors": (
            "max_hourly_rate_vnd",
            p("integer", "Học phí tối đa mỗi giờ bằng VND"),
        ),
        "vi_lookup_exam_schedule": ("session_number", p("integer", "Số thứ tự đợt thi")),
        "vi_search_scholarships": ("minimum_gpa", p("number", "GPA tối thiểu")),
        "vi_search_medical_facilities": ("radius_km", p("number", "Bán kính tìm kiếm bằng km")),
        "vi_book_medical_appointment": ("patient_age", p("integer", "Tuổi người khám")),
        "vi_search_pharmacies": ("radius_km", p("number", "Bán kính tìm kiếm bằng km")),
        "vi_compare_prices": ("max_price_vnd", p("integer", "Giá tối đa bằng VND")),
        "vi_search_discount_codes": (
            "minimum_order_vnd",
            p("integer", "Giá trị đơn tối thiểu bằng VND"),
        ),
        "vi_estimate_ride_fare": ("passengers", p("integer", "Số hành khách")),
        "vi_search_public_transport": ("max_transfers", p("integer", "Số lần chuyển tuyến tối đa")),
        "vi_lookup_traffic_fines": ("violation_year", p("integer", "Năm xảy ra vi phạm")),
        "vi_calculate_route_distance": (
            "average_speed_kmh",
            p("number", "Tốc độ trung bình dự kiến bằng km/h"),
        ),
        "vi_search_admin_procedures": (
            "max_processing_days",
            p("integer", "Thời gian xử lý tối đa bằng ngày"),
        ),
        "vi_lookup_application_status": ("submission_year", p("integer", "Năm nộp hồ sơ")),
        "vi_lookup_land_price_reference": ("area_m2", p("number", "Diện tích bằng mét vuông")),
        "vi_convert_lunar_date": ("timezone_offset", p("number", "Múi giờ UTC")),
        "vi_search_events": ("tickets", p("integer", "Số vé cần tìm")),
        "vi_search_karaoke_songs": ("release_year", p("integer", "Năm phát hành")),
        "vi_lookup_zodiac_information": ("reference_year", p("integer", "Năm tham khảo")),
        "vi_get_weather_forecast": ("forecast_days", p("integer", "Số ngày dự báo")),
        "vi_recommend_crop_calendar": (
            "area_hectares",
            p("number", "Diện tích canh tác bằng hecta"),
        ),
        "vi_lookup_agricultural_prices": ("quantity", p("number", "Số lượng cần quy đổi")),
        "vi_search_agricultural_suppliers": (
            "max_distance_km",
            p("number", "Khoảng cách tối đa bằng km"),
        ),
    }
    boolean_extensions = {
        "vi_lookup_admission_scores": (
            "public_university_only",
            p("boolean", "Chỉ tra trường công lập"),
        ),
        "vi_lookup_exam_schedule": ("weekend_only", p("boolean", "Chỉ tìm lịch cuối tuần")),
        "vi_search_scholarships": ("full_funding_only", p("boolean", "Chỉ tìm học bổng toàn phần")),
        "vi_lookup_medicine_information": (
            "include_prescription_requirements",
            p("boolean", "Có kèm thông tin về yêu cầu đơn thuốc hay không"),
        ),
        "vi_track_shipment": ("include_history", p("boolean", "Có lấy lịch sử vận chuyển")),
        "vi_lookup_application_status": (
            "include_history",
            p("boolean", "Có lấy lịch sử xử lý hồ sơ"),
        ),
        "vi_lookup_land_price_reference": (
            "include_coefficient",
            p("boolean", "Có kèm hệ số điều chỉnh"),
        ),
        "vi_calculate_registration_fee": (
            "first_registration",
            p("boolean", "Có phải đăng ký lần đầu"),
        ),
        "vi_convert_lunar_date": ("include_can_chi", p("boolean", "Có kèm thông tin can chi")),
        "vi_search_karaoke_songs": ("duet_available", p("boolean", "Có phiên bản song ca")),
        "vi_lookup_zodiac_information": ("include_lunar_age", p("boolean", "Có tính tuổi âm")),
        "vi_recommend_crop_calendar": (
            "organic_method",
            p("boolean", "Có ưu tiên canh tác hữu cơ"),
        ),
    }
    numeric_extensions = dict(list(numeric_extensions.items())[:12])
    boolean_extensions = dict(list(boolean_extensions.items())[:5])
    by_name = {item["name"]: item for item in definitions}
    for name, (key, schema) in numeric_extensions.items():
        by_name[name]["parameters"]["properties"][key] = schema
    for name, (key, schema) in boolean_extensions.items():
        by_name[name]["parameters"]["properties"][key] = schema
    return definitions


@dataclass(frozen=True)
class Config:
    output_dir: Path
    dataset_version: str
    as_of: str
    reference_datetime: str
    seed: int
    candidate_count: int
    splits: dict[str, dict[str, Any]]
    multi_call_ratio: float
    negative_distribution: dict[str, float]
    query_style_distribution: dict[str, float]
    max_distribution_deviation: float

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            output_dir=Path(raw["output_dir"]),
            dataset_version=str(raw["dataset_version"]),
            as_of=str(raw["as_of"]),
            reference_datetime=str(raw["reference_datetime"]),
            seed=int(raw["seed"]),
            candidate_count=int(raw["candidate_count"]),
            splits=dict(raw["splits"]),
            multi_call_ratio=float(raw["multi_call_ratio"]),
            negative_distribution={k: float(v) for k, v in raw["negative_distribution"].items()},
            query_style_distribution={
                k: float(v) for k, v in raw["query_style_distribution"].items()
            },
            max_distribution_deviation=float(raw["quality"]["max_distribution_deviation"]),
        )


def largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value / sum(weights.values()) for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    for key in sorted(weights, key=lambda item: (raw[item] - result[item], item), reverse=True)[
        : total - sum(result.values())
    ]:
        result[key] += 1
    return result


def balanced(total: int, values: list[str]) -> list[str]:
    return [values[index % len(values)] for index in range(total)]


def surface_location(value: str, index: int) -> str:
    choices = LOCATION_SURFACES.get(value, [value])
    return choices[index % len(choices)]


def money_surface(value: int, index: int) -> str:
    if value >= 1_000_000:
        millions = value / 1_000_000
        return f"{millions:g} triệu".replace(".", ",")
    if value % 1_000 == 0 and index % 3 == 1:
        return f"{value // 1_000}k"
    return f"{value:,} đồng".replace(",", ".")


def number_surface(value: int | float) -> str:
    return f"{value:g}".replace(".", ",")


def date_value(index: int) -> tuple[str, str]:
    value = date(2026, 8, 13) + timedelta(days=index % 365)
    surfaces = [
        value.strftime("%d/%m/%Y"),
        value.strftime("%d-%m-%Y"),
        f"ngày {value.day} tháng {value.month} năm {value.year}",
    ]
    return value.isoformat(), surfaces[index % len(surfaces)]


def values_for_tool(name: str, index: int) -> dict[str, tuple[Any, str]]:
    province = PROVINCES[(index * 7) % len(PROVINCES)]
    other = PROVINCES[(index * 7 + 11) % len(PROVINCES)]
    if other == province:
        other = PROVINCES[(index + 1) % len(PROVINCES)]
    location = surface_location(province, index)
    other_surface = surface_location(other, index + 1)
    date, date_surface = date_value(index)
    money = 50_000 + (index % 40) * 50_000
    common: dict[str, dict[str, tuple[Any, str]]] = {
        "vi_search_restaurants": {
            "location": (province, location),
            "dish": (FOODS[index % len(FOODS)], FOODS[index % len(FOODS)]),
            "max_price_vnd": (money, money_surface(money, index)),
        },
        "vi_order_food": {
            "dish": (FOODS[index % len(FOODS)], FOODS[index % len(FOODS)]),
            "quantity": (index % 4 + 1, str(index % 4 + 1)),
            "delivery_location": (
                DISTRICTS[index % len(DISTRICTS)],
                DISTRICTS[index % len(DISTRICTS)],
            ),
            "spicy": (index % 2 == 0, "có cay" if index % 2 == 0 else "không cay"),
        },
        "vi_search_recipes": {
            "dish": (FOODS[index % len(FOODS)], FOODS[index % len(FOODS)]),
            "difficulty": (
                ["dễ", "trung bình", "khó"][index % 3],
                ["dễ", "trung bình", "khó"][index % 3],
            ),
            "max_minutes": ((index % 4 + 2) * 15, f"{(index % 4 + 2) * 15} phút"),
        },
        "vi_estimate_meal_cost": {
            "people": (index % 6 + 2, str(index % 6 + 2)),
            "meal_type": (
                ["bình dân", "trung cấp", "cao cấp"][index % 3],
                ["bình dân", "trung cấp", "cao cấp"][index % 3],
            ),
            "location": (province, location),
        },
        "vi_search_domestic_flights": {
            "origin": (province, location),
            "destination": (other, other_surface),
            "departure_date": (date, date_surface),
            "airline": (AIRLINES[index % len(AIRLINES)], AIRLINES[index % len(AIRLINES)]),
        },
        "vi_search_accommodations": {
            "location": (province, location),
            "check_in_date": (date, date_surface),
            "nights": (index % 4 + 1, f"{index % 4 + 1} đêm"),
            "max_price_vnd": (money, money_surface(money, index)),
        },
        "vi_search_intercity_buses": {
            "origin": (province, location),
            "destination": (other, other_surface),
            "departure_date": (date, date_surface),
            "seat_type": (
                ["ghế ngồi", "giường nằm", "limousine"][index % 3],
                ["ghế ngồi", "giường nằm", "limousine"][index % 3],
            ),
        },
        "vi_search_attractions": {
            "location": (province, location),
            "category": (
                ["thiên nhiên", "lịch sử", "văn hóa", "giải trí"][index % 4],
                ["thiên nhiên", "lịch sử", "văn hóa", "giải trí"][index % 4],
            ),
            "family_friendly_only": (
                index % 2 == 0,
                "chỉ lấy nơi phù hợp với gia đình"
                if index % 2 == 0
                else "không giới hạn theo tiêu chí gia đình",
            ),
        },
        "vi_convert_currency": {
            "amount": (
                [100, 500, 1000, 2_000_000][index % 4],
                money_surface(2_000_000, index)
                if index % 4 == 3
                else str([100, 500, 1000, 2_000_000][index % 4]),
            ),
            "from_currency": (
                ["USD", "EUR", "JPY", "VND"][index % 4],
                ["USD", "EUR", "JPY", "VND"][index % 4],
            ),
            "to_currency": (
                ["VND", "USD", "VND", "KRW"][index % 4],
                ["VND", "USD", "VND", "KRW"][index % 4],
            ),
        },
        "vi_calculate_loan_payment": {
            "principal_vnd": (
                [50_000_000, 100_000_000, 500_000_000][index % 3],
                money_surface([50_000_000, 100_000_000, 500_000_000][index % 3], index),
            ),
            "annual_rate_percent": (
                [6.5, 8.2, 10.5][index % 3],
                f"{number_surface([6.5, 8.2, 10.5][index % 3])}%",
            ),
            "term_months": ([12, 24, 60][index % 3], f"{[12, 24, 60][index % 3]} tháng"),
        },
        "vi_calculate_savings_interest": {
            "deposit_vnd": (
                [20_000_000, 100_000_000, 300_000_000][index % 3],
                money_surface([20_000_000, 100_000_000, 300_000_000][index % 3], index),
            ),
            "annual_rate_percent": (
                [4.8, 5.5, 6.0][index % 3],
                f"{number_surface([4.8, 5.5, 6.0][index % 3])}%",
            ),
            "term_months": ([3, 6, 12][index % 3], f"{[3, 6, 12][index % 3]} tháng"),
            "interest_type": (
                ["cuối kỳ", "hàng tháng"][index % 2],
                ["cuối kỳ", "hàng tháng"][index % 2],
            ),
        },
        "vi_calculate_personal_income_tax": {
            "gross_income_vnd": (
                [20_000_000, 35_000_000, 60_000_000][index % 3],
                money_surface([20_000_000, 35_000_000, 60_000_000][index % 3], index),
            ),
            "dependents": (index % 3, str(index % 3)),
            "insurance_vnd": (
                [1_000_000, 2_000_000, 3_000_000][index % 3],
                money_surface([1_000_000, 2_000_000, 3_000_000][index % 3], index),
            ),
        },
        "vi_lookup_admission_scores": {
            "university": (
                UNIVERSITIES[index % len(UNIVERSITIES)],
                UNIVERSITIES[index % len(UNIVERSITIES)],
            ),
            "major": (
                ["Công nghệ thông tin", "Y khoa", "Kinh tế", "Ngôn ngữ Anh"][index % 4],
                ["Công nghệ thông tin", "Y khoa", "Kinh tế", "Ngôn ngữ Anh"][index % 4],
            ),
            "year": ([2024, 2025, 2026][index % 3], str([2024, 2025, 2026][index % 3])),
        },
        "vi_search_tutors": {
            "subject": (SUBJECTS[index % len(SUBJECTS)], SUBJECTS[index % len(SUBJECTS)]),
            "location": (province, location),
            "grade": (index % 12 + 1, f"lớp {index % 12 + 1}"),
            "online": (index % 2 == 0, "học online" if index % 2 == 0 else "học trực tiếp"),
        },
        "vi_lookup_exam_schedule": {
            "exam": (["THPT", "ĐGNL", "IELTS"][index % 3], ["THPT", "ĐGNL", "IELTS"][index % 3]),
            "location": (province, location),
            "year": ([2026, 2027][index % 2], str([2026, 2027][index % 2])),
        },
        "vi_search_scholarships": {
            "study_level": (
                ["đại học", "thạc sĩ", "tiến sĩ"][index % 3],
                ["đại học", "thạc sĩ", "tiến sĩ"][index % 3],
            ),
            "field": (
                ["CNTT", "kinh tế", "y sinh", "môi trường"][index % 4],
                ["CNTT", "kinh tế", "y sinh", "môi trường"][index % 4],
            ),
            "destination_country": (
                ["Nhật Bản", "Hàn Quốc", "Úc", "Đức"][index % 4],
                ["Nhật Bản", "Hàn Quốc", "Úc", "Đức"][index % 4],
            ),
        },
        "vi_search_medical_facilities": {
            "specialty": (
                ["tim mạch", "da liễu", "nhi", "răng hàm mặt"][index % 4],
                ["tim mạch", "da liễu", "nhi", "răng hàm mặt"][index % 4],
            ),
            "location": (province, location),
            "accepts_bhyt_only": (
                index % 2 == 0,
                "chỉ lấy cơ sở nhận BHYT" if index % 2 == 0 else "không giới hạn theo BHYT",
            ),
        },
        "vi_book_medical_appointment": {
            "facility": (HOSPITALS[index % len(HOSPITALS)], HOSPITALS[index % len(HOSPITALS)]),
            "specialty": (
                ["tim mạch", "da liễu", "nhi", "răng hàm mặt"][index % 4],
                ["tim mạch", "da liễu", "nhi", "răng hàm mặt"][index % 4],
            ),
            "appointment_date": (date, date_surface),
            "time_period": (["sáng", "chiều"][index % 2], ["buổi sáng", "buổi chiều"][index % 2]),
        },
        "vi_lookup_medicine_information": {
            "medicine": (
                ["paracetamol", "loratadine", "oresol", "ibuprofen"][index % 4],
                ["paracetamol", "loratadine", "oresol", "ibuprofen"][index % 4],
            ),
            "information_type": (
                ["công dụng", "cách dùng", "tác dụng phụ", "chống chỉ định"][index % 4],
                ["công dụng", "cách dùng", "tác dụng phụ", "chống chỉ định"][index % 4],
            ),
        },
        "vi_search_pharmacies": {
            "location": (province, location),
            "open_now_only": (
                index % 2 == 0,
                "chỉ lấy nhà thuốc đang mở cửa"
                if index % 2 == 0
                else "không lọc theo trạng thái mở cửa",
            ),
            "medicine": (
                ["paracetamol", "oresol", "nước muối sinh lý"][index % 3],
                ["paracetamol", "oresol", "nước muối sinh lý"][index % 3],
            ),
        },
        "vi_search_products": {
            "product": (
                ["tai nghe Bluetooth", "nồi chiên không dầu", "điện thoại", "máy lọc không khí"][
                    index % 4
                ],
                ["tai nghe Bluetooth", "nồi chiên không dầu", "điện thoại", "máy lọc không khí"][
                    index % 4
                ],
            ),
            "platform": (
                MARKETPLACES[index % len(MARKETPLACES)],
                MARKETPLACES[index % len(MARKETPLACES)],
            ),
            "max_price_vnd": (money, money_surface(money, index)),
            "official_store_only": (
                index % 2 == 0,
                "chỉ lấy gian hàng chính hãng"
                if index % 2 == 0
                else "không giới hạn loại gian hàng",
            ),
        },
        "vi_track_shipment": {
            "tracking_code": (
                f"VN{index % 997:03d}A{(index * 17) % 9999:04d}",
                f"VN{index % 997:03d}A{(index * 17) % 9999:04d}",
            ),
            "carrier": (CARRIERS[index % len(CARRIERS)], CARRIERS[index % len(CARRIERS)]),
        },
        "vi_compare_prices": {
            "product": (
                ["tai nghe Bluetooth", "nồi chiên không dầu", "điện thoại", "máy lọc không khí"][
                    index % 4
                ],
                ["tai nghe Bluetooth", "nồi chiên không dầu", "điện thoại", "máy lọc không khí"][
                    index % 4
                ],
            ),
            "condition": (
                ["mới", "đã qua sử dụng"][index % 2],
                ["mới", "đã qua sử dụng"][index % 2],
            ),
            "include_shipping": (
                index % 2 == 0,
                "tính cả phí ship" if index % 2 == 0 else "không tính phí ship",
            ),
        },
        "vi_search_discount_codes": {
            "platform": (
                MARKETPLACES[index % len(MARKETPLACES)],
                MARKETPLACES[index % len(MARKETPLACES)],
            ),
            "category": (
                ["điện tử", "gia dụng", "thời trang", "mẹ và bé"][index % 4],
                ["điện tử", "gia dụng", "thời trang", "mẹ và bé"][index % 4],
            ),
            "minimum_discount_percent": (
                [10, 15, 20, 30][index % 4],
                f"giảm ít nhất {[10, 15, 20, 30][index % 4]}%",
            ),
        },
        "vi_estimate_ride_fare": {
            "pickup": (f"trung tâm {province}", f"trung tâm {location}"),
            "destination": (f"sân bay {province}", f"sân bay ở {location}"),
            "provider": (RIDE_APPS[index % len(RIDE_APPS)], RIDE_APPS[index % len(RIDE_APPS)]),
            "vehicle_type": (["xe máy", "ô tô"][index % 2], ["xe máy", "ô tô"][index % 2]),
        },
        "vi_search_public_transport": {
            "origin": (f"bến xe trung tâm {province}", f"bến xe ở {location}"),
            "destination": (f"ga trung tâm {province}", f"ga trung tâm {location}"),
            "mode": (
                ["xe buýt", "metro", "kết hợp"][index % 3],
                ["xe buýt", "metro", "kết hợp"][index % 3],
            ),
            "departure_time": (
                ["07:30", "09:00", "17:45"][index % 3],
                ["7 giờ 30", "9 giờ", "17 giờ 45"][index % 3],
            ),
        },
        "vi_lookup_traffic_fines": {
            "license_plate": (
                f"{30 + index % 60}A-{10000 + index % 89999}",
                f"{30 + index % 60}A-{10000 + index % 89999}",
            ),
            "vehicle_type": (["ô tô", "xe máy"][index % 2], ["ô tô", "xe máy"][index % 2]),
        },
        "vi_calculate_route_distance": {
            "origin": (province, location),
            "destination": (other, other_surface),
            "travel_mode": (
                ["ô tô", "xe máy", "đi bộ"][index % 3],
                ["ô tô", "xe máy", "đi bộ"][index % 3],
            ),
        },
        "vi_search_admin_procedures": {
            "procedure": (
                ["đăng ký khai sinh", "cấp hộ chiếu", "đổi giấy phép lái xe", "đăng ký tạm trú"][
                    index % 4
                ],
                ["đăng ký khai sinh", "cấp hộ chiếu", "đổi giấy phép lái xe", "đăng ký tạm trú"][
                    index % 4
                ],
            ),
            "location": (province, location),
            "online_only": (
                index % 2 == 0,
                "chỉ lấy thủ tục làm trực tuyến"
                if index % 2 == 0
                else "không giới hạn hình thức thực hiện",
            ),
        },
        "vi_lookup_application_status": {
            "application_code": (f"HS26-{index % 999999:06d}", f"HS26-{index % 999999:06d}"),
            "procedure": (
                ["hộ chiếu", "đăng ký cư trú", "giấy phép lái xe"][index % 3],
                ["hộ chiếu", "đăng ký cư trú", "giấy phép lái xe"][index % 3],
            ),
        },
        "vi_lookup_land_price_reference": {
            "location": (DISTRICTS[index % len(DISTRICTS)], DISTRICTS[index % len(DISTRICTS)]),
            "land_type": (
                ["đất ở", "đất nông nghiệp", "đất thương mại"][index % 3],
                ["đất ở", "đất nông nghiệp", "đất thương mại"][index % 3],
            ),
            "year": ([2025, 2026][index % 2], str([2025, 2026][index % 2])),
        },
        "vi_calculate_registration_fee": {
            "asset_type": (
                ["ô tô", "xe máy", "nhà đất"][index % 3],
                ["ô tô", "xe máy", "nhà đất"][index % 3],
            ),
            "asset_value_vnd": (
                [700_000_000, 50_000_000, 2_000_000_000][index % 3],
                money_surface([700_000_000, 50_000_000, 2_000_000_000][index % 3], index),
            ),
            "location": (province, location),
        },
        "vi_convert_lunar_date": {
            "date": (date, date_surface),
            "direction": (
                ["dương sang âm", "âm sang dương"][index % 2],
                ["dương sang âm", "âm sang dương"][index % 2],
            ),
        },
        "vi_search_events": {
            "location": (province, location),
            "event_type": (EVENTS[index % len(EVENTS)], EVENTS[index % len(EVENTS)]),
            "date": (date, date_surface),
            "max_price_vnd": (money, money_surface(money, index)),
        },
        "vi_search_karaoke_songs": {
            "title": (
                [
                    "Nối vòng tay lớn",
                    "Một vòng Việt Nam",
                    "Có chàng trai viết lên cây",
                    "Đi để trở về",
                ][index % 4],
                [
                    "Nối vòng tay lớn",
                    "Một vòng Việt Nam",
                    "Có chàng trai viết lên cây",
                    "Đi để trở về",
                ][index % 4],
            ),
            "singer": (
                ["Trịnh Công Sơn", "Tùng Dương", "Phan Mạnh Quỳnh", "Soobin"][index % 4],
                ["Trịnh Công Sơn", "Tùng Dương", "Phan Mạnh Quỳnh", "Soobin"][index % 4],
            ),
            "karaoke_system": (
                ["Arirang", "California", "VietKTV"][index % 3],
                ["Arirang", "California", "VietKTV"][index % 3],
            ),
        },
        "vi_lookup_zodiac_information": {
            "birth_year": (1990 + index % 20, str(1990 + index % 20)),
            "zodiac": (
                ZODIACS[(1990 + index % 20 - 4) % len(ZODIACS)],
                ZODIACS[(1990 + index % 20 - 4) % len(ZODIACS)],
            ),
            "topic": (
                ["tổng quan", "công việc", "tình cảm"][index % 3],
                ["tổng quan", "công việc", "tình cảm"][index % 3],
            ),
        },
        "vi_get_weather_forecast": {
            "location": (province, location),
            "date": (date, date_surface),
            "include_rain_probability": (
                index % 2 == 0,
                "có xác suất mưa" if index % 2 == 0 else "không cần xác suất mưa",
            ),
        },
        "vi_recommend_crop_calendar": {
            "crop": (CROPS[index % len(CROPS)], CROPS[index % len(CROPS)]),
            "location": (province, location),
            "season": (
                ["đông xuân", "hè thu", "thu đông"][index % 3],
                ["đông xuân", "hè thu", "thu đông"][index % 3],
            ),
        },
        "vi_lookup_agricultural_prices": {
            "product": (CROPS[index % len(CROPS)], CROPS[index % len(CROPS)]),
            "location": (province, location),
            "date": (date, date_surface),
            "unit": (["kg", "tấn"][index % 2], ["kg", "tấn"][index % 2]),
        },
        "vi_search_agricultural_suppliers": {
            "product": (
                ["phân hữu cơ", "hạt giống", "thuốc sinh học", "hệ thống tưới"][index % 4],
                ["phân hữu cơ", "hạt giống", "thuốc sinh học", "hệ thống tưới"][index % 4],
            ),
            "location": (province, location),
            "certified_only": (
                index % 2 == 0,
                "chỉ lấy nhà cung cấp có chứng nhận"
                if index % 2 == 0
                else "không giới hạn theo chứng nhận",
            ),
        },
    }
    return common[name]


def optional_keys(name: str, schema: dict[str, Any], index: int) -> list[str]:
    required = set(schema["parameters"]["required"])
    optional = [key for key in schema["parameters"]["properties"] if key not in required]
    return [key for offset, key in enumerate(optional) if (index + offset) % 5 < 3]


def extension_value(key: str, schema: dict[str, Any], index: int) -> tuple[Any, str]:
    if schema.get("type") == "boolean":
        boolean_value = index % 2 == 0
        surfaces = {
            "public_university_only": ("chỉ tra trường công lập", "không giới hạn trường công lập"),
            "weekend_only": ("chỉ lấy lịch thi cuối tuần", "không giới hạn ngày thi"),
            "full_funding_only": ("chỉ lấy học bổng toàn phần", "không giới hạn mức tài trợ"),
            "include_prescription_requirements": (
                "có kèm thông tin về yêu cầu đơn thuốc",
                "không kèm thông tin về yêu cầu đơn thuốc",
            ),
            "include_history": ("có lấy lịch sử vận chuyển", "không lấy lịch sử vận chuyển"),
        }
        return boolean_value, surfaces[key][0 if boolean_value else 1]
    if schema.get("type") == "integer":
        integer_values = {
            "max_delivery_minutes": [20, 30, 45, 60],
            "servings": [2, 3, 4, 6],
            "drinks_per_person": [1, 2],
            "passengers": [1, 2, 3, 4, 5],
            "guests": [1, 2, 3, 4, 6],
        }
        integer_value = integer_values.get(key, list(range(1, 9)))[
            index % len(integer_values.get(key, list(range(1, 9))))
        ]
        if key.endswith("_vnd"):
            unit = (
                5_000_000
                if key == "down_payment_vnd"
                else 1_000_000
                if key == "monthly_contribution_vnd"
                else 100_000
            )
            integer_value = (index % 8 + 1) * unit
            return integer_value, money_surface(integer_value, index)
        return integer_value, str(integer_value)
    number_value = round(1.5 + index % 7 * 0.5, 1)
    return number_value, str(number_value).replace(".", ",")


def extension_phrase(key: str, schema: dict[str, Any], surface: str) -> str:
    if schema.get("type") == "boolean":
        return surface
    phrases = {
        "min_rating": f"điểm đánh giá từ {surface} trở lên",
        "max_delivery_minutes": f"giao trong tối đa {surface} phút",
        "servings": f"cho {surface} khẩu phần",
        "drinks_per_person": f"mỗi người dùng {surface} đồ uống",
        "passengers": f"cho {surface} hành khách",
        "guests": f"cho {surface} khách",
        "max_distance_km": f"trong bán kính {surface} km",
        "fee_percent": f"phí quy đổi {surface}%",
        "down_payment_vnd": f"trả trước {surface}",
        "monthly_contribution_vnd": f"gửi thêm {surface} mỗi tháng",
        "charity_vnd": f"khấu trừ từ thiện {surface}",
    }
    return phrases.get(key, f"{schema['description']}: {surface}")


def clause(name: str, values: dict[str, tuple[Any, str]], keys: list[str]) -> str:
    v = {key: values[key][1] for key in keys}
    recipes: dict[str, Callable[[dict[str, str]], str]] = {
        "vi_search_restaurants": lambda x: (
            f"tìm quán {x['dish']} ở {x['location']}"
            + (f", giá dưới {x['max_price_vnd']}" if "max_price_vnd" in x else "")
        ),
        "vi_order_food": lambda x: (
            f"đặt {x['quantity']} phần {x['dish']} giao đến {x['delivery_location']}"
            + (f", {x['spicy']}" if "spicy" in x else "")
        ),
        "vi_search_recipes": lambda x: (
            f"tìm công thức {x['dish']}"
            + (f" mức {x['difficulty']}" if "difficulty" in x else "")
            + (f" nấu trong {x['max_minutes']}" if "max_minutes" in x else "")
        ),
        "vi_estimate_meal_cost": lambda x: (
            f"ước tính chi phí bữa ăn {x['meal_type']} cho {x['people']} người ở {x['location']}"
        ),
        "vi_search_domestic_flights": lambda x: (
            f"tìm chuyến bay từ {x['origin']} đến {x['destination']} vào {x['departure_date']}"
            + (f" của {x['airline']}" if "airline" in x else "")
        ),
        "vi_search_accommodations": lambda x: (
            f"tìm chỗ ở tại {x['location']} nhận phòng {x['check_in_date']} trong {x['nights']}"
            + (f", dưới {x['max_price_vnd']} mỗi đêm" if "max_price_vnd" in x else "")
        ),
        "vi_search_intercity_buses": lambda x: (
            f"tìm xe khách từ {x['origin']} đi {x['destination']} vào {x['departure_date']}"
            + (f", loại {x['seat_type']}" if "seat_type" in x else "")
        ),
        "vi_search_attractions": lambda x: (
            f"tìm điểm tham quan ở {x['location']}"
            + (f" thuộc nhóm {x['category']}" if "category" in x else "")
            + (f", {x['family_friendly_only']}" if "family_friendly_only" in x else "")
        ),
        "vi_convert_currency": lambda x: (
            f"đổi {x['amount']} {x['from_currency']} sang {x['to_currency']}"
        ),
        "vi_calculate_loan_payment": lambda x: (
            f"tính tiền trả khoản vay {x['principal_vnd']}, lãi suất {x['annual_rate_percent']} trong {x['term_months']}"
        ),
        "vi_calculate_savings_interest": lambda x: (
            f"tính lãi khi gửi {x['deposit_vnd']} với lãi suất {x['annual_rate_percent']} trong {x['term_months']}"
            + (f", nhận lãi {x['interest_type']}" if "interest_type" in x else "")
        ),
        "vi_calculate_personal_income_tax": lambda x: (
            f"ước tính thuế TNCN cho thu nhập {x['gross_income_vnd']} và {x['dependents']} người phụ thuộc"
            + (f", bảo hiểm {x['insurance_vnd']}" if "insurance_vnd" in x else "")
        ),
        "vi_lookup_admission_scores": lambda x: (
            f"tra điểm chuẩn {x.get('major', 'các ngành')} của {x['university']} năm {x['year']}"
        ),
        "vi_search_tutors": lambda x: (
            f"tìm gia sư {x['subject']} ở {x['location']}"
            + (f" cho {x['grade']}" if "grade" in x else "")
            + (f", {x['online']}" if "online" in x else "")
        ),
        "vi_lookup_exam_schedule": lambda x: (
            f"tra lịch thi {x['exam']} năm {x['year']}"
            + (f" tại {x['location']}" if "location" in x else "")
        ),
        "vi_search_scholarships": lambda x: (
            f"tìm học bổng {x['study_level']} ngành {x['field']}"
            + (f" tại {x['destination_country']}" if "destination_country" in x else "")
        ),
        "vi_search_medical_facilities": lambda x: (
            f"tìm cơ sở khám {x['specialty']} ở {x['location']}"
            + (f", {x['accepts_bhyt_only']}" if "accepts_bhyt_only" in x else "")
        ),
        "vi_book_medical_appointment": lambda x: (
            f"đặt lịch khám {x['specialty']} tại {x['facility']} vào {x['appointment_date']}"
            + (f" {x['time_period']}" if "time_period" in x else "")
        ),
        "vi_lookup_medicine_information": lambda x: (
            f"tra {x.get('information_type', 'thông tin')} của thuốc {x['medicine']}"
        ),
        "vi_search_pharmacies": lambda x: (
            f"tìm nhà thuốc ở {x['location']}"
            + (f", {x['open_now_only']}" if "open_now_only" in x else "")
            + (f", có {x['medicine']}" if "medicine" in x else "")
        ),
        "vi_search_products": lambda x: (
            f"tìm {x['product']} trên {x['platform']}"
            + (f" dưới {x['max_price_vnd']}" if "max_price_vnd" in x else "")
            + (f", {x['official_store_only']}" if "official_store_only" in x else "")
        ),
        "vi_track_shipment": lambda x: (
            f"tra đơn hàng mã {x['tracking_code']}"
            + (f" của {x['carrier']}" if "carrier" in x else "")
        ),
        "vi_compare_prices": lambda x: (
            f"so sánh giá {x['product']}"
            + (f" loại {x['condition']}" if "condition" in x else "")
            + (f", {x['include_shipping']}" if "include_shipping" in x else "")
        ),
        "vi_search_discount_codes": lambda x: (
            f"tìm mã giảm giá trên {x['platform']}"
            + (f" cho danh mục {x['category']}" if "category" in x else "")
            + (f", {x['minimum_discount_percent']}" if "minimum_discount_percent" in x else "")
        ),
        "vi_estimate_ride_fare": lambda x: (
            f"ước tính cước {x.get('vehicle_type', 'chuyến xe')}"
            + (f" {x['provider']}" if "provider" in x else "")
            + f" từ {x['pickup']} đến {x['destination']}"
        ),
        "vi_search_public_transport": lambda x: (
            f"tìm đường từ {x['origin']} đến {x['destination']} bằng {x.get('mode', 'phương tiện công cộng')}"
            + (f" lúc {x['departure_time']}" if "departure_time" in x else "")
        ),
        "vi_lookup_traffic_fines": lambda x: (
            f"tra phạt nguội {x['vehicle_type']} biển số {x['license_plate']}"
        ),
        "vi_calculate_route_distance": lambda x: (
            f"tính khoảng cách từ {x['origin']} đến {x['destination']}"
            + (f" khi đi bằng {x['travel_mode']}" if "travel_mode" in x else "")
        ),
        "vi_search_admin_procedures": lambda x: (
            f"tra thủ tục {x['procedure']} tại {x['location']}"
            + (f", {x['online_only']}" if "online_only" in x else "")
        ),
        "vi_lookup_application_status": lambda x: (
            f"tra trạng thái hồ sơ {x['application_code']}"
            + (f" về {x['procedure']}" if "procedure" in x else "")
        ),
        "vi_lookup_land_price_reference": lambda x: (
            f"tra giá {x['land_type']} tại {x['location']} năm {x['year']}"
        ),
        "vi_calculate_registration_fee": lambda x: (
            f"tính lệ phí trước bạ cho {x['asset_type']} trị giá {x['asset_value_vnd']} tại {x['location']}"
        ),
        "vi_convert_lunar_date": lambda x: f"đổi ngày {x['date']} theo chiều {x['direction']}",
        "vi_search_events": lambda x: (
            f"tìm {x.get('event_type', 'sự kiện')} ở {x['location']} vào {x['date']}"
            + (f", vé dưới {x['max_price_vnd']}" if "max_price_vnd" in x else "")
        ),
        "vi_search_karaoke_songs": lambda x: (
            f"tìm mã karaoke bài {x['title']}"
            + (f" của {x['singer']}" if "singer" in x else "")
            + (f" trên {x['karaoke_system']}" if "karaoke_system" in x else "")
        ),
        "vi_lookup_zodiac_information": lambda x: (
            f"tra thông tin {x.get('topic', 'tổng quan')} cho người sinh năm {x['birth_year']}"
            + (f" tuổi {x['zodiac']}" if "zodiac" in x else "")
        ),
        "vi_get_weather_forecast": lambda x: (
            f"xem thời tiết {x['location']} vào {x['date']}"
            + (f", {x['include_rain_probability']}" if "include_rain_probability" in x else "")
        ),
        "vi_recommend_crop_calendar": lambda x: (
            f"gợi ý lịch trồng {x['crop']}"
            + (f" vụ {x['season']}" if "season" in x else " phù hợp")
            + f" tại {x['location']}"
        ),
        "vi_lookup_agricultural_prices": lambda x: (
            f"tra giá {x['product']} tại {x['location']}"
            + (f" vào {x['date']}" if "date" in x else "")
            + (f" theo {x['unit']}" if "unit" in x else "")
        ),
        "vi_search_agricultural_suppliers": lambda x: (
            f"tìm nhà cung cấp {x['product']} ở {x['location']}"
            + (f", {x['certified_only']}" if "certified_only" in x else "")
        ),
    }
    return recipes[name](v)


STYLE_WRAPPERS: dict[str, list[Callable[[str], str]]] = {
    "neutral": [
        lambda x: x[0].upper() + x[1:] + ".",
        lambda x: "Tôi muốn " + x + ".",
        lambda x: "Hãy " + x + ".",
    ],
    "polite": [
        lambda x: "Bạn vui lòng giúp tôi " + x + " nhé.",
        lambda x: "Bạn có thể giúp tôi " + x + " được không?",
        lambda x: "Phiền bạn " + x + " giúp tôi.",
    ],
    "conversational": [
        lambda x: "Cho mình " + x + " với.",
        lambda x: "Mình đang cần " + x + ", giúp mình nhé.",
        lambda x: "Này, giúp tôi " + x + " được không?",
    ],
    "noisy": [
        lambda x: "Giúp mk " + x + " nha.",
        lambda x: "Cho tui " + x + " vs.",
        lambda x: "Cần " + x + " gấp ạ.",
    ],
    "code_mixed": [
        lambda x: "Bạn check giúp tôi để " + x + ".",
        lambda x: "Please giúp mình " + x + " nhé.",
        lambda x: "Mình cần bạn support " + x + ".",
    ],
}


NEGATIVE_TEMPLATES = {
    "hard_near_miss": [
        "Tôi đang phân vân liệu {topic} có thật sự đáng tin không, chưa cần tra cứu hay thực hiện gì cả.",
        "Theo bạn, mọi người thường nghĩ gì về {topic}? Tôi chỉ muốn trò chuyện thôi.",
        "Hãy giải thích khái niệm {topic}; đừng tìm kiếm, đặt chỗ hay tính toán giúp tôi.",
        "Tôi vừa nghe nhắc đến {topic}, nhưng hiện tại tôi không yêu cầu thao tác nào.",
    ],
    "non_action": [
        "Hôm nay câu chuyện về {topic} nghe thú vị thật đấy!",
        "Tôi khá thích đọc tin về {topic} vào cuối tuần.",
        "Nhắc đến {topic} làm tôi nhớ chuyến đi năm ngoái.",
        "{topic_cap} là chủ đề cả nhà tôi hay bàn trong bữa cơm.",
    ],
    "out_of_scope": [
        "Hãy viết một bài thơ lục bát về {topic}.",
        "Giải thích lịch sử hình thành của {topic} bằng ba đoạn văn.",
        "Soạn một câu chuyện ngắn hài hước có nhắc đến {topic}.",
        "Dịch cụm từ {topic} sang tiếng Pháp và giải thích ngữ pháp.",
    ],
    "negated_hypothetical": [
        "Nếu sau này cần {topic} thì quy trình có phức tạp không? Hiện giờ đừng thực hiện nhé.",
        "Tôi không muốn bạn tìm hay đặt gì liên quan đến {topic}; chỉ cho biết đây là lĩnh vực gì.",
        "Giả sử một người muốn xử lý {topic}, họ nên cân nhắc điều gì? Không thao tác giúp tôi.",
        "Đừng gọi công cụ nào; tôi chỉ đang tưởng tượng tình huống về {topic} thôi.",
    ],
}

DIVERSITY_CONTEXTS = [
    "Tôi cần thông tin này để bàn lại với gia đình.",
    "Kết quả sẽ giúp tôi chốt kế hoạch cá nhân.",
    "Tôi muốn kiểm tra kỹ trước khi quyết định.",
    "Việc này đang nằm trong danh sách cần chuẩn bị của tôi.",
    "Mình cần một phương án rõ ràng để trao đổi với bạn bè.",
    "Tôi sẽ dùng kết quả làm căn cứ so sánh các lựa chọn.",
    "Thông tin này phục vụ một kế hoạch cuối tuần của tôi.",
    "Mình đang gom dữ liệu để sắp xếp công việc cho hợp lý.",
    "Tôi cần kết quả gọn để gửi lại cho người thân.",
    "Đây là bước cuối trước khi tôi thống nhất phương án.",
    "Mình muốn xem lựa chọn nào thực tế hơn trong trường hợp này.",
    "Tôi đang hỗ trợ một người bạn chuẩn bị việc này.",
    "Kết quả sẽ được dùng trong buổi trao đổi sắp tới.",
    "Tôi muốn lưu thông tin này vào kế hoạch riêng.",
    "Mình cần đối chiếu với một phương án đã có.",
    "Tôi đang cân nhắc nên muốn dữ liệu thật rõ ràng.",
    "Thông tin này sẽ giúp tôi chủ động sắp xếp lịch.",
    "Mình cần hoàn tất phần chuẩn bị còn thiếu.",
    "Tôi muốn tham khảo trước khi trao đổi với cả nhóm.",
    "Kết quả sẽ giúp tôi rà lại ngân sách và thời gian.",
    "Mình đang chuẩn bị một phương án dự phòng.",
    "Tôi cần dữ liệu để cập nhật vào ghi chú cá nhân.",
    "Đây là nội dung tôi muốn xác nhận trước khi tiếp tục.",
    "Mình muốn xử lý dứt điểm phần việc này trong hôm nay.",
    "Tôi cần một kết quả cụ thể để tránh phải tra lại.",
    "Thông tin này sẽ được dùng để hoàn thiện lịch trình.",
    "Mình đang soạn danh sách lựa chọn cho một người thân.",
    "Tôi muốn có căn cứ trước khi thay đổi kế hoạch.",
    "Kết quả sẽ giúp tôi kiểm tra tính khả thi của phương án.",
    "Mình cần tổng hợp phần này cùng các ghi chú khác.",
    "Tôi đang chuẩn bị nội dung cho một cuộc trao đổi ngắn.",
    "Thông tin này giúp tôi tránh bỏ sót một việc quan trọng.",
]

DIVERSITY_DETAILS = [
    "Ưu tiên kết quả ngắn gọn và dễ đối chiếu.",
    "Tôi sẽ tự kiểm tra lại thông tin sau.",
    "Không cần trình bày thêm phần giải thích dài.",
    "Mình muốn lưu lại kết quả để dùng khi cần.",
    "Hãy giữ đúng các điều kiện đã nêu.",
    "Tôi chỉ cần phương án phù hợp nhất.",
    "Mình muốn xem kết quả trước rồi mới quyết định.",
    "Thông tin rõ ràng sẽ giúp tôi xử lý nhanh hơn.",
    "Tôi đang ưu tiên phương án dễ thực hiện.",
    "Mình sẽ đối chiếu kết quả với ghi chú hiện có.",
    "Tôi muốn tránh phải thay đổi kế hoạch vào phút chót.",
    "Mình cần câu trả lời đủ cụ thể để sử dụng ngay.",
    "Tôi sẽ cân nhắc thêm sau khi có kết quả.",
    "Mình muốn hoàn tất việc chuẩn bị theo từng bước.",
    "Tôi cần thông tin có thể kiểm tra lại dễ dàng.",
    "Mình đang ưu tiên tính thực tế của lựa chọn.",
]

NEGATIVE_DIVERSITY_CONTEXTS = [
    "Đây chỉ là một câu chuyện bên lề của tôi.",
    "Mình đang chia sẻ suy nghĩ chứ chưa cần tra cứu.",
    "Tôi chỉ muốn nói thêm về chủ đề này một chút.",
    "Chuyện này khiến tôi nhớ đến một cuộc trò chuyện cũ.",
    "Mình chưa có yêu cầu cụ thể nào cần xử lý.",
    "Tôi đang ghi lại cảm nhận cá nhân thôi.",
    "Đây chưa phải lúc tôi muốn thực hiện thao tác nào.",
    "Mình chỉ đang nối tiếp câu chuyện vừa nhắc tới.",
    "Tôi không cần tìm kiếm hay tính toán trong tình huống này.",
    "Đây chỉ là điều tôi chợt nghĩ đến hôm nay.",
    "Mình muốn giữ cuộc trao đổi ở mức trò chuyện.",
    "Tôi chỉ đang nêu một ý nghĩ chưa thành kế hoạch.",
    "Chủ đề này hiện chỉ mang tính tham khảo chung.",
    "Mình chưa muốn biến suy nghĩ này thành hành động.",
    "Tôi chỉ kể lại điều vừa nghe trong một cuộc trò chuyện.",
    "Đây là nhận xét cá nhân, không phải yêu cầu thao tác.",
    "Mình đang suy nghĩ thành tiếng về chuyện này thôi.",
    "Tôi chưa cần hệ thống xử lý thêm điều gì.",
    "Đây chỉ là một liên tưởng ngẫu nhiên của tôi.",
    "Mình muốn trao đổi ý kiến chứ không cần dùng công cụ.",
    "Tôi đang nói về chủ đề này theo nghĩa chung.",
    "Chuyện này chưa dẫn đến quyết định cụ thể nào.",
    "Mình chỉ muốn lưu lại suy nghĩ hiện tại.",
    "Tôi chưa định tìm, đặt hoặc tra cứu gì cả.",
    "Đây là phần trò chuyện không cần kết quả thực thi.",
    "Mình chỉ đang mô tả một tình huống giả định.",
    "Tôi muốn dừng ở việc bàn luận chung.",
    "Chủ đề này chỉ xuất hiện trong câu chuyện của tôi.",
    "Mình chưa cần một dịch vụ nào được thực hiện.",
    "Tôi chỉ đang nhắc lại một ý kiến đã nghe.",
    "Đây chưa phải một đề nghị tra cứu cụ thể.",
    "Mình chỉ muốn nói cho rõ bối cảnh câu chuyện.",
]

DOMAIN_TOPICS = {
    "food": ["ẩm thực Việt Nam", "phở và bún bò", "văn hóa ăn hàng"],
    "travel": ["du lịch nội địa", "chuyến đi xuyên Việt", "khách sạn và xe khách"],
    "finance": ["tài chính cá nhân", "lãi suất ngân hàng", "tỷ giá tiền tệ"],
    "education": ["thi cử và đại học", "việc học thêm", "học bổng du học"],
    "health": ["chăm sóc sức khỏe", "bệnh viện và nhà thuốc", "thói quen khám bệnh"],
    "ecommerce": ["mua sắm trực tuyến", "săn mã giảm giá", "giao hàng thương mại điện tử"],
    "transport": ["giao thông đô thị", "xe công nghệ", "phương tiện công cộng"],
    "public_services": ["thủ tục hành chính", "dịch vụ công trực tuyến", "hồ sơ giấy tờ"],
    "entertainment": ["sự kiện giải trí", "karaoke Việt Nam", "lịch âm và con giáp"],
    "agriculture": ["nông nghiệp Việt Nam", "mùa vụ", "giá nông sản"],
}


def registry_for_split(split: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"seen"}
    if split == "val_unseen":
        allowed.add("dev_unseen")
    if split.startswith("test"):
        allowed.add("test_unseen")
    return [item for item in tools if item["x-tool-split"] in allowed]


def gold_pool(split: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = (
        "dev_unseen"
        if split == "val_unseen"
        else "test_unseen"
        if split == "test_unseen"
        else "seen"
    )
    return [item for item in tools if item["x-tool-split"] == expected]


def candidate_tools(
    registry: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    domain: str,
    seed: int,
    count: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    chosen = {item["name"]: item for item in gold}
    same_domain = [
        item for item in registry if item["x-domain"] == domain and item["name"] not in chosen
    ]
    rng.shuffle(same_domain)
    for item in same_domain:
        if len(chosen) >= min(count, len(gold) + 4):
            break
        chosen[item["name"]] = item
    related = [
        item
        for item in registry
        if item["x-domain"] in RELATED_DOMAINS[domain] and item["name"] not in chosen
    ]
    rng.shuffle(related)
    for item in related:
        if len(chosen) >= min(count, len(gold) + 5):
            break
        chosen[item["name"]] = item
    rest = [item for item in registry if item["name"] not in chosen]
    rng.shuffle(rest)
    for item in rest:
        if len(chosen) >= count:
            break
        chosen[item["name"]] = item
    result = list(chosen.values())
    rng.shuffle(result)
    return result


def marked_surface(call_index: int, key: str, surface: str) -> str:
    return f"[[ARG:{call_index}:{key}]]{surface}[[/ARG]]"


def strip_argument_markers(
    marked_query: str, canonical_values: dict[tuple[int, str], Any]
) -> tuple[str, list[dict[str, Any]]]:
    pattern = re.compile(r"\[\[ARG:(\d+):([a-z0-9_]+)\]\](.*?)\[\[/ARG\]\]")
    plain_parts: list[str] = []
    output: list[dict[str, Any]] = []
    cursor = 0
    plain_length = 0
    for match in pattern.finditer(marked_query):
        prefix = marked_query[cursor : match.start()]
        plain_parts.append(prefix)
        plain_length += len(prefix)
        surface = match.group(3)
        call_index = int(match.group(1))
        key = match.group(2)
        start = plain_length
        end = start + len(surface)
        plain_parts.append(surface)
        plain_length = end
        output.append(
            {
                "call_index": call_index,
                "parameter_path": key,
                "surface": surface,
                "start": start,
                "end": end,
                "canonical_value": canonical_values[(call_index, key)],
            }
        )
        cursor = match.end()
    suffix = marked_query[cursor:]
    plain_parts.append(suffix)
    query = "".join(plain_parts)
    if "[[ARG:" in query or "[[/ARG]]" in query:
        raise ValueError("unparsed argument marker")
    return query, output


def clean_tool(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("x-")}


def generate_positive(
    sample_index: int,
    split: str,
    style: str,
    difficulty: str,
    primary: dict[str, Any],
    secondary: dict[str, Any] | None,
    tools: list[dict[str, Any]],
    cfg: Config,
) -> dict[str, Any]:
    selected = [primary] + ([secondary] if secondary is not None else [])
    clauses: list[str] = []
    calls = []
    canonical_values: dict[tuple[int, str], Any] = {}
    for offset, selected_tool in enumerate(selected):
        values = values_for_tool(selected_tool["name"], sample_index * 3 + offset)
        required = list(selected_tool["parameters"]["required"])
        chosen_optional = optional_keys(selected_tool["name"], selected_tool, sample_index + offset)
        extension_keys = [key for key in chosen_optional if key not in values]
        for key in extension_keys:
            values[key] = extension_value(
                key, selected_tool["parameters"]["properties"][key], sample_index + offset
            )
        keys = required + chosen_optional
        args = {key: values[key][0] for key in keys}
        calls.append({"name": selected_tool["name"], "arguments": args})
        marked_values = {
            key: (values[key][0], marked_surface(offset, key, values[key][1])) for key in keys
        }
        canonical_values.update({(offset, key): values[key][0] for key in keys})
        clause_text = clause(selected_tool["name"], marked_values, keys)
        extras = [
            extension_phrase(
                key, selected_tool["parameters"]["properties"][key], marked_values[key][1]
            )
            for key in extension_keys
        ]
        if extras:
            clause_text += ", " + ", ".join(extras)
        clauses.append(clause_text)
    body = clauses[0] if len(clauses) == 1 else f"{clauses[0]}, đồng thời {clauses[1]}"
    wrapper = STYLE_WRAPPERS[style][sample_index % len(STYLE_WRAPPERS[style])]
    query, argument_mentions = strip_argument_markers(wrapper(body), canonical_values)
    domain = primary["x-domain"]
    registry = registry_for_split(split, tools)
    candidates = candidate_tools(
        registry, selected, domain, cfg.seed * 100_000 + sample_index, cfg.candidate_count
    )
    return {
        "id": f"custom_vi_v1_{sample_index + 1:06d}",
        "source": "custom_vi",
        "query": query,
        "function_calls": calls,
        "tools": [clean_tool(item) for item in candidates],
        "metadata": {
            "dataset_version": cfg.dataset_version,
            "as_of": cfg.as_of,
            "split": split,
            "evaluation_track": cfg.splits[split]["track"],
            "tool_split": primary["x-tool-split"],
            "domain": domain,
            "difficulty": difficulty,
            "query_style": style,
            "negative_type": None,
            "scenario_family_id": f"positive:{primary['name']}:{sample_index // 80:04d}",
            "candidate_seed": cfg.seed * 100_000 + sample_index,
            "reference_datetime": cfg.reference_datetime,
            "argument_mentions": argument_mentions,
            "generation_method": "codex-authored-template-frame-first",
        },
    }


def generate_negative(
    sample_index: int,
    split: str,
    style: str,
    negative_type: str,
    domain: str,
    tools: list[dict[str, Any]],
    cfg: Config,
) -> dict[str, Any]:
    topic = DOMAIN_TOPICS[domain][sample_index % len(DOMAIN_TOPICS[domain])]
    template = NEGATIVE_TEMPLATES[negative_type][
        sample_index % len(NEGATIVE_TEMPLATES[negative_type])
    ]
    base = template.format(topic=topic, topic_cap=topic[0].upper() + topic[1:])
    if style == "polite":
        query = "Mình xin chia sẻ một chút: " + base[0].lower() + base[1:]
    elif style == "conversational":
        query = "Nói chuyện chút nhé, " + base[0].lower() + base[1:]
    elif style == "noisy":
        query = "Nói xíu thôi nha: " + base[0].lower() + base[1:]
    elif style == "code_mixed":
        query = "Just chatting: " + base[0].lower() + base[1:]
    else:
        query = base
    registry = registry_for_split(split, tools)
    candidates = candidate_tools(
        registry, [], domain, cfg.seed * 100_000 + sample_index, cfg.candidate_count
    )
    hard_distractors = [
        item["name"]
        for item in candidates
        if item["x-domain"] == domain or item["x-domain"] in RELATED_DOMAINS[domain]
    ][:5]
    return {
        "id": f"custom_vi_v1_{sample_index + 1:06d}",
        "source": "custom_vi",
        "query": query,
        "function_calls": [],
        "tools": [clean_tool(item) for item in candidates],
        "metadata": {
            "dataset_version": cfg.dataset_version,
            "as_of": cfg.as_of,
            "split": split,
            "evaluation_track": cfg.splits[split]["track"],
            "tool_split": "none",
            "domain": domain,
            "difficulty": "hard" if negative_type == "hard_near_miss" else "normal",
            "query_style": style,
            "negative_type": negative_type,
            "scenario_family_id": f"negative:{negative_type}:{domain}:{sample_index // 400:04d}",
            "candidate_seed": cfg.seed * 100_000 + sample_index,
            "reference_datetime": cfg.reference_datetime,
            "argument_mentions": [],
            "hard_distractor_names": hard_distractors,
            "generation_method": "codex-authored-template-frame-first",
        },
    }


def normalize_query(text: str) -> str:
    value = unicodedata.normalize("NFC", text).casefold()
    value = re.sub(r"[^\wÀ-ỹ<>]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def query_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold(), flags=re.UNICODE))


def diversity_block(record: dict[str, Any]) -> tuple[str, ...]:
    return ("all",)


def token_prefix(tokens: set[str], threshold: float) -> list[str]:
    ordered = sorted(tokens, key=lambda token: hashlib.sha256(token.encode("utf-8")).digest())
    prefix_length = len(tokens) - math.ceil(threshold * len(tokens)) + 1
    return ordered[: max(1, prefix_length)]


def near_duplicate_indexes(
    tokens: set[str],
    prior: list[set[str]],
    prefix_index: dict[str, set[int]],
    threshold: float,
) -> list[tuple[int, float]]:
    candidates: set[int] = set()
    for token in token_prefix(tokens, threshold):
        candidates.update(prefix_index.get(token, set()))
    matches: list[tuple[int, float]] = []
    for index in candidates:
        other = prior[index]
        if min(len(tokens), len(other)) / max(len(tokens), len(other)) < threshold:
            continue
        similarity = len(tokens & other) / len(tokens | other)
        if similarity >= threshold:
            matches.append((index, similarity))
    return matches


def add_to_prefix_index(
    tokens: set[str], prior_index: int, prefix_index: dict[str, set[int]], threshold: float
) -> None:
    for token in token_prefix(tokens, threshold):
        prefix_index.setdefault(token, set()).add(prior_index)


def diversify_queries(records: list[dict[str, Any]], threshold: float = 0.90) -> None:
    normalized_seen: set[str] = set()
    block_tokens: dict[tuple[str, ...], list[set[str]]] = defaultdict(list)
    block_prefixes: dict[tuple[str, ...], dict[str, set[int]]] = defaultdict(dict)
    for record in records:
        original = record["query"]
        serial = int(record["id"].rsplit("_", 1)[-1])
        candidates: list[tuple[str, str | None]] = [(original, None)]
        contexts = DIVERSITY_CONTEXTS if record["function_calls"] else NEGATIVE_DIVERSITY_CONTEXTS
        separator = " " if original.endswith((".", "!", "?")) else ". "
        for offset in range(len(contexts)):
            context = contexts[(serial * 17 + offset * 11) % len(contexts)]
            candidates.append((f"{original}{separator}{context}", context))
        for context_offset in range(len(contexts)):
            context = contexts[(serial * 17 + context_offset * 11) % len(contexts)]
            for detail_offset in range(len(DIVERSITY_DETAILS)):
                detail = DIVERSITY_DETAILS[
                    (serial * 7 + detail_offset * 5) % len(DIVERSITY_DETAILS)
                ]
                combined = f"{context} {detail}"
                candidates.append((f"{original}{separator}{combined}", combined))
        accepted: tuple[str, str | None] | None = None
        block = diversity_block(record)
        prior = block_tokens[block]
        prefix_index = block_prefixes[block]
        for candidate, optional_context in candidates:
            normalized = normalize_query(candidate)
            tokens = query_tokens(candidate)
            near_duplicate = bool(near_duplicate_indexes(tokens, prior, prefix_index, threshold))
            if normalized not in normalized_seen and not near_duplicate:
                accepted = candidate, optional_context
                break
        if accepted is None:
            raise ValueError(f"unable to diversify query below lexical threshold: {record['id']}")
        accepted_query, accepted_context = accepted
        record["query"] = accepted_query
        record["metadata"]["diversity_context"] = accepted_context
        normalized_seen.add(normalize_query(record["query"]))
        accepted_tokens = query_tokens(record["query"])
        add_to_prefix_index(accepted_tokens, len(prior), prefix_index, threshold)
        prior.append(accepted_tokens)


def generate(cfg: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tools = build_tools()
    records: list[dict[str, Any]] = []
    sample_index = 0
    for split, quotas in cfg.splits.items():
        positive_count = int(quotas["positive"])
        negative_count = int(quotas["negative"])
        styles = []
        for style, count in largest_remainder(
            positive_count + negative_count, cfg.query_style_distribution
        ).items():
            styles.extend([style] * count)
        random.Random(cfg.seed + len(records)).shuffle(styles)
        positives = gold_pool(split, tools)
        multi_count = round(positive_count * cfg.multi_call_ratio)
        domains = list(DOMAIN_NAMES)
        for local_index in range(positive_count):
            primary = positives[local_index % len(positives)]
            secondary = None
            if local_index < multi_count:
                same_domain = [
                    item
                    for item in positives
                    if item["x-domain"] == primary["x-domain"] and item["name"] != primary["name"]
                ]
                if same_domain:
                    secondary = same_domain[local_index % len(same_domain)]
                else:
                    other_tools = [item for item in positives if item["name"] != primary["name"]]
                    secondary = other_tools[local_index % len(other_tools)]
            difficulty = "hard" if secondary is not None or local_index % 5 == 0 else "normal"
            records.append(
                generate_positive(
                    sample_index,
                    split,
                    styles[local_index],
                    difficulty,
                    primary,
                    secondary,
                    tools,
                    cfg,
                )
            )
            sample_index += 1
        negative_types = []
        for kind, count in largest_remainder(negative_count, cfg.negative_distribution).items():
            negative_types.extend([kind] * count)
        random.Random(cfg.seed + len(records)).shuffle(negative_types)
        for local_index in range(negative_count):
            records.append(
                generate_negative(
                    sample_index,
                    split,
                    styles[positive_count + local_index],
                    negative_types[local_index],
                    domains[local_index % len(domains)],
                    tools,
                    cfg,
                )
            )
            sample_index += 1
    diversify_queries(records)
    return tools, records


def reconstruct(record: dict[str, Any]) -> list[dict[str, Any]]:
    mentions = record["metadata"]["argument_mentions"]
    if not mentions:
        return []
    by_call: dict[int, dict[str, Any]] = defaultdict(dict)
    for mention in mentions:
        by_call[int(mention["call_index"])][mention["parameter_path"]] = mention["canonical_value"]
    names = [call["name"] for call in record["function_calls"]]
    return [{"name": names[index], "arguments": by_call[index]} for index in sorted(by_call)]


def lexical_duplicate_stats(
    records: list[dict[str, Any]], threshold: float = 0.90
) -> dict[str, int | float]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[diversity_block(record)].append(record)
    pair_count = 0
    cross_split_pair_count = 0
    for group in groups.values():
        prior_tokens: list[set[str]] = []
        prior_records: list[dict[str, Any]] = []
        prefix_index: dict[str, set[int]] = {}
        for record in group:
            tokens = query_tokens(record["query"])
            for index, _ in near_duplicate_indexes(tokens, prior_tokens, prefix_index, threshold):
                pair_count += 1
                if record["metadata"]["split"] != prior_records[index]["metadata"]["split"]:
                    cross_split_pair_count += 1
            add_to_prefix_index(tokens, len(prior_tokens), prefix_index, threshold)
            prior_tokens.append(tokens)
            prior_records.append(record)
    return {
        "threshold": threshold,
        "pairs": pair_count,
        "cross_split_pairs": cross_split_pair_count,
    }


def validate(
    tools: list[dict[str, Any]], records: list[dict[str, Any]], cfg: Config
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    ids: set[str] = set()
    normalized_queries: set[str] = set()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    style_counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    call_counts: Counter[str] = Counter()
    positive_call_counts_by_tool: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    families: dict[str, set[str]] = defaultdict(set)
    optional_present = 0
    optional_absent = 0
    hard_negative_min_distractors = cfg.candidate_count
    valid_record_count = 0
    language_violations: list[dict[str, str]] = []
    all_names = {item["name"] for item in tools}
    schema_by_name = {item["name"]: item for item in tools}
    parameter_types = Counter(
        "enum" if "enum" in schema else str(schema.get("type"))
        for item in tools
        for schema in item["parameters"].get("properties", {}).values()
    )
    parameter_total = sum(parameter_types.values())
    parameter_targets = {"string": 0.45, "integer_number": 0.25, "enum": 0.20, "boolean": 0.10}
    parameter_rates = {
        "string": parameter_types["string"] / parameter_total,
        "integer_number": (parameter_types["integer"] + parameter_types["number"])
        / parameter_total,
        "enum": parameter_types["enum"] / parameter_total,
        "boolean": parameter_types["boolean"] / parameter_total,
    }
    for name, target in parameter_targets.items():
        if abs(parameter_rates[name] - target) > cfg.max_distribution_deviation:
            errors.append(
                {
                    "error": "parameter type distribution outside tolerance",
                    "type": name,
                    "actual": parameter_rates[name],
                    "target": target,
                }
            )
    for tool_item in tools:
        parameter_count = len(tool_item["parameters"].get("properties", {}))
        if not 1 <= parameter_count <= 5:
            errors.append(
                {
                    "id": tool_item["name"],
                    "error": f"parameter count outside 1-5: {parameter_count}",
                }
            )
        try:
            Draft202012Validator.check_schema(tool_item["parameters"])
        except Exception as exc:
            errors.append({"id": tool_item["name"], "error": f"invalid tool schema: {exc}"})
    pii_re = re.compile(r"\b(?:\d{12}|\d{9,16}|[A-Z]{2}\d{8,12})\b")
    for record in records:
        rid = str(record.get("id"))
        local: list[str] = []
        if list(record) != ["id", "source", "query", "function_calls", "tools", "metadata"]:
            local.append("top-level contract/order mismatch")
        if rid in ids:
            local.append("duplicate id")
        ids.add(rid)
        normalized = normalize_query(record["query"])
        if normalized in normalized_queries:
            local.append("normalized duplicate query")
        normalized_queries.add(normalized)
        candidate_names = [item.get("name") for item in record["tools"]]
        if (
            len(candidate_names) != cfg.candidate_count
            or len(set(candidate_names)) != cfg.candidate_count
        ):
            local.append("candidate count/uniqueness mismatch")
        if not set(candidate_names).issubset(all_names):
            local.append("unknown candidate tool")
        for candidate in record["tools"]:
            name = candidate.get("name")
            if name in schema_by_name and candidate != clean_tool(schema_by_name[name]):
                local.append("candidate schema differs from registry")
        for call in record["function_calls"]:
            positive_call_counts_by_tool[call["name"]] += 1
            if call["name"] not in candidate_names:
                local.append("gold tool absent from candidates")
                continue
            schema = schema_by_name[call["name"]]["parameters"]
            validation_errors = list(Draft202012Validator(schema).iter_errors(call["arguments"]))
            if validation_errors:
                local.append("gold arguments fail schema")
            optional = set(schema.get("properties", {})) - set(schema.get("required", []))
            optional_present += len(optional & set(call["arguments"]))
            optional_absent += len(optional - set(call["arguments"]))
        expected_mentions = {
            (call_index, key): value
            for call_index, call in enumerate(record["function_calls"])
            for key, value in call["arguments"].items()
        }
        actual_mentions: dict[tuple[int, str], Any] = {}
        mention_spans: list[tuple[int, int]] = []
        for mention in record["metadata"]["argument_mentions"]:
            if record["query"][mention["start"] : mention["end"]] != mention["surface"]:
                local.append("argument mention offset mismatch")
            mention_key = (int(mention["call_index"]), str(mention["parameter_path"]))
            if mention_key in actual_mentions:
                local.append("duplicate argument mention")
            actual_mentions[mention_key] = mention["canonical_value"]
            mention_spans.append((int(mention["start"]), int(mention["end"])))
        if actual_mentions != expected_mentions:
            local.append("argument mention coverage/value mismatch")
        for left, right in zip(sorted(mention_spans), sorted(mention_spans)[1:], strict=False):
            if left[1] > right[0]:
                local.append("argument mention spans overlap")
        if reconstruct(record) != record["function_calls"]:
            local.append("argument mention reconstruction mismatch")
        if (
            record["metadata"]["negative_type"] == "hard_near_miss"
            and len(record["metadata"].get("hard_distractor_names", [])) < 3
        ):
            local.append("hard negative has fewer than three hard distractors")
        if record["metadata"]["negative_type"] == "hard_near_miss":
            hard_negative_min_distractors = min(
                hard_negative_min_distractors,
                len(record["metadata"].get("hard_distractor_names", [])),
            )
        if pii_re.search(record["query"]):
            local.append("possible real PII")
        if "[[ARG:" in record["query"] or "[[/ARG]]" in record["query"]:
            local.append("unstripped argument marker")
        for pattern in (
            r"\bcó chỉ\b",
            r"không cần phù hợp",
            r"không bắt buộc chứng nhận",
            r"bằng phút\s*:",
            r"\bSố (?:khẩu phần|hành khách|khách lưu trú)\s*:",
        ):
            if re.search(pattern, record["query"], flags=re.IGNORECASE):
                language_violations.append({"id": str(rid), "pattern": pattern})
                local.append("known awkward Vietnamese pattern")
        for call in record["function_calls"]:
            if call["name"] == "vi_lookup_zodiac_information" and "zodiac" in call["arguments"]:
                expected_zodiac = ZODIACS[(int(call["arguments"]["birth_year"]) - 4) % len(ZODIACS)]
                if call["arguments"]["zodiac"] != expected_zodiac:
                    local.append("birth year and zodiac mismatch")
        split = record["metadata"]["split"]
        if split == "train" and any(
            schema_by_name[name]["x-tool-split"] != "seen" for name in candidate_names
        ):
            local.append("unseen schema leaked into train")
        if (
            split in {"train", "val_seen"}
            and record["function_calls"]
            and any(
                schema_by_name[call["name"]]["x-tool-split"] != "seen"
                for call in record["function_calls"]
            )
        ):
            local.append("wrong seen gold tool")
        if (
            split == "val_unseen"
            and record["function_calls"]
            and any(
                schema_by_name[call["name"]]["x-tool-split"] != "dev_unseen"
                for call in record["function_calls"]
            )
        ):
            local.append("wrong dev-unseen gold tool")
        if (
            split == "test_unseen"
            and record["function_calls"]
            and any(
                schema_by_name[call["name"]]["x-tool-split"] != "test_unseen"
                for call in record["function_calls"]
            )
        ):
            local.append("wrong test-unseen gold tool")
        family = record["metadata"]["scenario_family_id"]
        families[family].add(split)
        split_counts[split]["positive" if record["function_calls"] else "negative"] += 1
        style_counts[record["metadata"]["query_style"]] += 1
        if record["metadata"]["negative_type"]:
            negative_counts[record["metadata"]["negative_type"]] += 1
        call_counts[str(len(record["function_calls"]))] += 1
        domains[record["metadata"]["domain"]] += 1
        if local:
            errors.append({"id": rid, "errors": local})
        else:
            valid_record_count += 1
    leaked_families = [family for family, splits in families.items() if len(splits) > 1]
    if leaked_families:
        errors.append({"error": "scenario family leakage", "count": len(leaked_families)})
    for tool_split in ("seen", "dev_unseen", "test_unseen"):
        tier_counts = [
            positive_call_counts_by_tool[item["name"]]
            for item in tools
            if item["x-tool-split"] == tool_split
        ]
        if not tier_counts or min(tier_counts) == 0 or max(tier_counts) - min(tier_counts) > 1:
            errors.append(
                {
                    "error": "positive call coverage is missing or imbalanced within tool split",
                    "tool_split": tool_split,
                    "counts": tier_counts,
                }
            )
    for split, quotas in cfg.splits.items():
        for label in ("positive", "negative"):
            if split_counts[split][label] != int(quotas[label]):
                errors.append({"error": "quota mismatch", "split": split, "label": label})
    positive_total = sum(1 for record in records if record["function_calls"])
    multi_total = sum(1 for record in records if len(record["function_calls"]) == 2)
    optional_absence_rate = (
        optional_absent / (optional_present + optional_absent)
        if optional_present + optional_absent
        else 0.0
    )
    if not 0.35 <= optional_absence_rate <= 0.50:
        errors.append(
            {"error": "optional absence rate outside 35-50%", "actual": optional_absence_rate}
        )
    lexical_stats = lexical_duplicate_stats(records)
    if lexical_stats["pairs"]:
        errors.append({"error": "lexical near-duplicate pairs at threshold", **lexical_stats})
    report = {
        "dataset": "CustomTools-VI",
        "version": cfg.dataset_version,
        "as_of": cfg.as_of,
        "status": "passed" if not errors else "failed",
        "total_records": len(records),
        "total_tools": len(tools),
        "split_counts": {key: dict(value) for key, value in split_counts.items()},
        "style_counts": dict(style_counts),
        "negative_type_counts": dict(negative_counts),
        "call_count_distribution": dict(call_counts),
        "domain_counts": dict(domains),
        "positive_call_counts_by_tool": dict(sorted(positive_call_counts_by_tool.items())),
        "parameter_type_distribution": dict(parameter_types),
        "parameter_type_rates": parameter_rates,
        "multi_call_positive_ratio": multi_total / positive_total if positive_total else 0.0,
        "optional_parameter_absence_rate": optional_absence_rate,
        "hard_negative_minimum_hard_distractors": hard_negative_min_distractors,
        "lexical_near_duplicate_audit": lexical_stats,
        "diversity_context_records": sum(
            record["metadata"].get("diversity_context") is not None for record in records
        ),
        "known_language_pattern_violations": language_violations,
        "exact_duplicate_ids": len(records) - len(ids),
        "normalized_duplicate_queries": len(records) - len(normalized_queries),
        "cross_split_scenario_family_leakage": len(leaked_families),
        "argument_mention_reconstruction_passed": sum(
            reconstruct(record) == record["function_calls"] for record in records
        ),
        "record_level_checks_passed": valid_record_count,
        "errors": errors,
        "limitations": [
            "AI-synthesized and Codex-authored; no human annotation or inter-annotator agreement.",
            "No actual external API execution was used.",
            "Lexical Jaccard deduplication is exhaustive across all records; BGE-M3 embedding deduplication was not run because model weights are not locally available.",
        ],
    }
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def dataset_card(report: dict[str, Any]) -> str:
    return f"""# CustomTools-VI v1

CustomTools-VI v1 là bộ dữ liệu function calling tiếng Việt tổng hợp, tập trung vào ngữ cảnh sử dụng tại Việt Nam.

## Quy mô

- 8.000 mẫu, 40 tool, 10 nhóm chức năng.
- 4.800 positive và 3.200 negative.
- 15% positive là hai parallel calls.
- Mỗi mẫu có đúng 10 candidate tools.
- Split: train 5.600, val seen 400, val unseen 400, test seen 800, test unseen 800.

## Phương pháp

Gold semantic frame, function calls, arguments, split và candidate set được tạo trước query. Query được diễn đạt bằng các mẫu tiếng Việt do Codex biên soạn. Mọi record qua JSON Schema Draft 2020-12, argument span, quota, candidate integrity, PII, exact/normalized deduplication, lexical near-duplicate audit và reconstruction từ argument mentions.

Dataset được mô tả chính xác là **AI-synthesized và deterministically validated**. Dataset không được gán nhãn bởi con người và không thực thi API thật.

## Mốc dữ liệu

`as_of=2026-08-12`. Catalog dùng 34 đơn vị hành chính cấp tỉnh hiện hành, đồng thời cho phép các alias khẩu ngữ hoặc lịch sử như `Sài Gòn`, `SG`, `HN` trong query.

## Contract

Mỗi JSONL record có đúng sáu trường top-level: `id`, `source`, `query`, `function_calls`, `tools`, `metadata`. No-call được biểu diễn bằng `function_calls: []`; không lưu `has_tool_call`.

## QA cuối

- Trạng thái: `{report["status"]}`
- Tổng mẫu: `{report["total_records"]}`
- Argument-mention reconstruction pass: `{report["argument_mention_reconstruction_passed"]}`
- Normalized duplicates: `{report["normalized_duplicate_queries"]}`
- Lexical near-duplicate pairs tại ngưỡng 0,90: `{report["lexical_near_duplicate_audit"]["pairs"]}`
- Cross-split family leakage: `{report["cross_split_scenario_family_leakage"]}`

## Hạn chế

- Không có human validation hoặc inter-annotator agreement.
- Không có actual API execution.
- Chưa chạy semantic dedup bằng BGE-M3 vì môi trường phát hành không có model weights; báo cáo chỉ công bố lexical Jaccard audit trong các semantic block.
- V1 chỉ dùng flat parameter schemas; array/nested object dành cho challenge track sau.
"""


def write_release(
    tools: list[dict[str, Any]],
    records: list[dict[str, Any]],
    report: dict[str, Any],
    cfg: Config,
    config_path: Path,
) -> dict[str, Any]:
    output = cfg.output_dir
    if output.exists():
        resolved = output.resolve()
        expected_parent = Path("data/custom_vi").resolve()
        if expected_parent not in resolved.parents:
            raise ValueError(f"refusing to replace unexpected output directory: {resolved}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    write_json(output / "tools.json", tools)
    write_json(
        output / "catalogs" / "administrative_units.json",
        {"as_of": cfg.as_of, "provinces": PROVINCES, "aliases": LOCATION_SURFACES},
    )
    write_json(
        output / "catalogs" / "vietnam_context.json",
        {
            "foods": FOODS,
            "airlines": AIRLINES,
            "banks": BANKS,
            "universities": UNIVERSITIES,
            "marketplaces": MARKETPLACES,
            "carriers": CARRIERS,
            "ride_apps": RIDE_APPS,
            "crops": CROPS,
        },
    )
    write_json(
        output / "prompts" / "generation_spec.json",
        {
            "method": "frame-first",
            "generator": "Codex-authored deterministic templates",
            "argument_alignment": "inline markers stripped after exact span capture",
            "verification": "JSON Schema plus argument-mention reconstruction and independent artifact audit",
            "style_wrappers": list(STYLE_WRAPPERS),
            "negative_types": list(NEGATIVE_TEMPLATES),
        },
    )
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_split[record["metadata"]["split"]].append(record)
    for split, values in by_split.items():
        write_jsonl(output / f"{split}.jsonl", values)
    write_json(output / "qa_report.json", report)
    write_jsonl(output / "failed.jsonl", report["errors"])
    (output / "dataset_card.md").write_text(dataset_card(report), encoding="utf-8", newline="\n")
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "generation_manifest.json"
    )
    manifest = {
        "dataset": "CustomTools-VI",
        "version": cfg.dataset_version,
        "as_of": cfg.as_of,
        "seed": cfg.seed,
        "generator": "src/data/custom_vi_dataset.py",
        "generator_sha256": sha256_file(Path(__file__)),
        "auditor": "src/data/audit_custom_vi_dataset.py",
        "auditor_sha256": sha256_file(Path("src/data/audit_custom_vi_dataset.py")),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        },
    }
    write_json(output / "generation_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate CustomTools-VI v1")
    parser.add_argument("--config", type=Path, default=Path("configs/data/custom_vi.yaml"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    tools, records = generate(cfg)
    report = validate(tools, records, cfg)
    if report["status"] != "passed":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if not args.validate_only:
        write_release(tools, records, report, cfg, args.config)
        from src.data.audit_custom_vi_dataset import audit_release, write_audit_report

        artifact_report = audit_release(cfg.output_dir, args.config)
        write_audit_report(cfg.output_dir, artifact_report)
        if artifact_report["status"] != "passed":
            print(
                json.dumps(
                    {"qa": report, "artifact_audit": artifact_report}, ensure_ascii=False, indent=2
                )
            )
            raise SystemExit(1)
        manifest = json.loads(
            (cfg.output_dir / "generation_manifest.json").read_text(encoding="utf-8")
        )
        print(
            json.dumps(
                {
                    "qa": report,
                    "artifact_audit": artifact_report,
                    "manifest_files": len(manifest["files"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
