"""Repair protected values and corrupted repeated translations in xLAM JSONL."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CORRUPTION_PATTERN = re.compile(
    r"Bản dịch:|ftrong|trtr|chtr|hoặce|perstrên|fhoặc|trtrêng|Chrlà|Spatrong|"
    r"schoặce|mtrênthly|trênly|trongchomation|whtại|tạieghoặcy|ttrtrêng|ctrên|"
    r"strtr|Parlà|Smột|Pmột|Một[A-Z]|jstrên|hoặcg|mộtlo|đểp|danh sáchen|dmộth|"
    r"temộtms|collectitrên|ctại|củaficial|potrtrêngts|pythtrên|trongg|atrong\b|Natural ngôn ngữ",
    re.IGNORECASE,
)

ENTITY_KEYS = {
    "address", "airportiatacode", "artist", "artist_id", "brand", "brandname",
    "cast_name", "city", "city1", "city2", "cityname", "company", "company_name",
    "companyname", "continent", "country", "country1", "country2", "country_name",
    "county", "currentcountry", "default_country", "departure_city", "destination",
    "gallery", "geo", "geography", "gcountry", "hotel", "homecountry", "localite",
    "location", "location_name", "locationname", "nom", "origin", "pblocation",
    "pjlocation", "province", "purchasecountry", "region", "regions", "resort",
    "restaurant", "state", "state_name", "statename", "store", "store_location",
    "street", "town",
}

TECHNICAL_KEYS = {
    "$ne", "act", "alias", "api_key", "apikey", "aptto", "auction_type", "audio_lang",
    "base", "blockchain", "body_frame", "breed", "breedtype", "business_id",
    "business_type", "callback", "captcha", "case", "cat", "cat_id", "category",
    "category_group", "category_id", "category_name", "chain_slug", "choice", "cid",
    "class_name", "client_id", "client_secret", "climate", "code", "collection",
    "column", "command", "compounding", "configuration", "connection_string", "convert_currency",
    "corsenabled", "country_code", "country_iso2", "countrycode", "countrycodes",
    "currency", "currency_base", "currency_code", "currency_quote", "currencycode",
    "currencies", "cursor", "data_gradient_style", "data_type", "date_posted", "datestring",
    "daylights", "degree", "delimiter", "dest_type", "device", "dietary_preferences",
    "directory", "document_id", "domain", "domain1", "domain2", "domain_key", "domainname",
    "drive_type", "duration", "email", "exchange", "exclude", "fastav", "feed", "field",
    "fields", "fileid", "filter", "filter1", "filter_by_currency", "filter_language",
    "filters", "firmware_hash", "fitness_level", "flag", "fmt", "format", "frequency",
    "from_symbol", "from_unit", "fullscreen", "gameid", "geo_type", "get_dash_url",
    "getmusic", "gl", "glanguage", "group_by", "group_id", "guid", "hair", "hash",
    "hash_tag", "hashtag_id", "hl", "https_only", "iata", "id_race", "identifier",
    "ig", "image_id", "image_size", "image_url", "include", "include_skills", "in_currency",
    "in_stock", "interval", "is_from", "is_id", "ismobile", "iso", "iso_a2", "iso_a3",
    "iso_code", "job_id", "json", "key1", "key2", "kun", "l", "lan", "lang",
    "lang_from", "lang_to", "langs", "language", "language_code", "languagecode",
    "languageid", "linkedin_url", "list_id", "locale", "locale_info", "login_id", "lr",
    "manufacturer", "market", "match_id", "match_type", "max_cursor", "max_id", "meta_property",
    "minecraftversion", "mkt", "mnemonic", "mode", "model", "nationality", "network",
    "newspaperid", "next_cursor", "next_max_id", "nft", "nftnews", "nutrition_type",
    "order", "order_by", "order_size", "ordering", "orientation", "outputformat", "page_id",
    "page_order", "pair_interval", "passport", "password", "performance_rating", "performanceid",
    "period", "phone_id", "playlist_id", "popular_only", "position", "postcode", "postal_code",
    "prefix", "pricerange", "product_condition", "product_id", "product_type", "profile_id",
    "propertytypes", "provider", "proxy", "quality", "random", "range", "rank", "record_type",
    "region_code", "region_id", "repo", "request_id", "risk_rating", "role", "room_type",
    "ruleset", "schemaname", "score_id", "screener_id", "sec_uid", "sec_user_id", "secret",
    "securityid", "series_slug", "series_ticker", "series_type", "sess", "session_id", "sex",
    "shape", "short_code", "show_slug", "side", "signend", "signstart", "singleav", "slug",
    "sm_uid", "songid", "sort", "sort_by", "sortby", "source_id", "source_language", "speaker",
    "statement_type", "stock", "stockcode", "strategy", "subreddit", "subset", "subtitle_lang",
    "symb", "symbol", "symbols", "table", "target_lang", "target_language", "tbm", "tcin",
    "theme", "themes", "ticker", "ticker_slug", "tickername", "time_interval", "to_currency",
    "to_symbol", "to_unit", "token", "tolanguage", "track_url", "tradingsymbol", "transactionid",
    "transmission", "txid", "uid", "uname", "unique_id", "units", "upload_date", "uri", "url",
    "user_id", "userid", "username", "uuid", "valuta", "vehicle_type", "veiculo_tipo", "version",
    "video_id", "videoid", "vin", "vsid", "website", "wmid",
}

CATEGORICAL_KEYS = {
    "activity", "available", "breed", "category", "color", "colorname", "difficulty",
    "equipment", "extra", "food", "frequency", "fruit", "gender", "genre", "genres", "goal",
    "group", "include", "ingredient", "ingredients", "material", "meat", "muscle", "ordering",
    "part", "period", "product_condition", "property", "section", "sex", "shape", "species",
    "sport", "statement_type", "theme", "type", "vegetable", "vegetables",
}

MANUAL_DESCRIPTIONS = {
    "python": "python",
    "Fetches a list of special markets for a given sport. This involves making an initial snapshot call followed by delta calls based on changes since a specified 'since' parameter. It can filter by whether odds are available, specific leagues, event types, and event IDs.": "Lấy danh sách thị trường đặc biệt cho một môn thể thao nhất định. Thao tác này thực hiện lệnh gọi snapshot ban đầu, sau đó là các lệnh gọi delta dựa trên những thay đổi kể từ tham số 'since' đã chỉ định. Có thể lọc theo việc tỷ lệ cược có sẵn hay không, giải đấu, loại sự kiện và ID sự kiện cụ thể.",
    "Explore the latest book releases in our store!": "Khám phá những đầu sách mới phát hành tại cửa hàng của chúng tôi!",
    "Fetches official FedExCup points earned per player for a given tournament ID and year.": "Lấy số điểm FedExCup chính thức mà mỗi người chơi giành được theo ID giải đấu và năm đã cho.",
    "The tournament ID for which to fetch the points.": "ID giải đấu cần lấy điểm.",
    "Flag indicating whether to return a cost estimate based on the provided parameters. Defaults to None.": "Cờ cho biết có trả về ước tính chi phí dựa trên các tham số đã cung cấp hay không. Mặc định là None.",
    "The Steam application ID for which to retrieve achievement percentages.": "ID ứng dụng Steam cần truy xuất tỷ lệ phần trăm thành tích.",
    "The address of the contract.": "Địa chỉ của hợp đồng.",
    "Structure of the information to be retrieved (e.g., 'base_info', 'mini_info', 'image'). Defaults to None.": "Cấu trúc của thông tin cần truy xuất (ví dụ: 'base_info', 'mini_info', 'image'). Mặc định là None.",
    "The identifier of the competition stage in which teams compete. Use the 'Get Competitions' operation to find valid identifiers. Defaults to None.": "Mã định danh của giai đoạn thi đấu mà các đội tham gia. Sử dụng thao tác 'Get Competitions' để tìm các mã định danh hợp lệ. Mặc định là None.",
    "The name of the contract.": "Tên của hợp đồng.",
    "The identifier of the country from which the teams originate. Use the 'Get Countries' operation to find valid identifiers. Defaults to None.": "Mã định danh của quốc gia nơi các đội xuất phát. Sử dụng thao tác 'Get Countries' để tìm các mã định danh hợp lệ. Mặc định là None.",
    "If specified, filters results to only include matches with available odds (True) or matches that will get odds in the future (False).": "Nếu được chỉ định, chỉ giữ lại các trận đấu đã có tỷ lệ cược (True) hoặc sẽ có tỷ lệ cược trong tương lai (False).",
    "Fetches information about cat breeds based on the specified breed type.": "Lấy thông tin về các giống mèo dựa trên loại giống đã chỉ định.",
    "The year for which to fetch the points.": "Năm cần lấy điểm.",
    "A full or partial name of a team (case-insensitive). Minimum length is 4 characters. Defaults to 'liverp'.": "Tên đầy đủ hoặc một phần của đội (không phân biệt chữ hoa chữ thường). Độ dài tối thiểu là 4 ký tự. Mặc định là 'liverp'.",
    "Join us in our mission to help the homeless. Every donation counts": "Hãy cùng chúng tôi thực hiện sứ mệnh hỗ trợ người vô gia cư. Mọi khoản đóng góp đều có ý nghĩa.",
    "Retrieve the global achievement percentages for a specific Steam app.": "Truy xuất tỷ lệ phần trăm thành tích toàn cầu của một ứng dụng Steam cụ thể.",
    "Fetches a list of teams that match the given parameters from the API.": "Lấy từ API danh sách các đội khớp với những tham số đã cho.",
    "Comma-separated list of IMDB IDs (e.g., 'tt0001702,tt0001856,tt0001856').": "Danh sách ID IMDB được phân tách bằng dấu phẩy (ví dụ: 'tt0001702,tt0001856,tt0001856').",
    "Fetches all TCIA collection names, optionally specifying the output format.": "Lấy tên của tất cả bộ sưu tập TCIA, có thể chỉ định định dạng đầu ra.",
}

URL_PATTERN = re.compile(r"(?:https?://|www\.|[\w.-]+\.(?:com|org|net|io|gov|edu)(?:/|$))", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f-]{27,}$", re.IGNORECASE)
CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9._:/-]{1,}$")
IDENTIFIER_PATTERN = re.compile(r"^(?=.*\d)[A-Za-z0-9_.:@/+\-]{3,}$")
SNAKE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+$")
PATH_PATTERN = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\.\.?[\\/])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--translations-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def walk_description_pairs(source: Any, translated: Any, pairs: dict[str, Counter[str]], key: str = "") -> None:
    if isinstance(source, dict) and isinstance(translated, dict):
        for child_key, child_value in source.items():
            if child_key in translated:
                walk_description_pairs(child_value, translated[child_key], pairs, child_key)
    elif isinstance(source, list) and isinstance(translated, list):
        for source_item, translated_item in zip(source, translated):
            walk_description_pairs(source_item, translated_item, pairs, key)
    elif key == "description" and isinstance(source, str) and isinstance(translated, str):
        pairs[source][translated] += 1


def choose_description(source: str, candidates: Counter[str]) -> str:
    if source in MANUAL_DESCRIPTIONS:
        return MANUAL_DESCRIPTIONS[source]
    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            bool(CORRUPTION_PATTERN.search(item[0])),
            item[0] == source,
            -item[1],
            len(item[0]),
        ),
    )
    return ranked[0][0]


def replace_descriptions(source: Any, translated: Any, canonical: dict[str, str], stats: Counter[str], key: str = "") -> None:
    if isinstance(source, dict) and isinstance(translated, dict):
        for child_key, child_value in source.items():
            if child_key in translated:
                replace_descriptions(child_value, translated[child_key], canonical, stats, child_key)
    elif isinstance(source, list) and isinstance(translated, list):
        for source_item, translated_item in zip(source, translated):
            replace_descriptions(source_item, translated_item, canonical, stats, key)
    elif key == "description" and isinstance(source, str) and isinstance(translated, str):
        replacement = canonical[source]
        if translated != replacement:
            stats["descriptions_repaired"] += 1
        return


def repair_descriptions(source: Any, translated: Any, canonical: dict[str, str], stats: Counter[str], key: str = "") -> Any:
    if isinstance(source, dict) and isinstance(translated, dict):
        return {
            child_key: repair_descriptions(child_value, translated[child_key], canonical, stats, child_key)
            for child_key, child_value in source.items()
        }
    if isinstance(source, list) and isinstance(translated, list):
        return [repair_descriptions(a, b, canonical, stats, key) for a, b in zip(source, translated)]
    if key == "description" and isinstance(source, str) and isinstance(translated, str):
        replacement = canonical[source]
        if translated != replacement:
            stats["descriptions_repaired"] += 1
        return replacement
    return translated


def is_technical_value(key: str, value: str, protected_values: set[str]) -> bool:
    normalized_key = key.lower()
    if normalized_key in ENTITY_KEYS or normalized_key in TECHNICAL_KEYS or normalized_key in CATEGORICAL_KEYS:
        return True
    if value in protected_values:
        return True
    if URL_PATTERN.search(value) or EMAIL_PATTERN.fullmatch(value) or UUID_PATTERN.fullmatch(value):
        return True
    if CODE_PATTERN.fullmatch(value) or IDENTIFIER_PATTERN.fullmatch(value) or SNAKE_PATTERN.fullmatch(value):
        return True
    if PATH_PATTERN.search(value) or value in {"true", "false", "null", "None"}:
        return True
    if value.startswith("#") and " " not in value:
        return True
    if normalized_key == "function" or any(token in normalized_key for token in ("_id", "uuid", "token", "secret", "hash", "cursor")):
        return True
    return False


def collect_protected_values(value: Any, key: str, protected_values: set[str]) -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            collect_protected_values(child_value, child_key, protected_values)
    elif isinstance(value, list):
        for item in value:
            collect_protected_values(item, key, protected_values)
    elif isinstance(value, str) and key.lower() in ENTITY_KEYS:
        protected_values.add(value)


def repair_arguments(source: Any, translated: Any, key: str, protected_values: set[str], stats: Counter[str]) -> Any:
    if isinstance(source, dict) and isinstance(translated, dict):
        return {
            child_key: repair_arguments(child_value, translated[child_key], child_key, protected_values, stats)
            for child_key, child_value in source.items()
        }
    if isinstance(source, list) and isinstance(translated, list):
        return [repair_arguments(a, b, key, protected_values, stats) for a, b in zip(source, translated)]
    if isinstance(source, str) and isinstance(translated, str):
        if is_technical_value(key, source, protected_values) or CORRUPTION_PATTERN.search(translated):
            if source != translated:
                stats["argument_values_restored"] += 1
            return source
    return translated


def load_argument_overrides(directory: Path) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = defaultdict(dict)
    for name in (
        "xlam_codex_argument_overrides.json",
        "xlam_natural_argument_overrides_final.json",
        "xlam_search_argument_overrides.json",
        "xlam_search_argument_overrides_remaining.json",
        "xlam_argument_overrides_review.json",
    ):
        path = directory / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for key, values in data.items():
            merged[key].update(values)
    return dict(merged)


def apply_argument_overrides(
    source: Any,
    translated: Any,
    key: str,
    overrides: dict[str, dict[str, str]],
    protected_values: set[str],
    stats: Counter[str],
) -> Any:
    if isinstance(source, dict) and isinstance(translated, dict):
        return {
            child_key: apply_argument_overrides(
                child_value, translated[child_key], child_key, overrides, protected_values, stats
            )
            for child_key, child_value in source.items()
        }
    if isinstance(source, list) and isinstance(translated, list):
        return [
            apply_argument_overrides(a, b, key, overrides, protected_values, stats)
            for a, b in zip(source, translated)
        ]
    if isinstance(source, str) and isinstance(translated, str):
        replacement = overrides.get(key, {}).get(source)
        if replacement is None and is_technical_value(key, source, protected_values):
            return translated
        if replacement is not None:
            if translated != replacement:
                stats["argument_overrides_applied"] += 1
            return replacement
    return translated


def load_query_overrides(directory: Path) -> dict[str, str]:
    merged: dict[str, str] = {}
    paths = sorted(directory.glob("xlam_codex_query_overrides*.json"))
    paths.append(directory / "xlam_corrupt_query_overrides.json")
    for path in paths:
        if path.exists() and path.stat().st_size:
            merged.update(json.loads(path.read_text(encoding="utf-8-sig")))
    return merged


def collect_changed_strings(source: Any, translated: Any, pairs: set[tuple[str, str]]) -> None:
    if isinstance(source, dict) and isinstance(translated, dict):
        for key, source_value in source.items():
            collect_changed_strings(source_value, translated[key], pairs)
    elif isinstance(source, list) and isinstance(translated, list):
        for source_value, translated_value in zip(source, translated):
            collect_changed_strings(source_value, translated_value, pairs)
    elif isinstance(source, str) and isinstance(translated, str) and source != translated:
        pairs.add((source, translated))


def sync_query_arguments(source: dict[str, Any], repaired: dict[str, Any], stats: Counter[str]) -> None:
    pairs: set[tuple[str, str]] = set()
    for index, source_call in enumerate(source["function_calls"]):
        collect_changed_strings(
            source_call["arguments"], repaired["function_calls"][index]["arguments"], pairs
        )
    query = repaired["query"]
    for source_value, translated_value in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(r"(?P<quote>['\"])" + re.escape(source_value) + r"(?P=quote)")
        query, occurrences = pattern.subn(
            lambda match: f"{match.group('quote')}{translated_value}{match.group('quote')}", query
        )
        stats["query_argument_literals_synchronized"] += occurrences
    repaired["query"] = query


def repair_embedded_single_letter_translation(query: str, stats: Counter[str]) -> str:
    repaired, count = re.subn(r"(?<=[A-Za-zÀ-ỹ])một|một(?=[A-Za-zÀ-ỹ])", "a", query)
    stats["embedded_single_letter_query_repairs"] += count
    return repaired


def main() -> None:
    args = parse_args()
    source_records = load_jsonl(args.source)
    translated_records = load_jsonl(args.input)
    if len(source_records) != len(translated_records):
        raise ValueError("Source and translated JSONL files have different line counts")
    description_pairs: dict[str, Counter[str]] = defaultdict(Counter)
    protected_values: set[str] = set()
    for source, translated in zip(source_records, translated_records):
        if source["id"] != translated["id"]:
            raise ValueError(f"Record ID mismatch: {source['id']} != {translated['id']}")
        walk_description_pairs(source, translated, description_pairs)
        for call in source["function_calls"]:
            collect_protected_values(call["arguments"], "", protected_values)
    canonical = {source: choose_description(source, candidates) for source, candidates in description_pairs.items()}
    query_overrides = load_query_overrides(args.translations_dir)
    argument_overrides = load_argument_overrides(args.translations_dir)
    stats: Counter[str] = Counter()
    repaired_records: list[dict[str, Any]] = []
    for source, translated in zip(source_records, translated_records):
        repaired = repair_descriptions(source, translated, canonical, stats)
        if source["id"] in query_overrides and repaired["query"] != query_overrides[source["id"]]:
            repaired["query"] = query_overrides[source["id"]]
            stats["query_overrides_applied"] += 1
        repaired["query"] = repair_embedded_single_letter_translation(repaired["query"], stats)
        for index, source_call in enumerate(source["function_calls"]):
            translated_arguments = repaired["function_calls"][index]["arguments"]
            repaired_arguments = repair_arguments(source_call["arguments"], translated_arguments, "", protected_values, stats)
            repaired_arguments = apply_argument_overrides(
                source_call["arguments"], repaired_arguments, "", argument_overrides, protected_values, stats
            )
            repaired["function_calls"][index]["arguments"] = repaired_arguments
        sync_query_arguments(source, repaired, stats)
        repaired_records.append(repaired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in repaired_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    stats["records"] = len(repaired_records)
    stats["canonical_descriptions"] = len(canonical)
    stats["protected_value_lexicon"] = len(protected_values)
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
