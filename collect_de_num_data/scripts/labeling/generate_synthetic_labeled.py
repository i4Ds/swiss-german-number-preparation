#!/usr/bin/env python3
"""Generate directly labeled synthetic number data through OpenRouter."""

import argparse
import csv
import calendar
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline_runtime import require_compute_node, storage_path

OUTPUT_DIR = storage_path("intermediate", "labels")
# OpenRouter's currently available synchronous endpoint for the current Lite
# model. Keep generation resumable at the application level rather than relying
# on the unavailable provider Batch API route.
MODEL = "google/gemini-3.5-flash-lite"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
EXPLORATORY_SHARE = 0.15
NUM2WORDS_CH_ROOT = Path(
    os.environ.get("NUM2WORDS_CH_ROOT", PROJECT_ROOT.parent / "num2words_CH")
).expanduser().resolve()
ZIP_SOURCE = NUM2WORDS_CH_ROOT / "helper_data" / "PLZ_Ortschaften.csv"

# Test adds 20 ZIP and 20 VERSION examples to the requested 200 samples.
PRESETS = {
    "test": {
        ("CURRENCY", "CHF"): 13, ("CURRENCY", "EUR"): 3,
        ("CURRENCY", "USD"): 2, ("CURRENCY", "OTHER"): 2,
        ("IBAN", "CH"): 7, ("IBAN", "OTHER"): 3,
        ("PHONE", "CH"): 32, ("PHONE", "OTHER"): 8,
        ("PRODUCT_ID", "MIXED"): 30, ("MATH_EXPRESSION", "MIXED"): 100,
        ("ZIP", "CH"): 20, ("VERSION", "MIXED"): 20,
        ("COORDINATE", "MIXED"): 20,
    },
    "full": {
        ("CURRENCY", "CHF"): 2000, ("CURRENCY", "EUR"): 500,
        ("CURRENCY", "USD"): 300, ("CURRENCY", "OTHER"): 200,
        ("IBAN", "CH"): 300, ("IBAN", "OTHER"): 100,
        ("PHONE", "CH"): 1600, ("PHONE", "OTHER"): 400,
        ("PRODUCT_ID", "MIXED"): 6000, ("MATH_EXPRESSION", "MIXED"): 3000,
        ("ZIP", "CH"): 1500, ("VERSION", "MIXED"): 500,
        ("COORDINATE", "MIXED"): 500,
    },
    "topup": {
        ("CURRENCY_BELOW_ONE", "CHF"): 100,
        ("CURRENCY_BELOW_ONE", "EUR"): 100,
        ("CURRENCY_BELOW_ONE", "USD"): 100,
        ("NUMBER_WITH_SUFFIX", "MIXED"): 100,
        ("NUMBER_WITH_SUFFIX", "DECADE"): 80,
        ("NEGATIVE", "NUMBER_CARDINAL"): 140,
        ("NEGATIVE", "DECIMAL"): 140,
        ("NEGATIVE", "CURRENCY"): 120,
        ("RANGE", "DATE"): 120,
        ("RATIO", "SCALE"): 80,
        ("COORDINATE", "MIXED"): 80,
    },
    "zip_date_time_refresh": {
        ("ZIP", "CH_REAL"): 1500,
        ("DATE", "NUMERIC"): 250,
        ("DATE", "WRITTEN_MONTH"): 250,
        ("DATE", "COMPACT"): 150,
        ("DATE", "ISO"): 150,
        ("DATE", "PARTIAL"): 200,
        ("TIME", "COLON"): 120,
        ("TIME", "DOT"): 100,
        ("TIME", "H_NOTATION"): 100,
        ("TIME", "COMPACT"): 100,
        ("TIME", "HOUR_CONTEXT"): 100,
        ("TIME", "SECONDS"): 80,
    },
    "tts_edge_cases": {
        ("TIME", "MULTI"): 600,
        ("PIN", "DIGITS"): 500,
    },
}

# Each generated batch is assigned a semantic area and subarea. This makes
# coverage intentional rather than merely asking the model for "variety".
CONTEXTS = {
    "CURRENCY": [
        ("retail", "groceries and pharmacies"),
        ("retail", "clothing, furniture, and electronics"),
        ("invoicing", "professional services and repairs"),
        ("subscriptions", "software, streaming, and memberships"),
        ("travel", "tickets, accommodation, and car rental"),
        ("household", "rent, utilities, and insurance"),
        ("public services", "fees, permits, and taxes"),
        ("work", "salary, expenses, and reimbursements"),
        ("finance", "transfers, investments, and exchange"),
        ("community", "donations, clubs, and events"),
        ("commerce", "discounts, refunds, and deposits"),
        ("health", "medical bills and prescriptions"),
    ],
    "IBAN": [
        ("payments", "invoice settlement"),
        ("payments", "standing orders"),
        ("payments", "refund instructions"),
        ("work", "salary and expense reimbursement"),
        ("membership", "club and association fees"),
        ("community", "donations and fundraising"),
        ("commerce", "online checkout and marketplace payout"),
        ("housing", "rent and property management"),
        ("utilities", "energy, telecom, and insurance"),
        ("administration", "forms and payment notices"),
    ],
    "PHONE": [
        ("service", "customer support"),
        ("service", "technical support and repairs"),
        ("health", "appointments and pharmacies"),
        ("travel", "reservations and transport"),
        ("delivery", "courier and collection notices"),
        ("administration", "public offices and schools"),
        ("housing", "property and maintenance"),
        ("hospitality", "restaurants and hotels"),
        ("work", "switchboards and internal contacts"),
        ("community", "clubs and event organization"),
        ("safety", "non-emergency information"),
        ("commerce", "shops and returns"),
    ],
    "PRODUCT_ID": [
        ("consumer electronics", "phones, audio, and cameras"),
        ("computing", "servers, peripherals, and storage"),
        ("software", "licenses, plugins, and internal builds"),
        ("automotive", "vehicle parts and accessories"),
        ("industrial", "tools, sensors, and machinery"),
        ("laboratory", "instruments, reagents, and consumables"),
        ("medical", "devices and spare parts"),
        ("logistics", "containers, scanners, and inventory"),
        ("home", "appliances, heating, and lighting"),
        ("construction", "materials, fixtures, and equipment"),
        ("retail", "article and catalogue numbers"),
        ("agriculture", "equipment and replacement parts"),
        ("telecom", "network hardware and SIM products"),
        ("energy", "meters, controllers, and batteries"),
        ("publishing", "editions and media identifiers"),
        ("education", "course materials and equipment"),
        ("publishing", "ISBN identifiers and catalogue references"),
        ("standards", "ISO identifiers and Unicode code points"),
        ("science", "compact formula-like labels such as CO2"),
    ],
    "MATH_EXPRESSION": [
        ("arithmetic", "addition, subtraction, multiplication, and division"),
        ("decimal arithmetic", "money-like and measurement values"),
        ("fractions", "simple and mixed fractions"),
        ("percentages", "discounts, growth, and proportions"),
        ("powers", "squares, exponents, and scientific notation"),
        ("roots", "square roots and geometric calculations"),
        ("brackets", "nested operation order"),
        ("algebra", "linear equations with variables"),
        ("algebra", "quadratic-style expressions"),
        ("inequalities", "comparisons and bounds"),
        ("functions", "sin, cos, tan, and log"),
        ("geometry", "areas, perimeters, and dimensions"),
        ("physics", "simple formula rearrangements"),
        ("statistics", "averages and ratios"),
        ("programming", "conditions and thresholds"),
        ("finance", "interest, rates, and percentage change"),
    ],
    "ZIP": [
        ("delivery", "parcel and return notices"),
        ("service", "coverage and dispatch areas"),
        ("administration", "form and correspondence references"),
        ("retail", "collection and branch notices"),
        ("housing", "building management correspondence"),
        ("health", "service-region references"),
        ("education", "campus and mailing references"),
        ("events", "venue-area notices"),
        ("transport", "service-zone references"),
        ("utilities", "regional service notices"),
    ],
    "DATE": [
        ("appointments", "medical, advisory, and service appointments"),
        ("deadlines", "applications, registrations, and submissions"),
        ("bookings", "travel, accommodation, and restaurant reservations"),
        ("delivery", "delivery, collection, and installation dates"),
        ("work", "meetings, interviews, and project milestones"),
        ("administration", "forms, notices, validity, and correspondence"),
        ("education", "exams, courses, enrolment, and parent meetings"),
        ("events", "concerts, exhibitions, markets, and club events"),
        ("publishing", "release, issue, and publication dates"),
        ("everyday", "visits, celebrations, and household plans"),
    ],
    "TIME": [
        ("appointments", "medical, advisory, and service appointments"),
        ("transport", "departures, arrivals, and platform announcements"),
        ("work", "meetings, shifts, calls, and interviews"),
        ("delivery", "delivery, collection, and installation windows"),
        ("education", "lessons, exams, and parent meetings"),
        ("events", "doors, starts, breaks, and closing times"),
        ("broadcast", "radio, television, and livestream schedules"),
        ("hospitality", "reservations, breakfast, and kitchen hours"),
        ("administration", "office hours and scheduled consultations"),
        ("everyday", "alarms, visits, calls, and household plans"),
    ],
    "PIN": [
        ("devices", "fictional device setup and pairing"),
        ("buildings", "fictional doors, lockers, and access panels"),
        ("accounts", "fictional account verification and recovery"),
        ("payments", "fictional card and terminal verification"),
        ("support", "fictional customer-support instructions"),
        ("travel", "fictional ticket machines and hotel safes"),
    ],
    "VERSION": [
        ("software", "application releases"),
        ("firmware", "device updates"),
        ("API", "compatibility requirements"),
        ("protocol", "network and data formats"),
        ("security", "patch and dependency requirements"),
        ("mobile", "app release notes"),
        ("web", "frontend and backend deployment"),
        ("database", "migration compatibility"),
        ("hardware", "driver and controller releases"),
        ("documentation", "specification revisions"),
        ("gaming", "game and engine updates"),
        ("research", "tool and workflow versions"),
    ],
    "CURRENCY_BELOW_ONE": [
        ("retail", "small grocery and kiosk purchases"),
        ("fees", "small service charges and adjustments"),
        ("discounts", "credits, rebates, and price reductions"),
        ("community", "small donations and contributions"),
        ("transport", "supplements, parking, and ticket adjustments"),
        ("digital", "microtransactions and usage charges"),
    ],
    "NUMBER_WITH_SUFFIX": [
        ("addresses", "fictional street and house-number references"),
        ("buildings", "entrances, wings, and building labels"),
        ("rooms", "rooms, storage areas, and internal locations"),
        ("objects", "lockers, parcels, seats, and numbered items"),
        ("administration", "fictional form and location references"),
        ("history", "decades, exhibitions, archives, and retrospectives"),
        ("culture", "music, fashion, and design decade references"),
    ],
    "RANGE": [
        ("history", "year ranges in archives and exhibitions"),
        ("education", "school years, course years, and academic periods"),
        ("events", "date ranges for schedules and programs"),
        ("administration", "validity periods and reporting years"),
        ("publishing", "edition years and archive spans"),
        ("sport", "season ranges and tournament years"),
    ],
    "RATIO": [
        ("maps", "map scales and route plans"),
        ("engineering", "technical scale drawings"),
        ("model making", "miniature and prototype scales"),
        ("printing", "layout and reproduction scales"),
        ("education", "geography and measurement exercises"),
    ],
    "COORDINATE": [
        ("geography", "latitude and longitude examples"),
        ("mapping", "surveying and route descriptions"),
        ("navigation", "coordinates with cardinal directions"),
        ("education", "geography exercises and atlas references"),
        ("science", "field measurements and location notes"),
    ],
    "NEGATIVE": [
        ("weather", "temperatures and temperature changes"),
        ("finance", "balances, corrections, and account movements"),
        ("science", "measurements, coordinates, and experimental values"),
        ("statistics", "scores, deviations, and percentage changes"),
        ("elevation", "depths and positions below a reference point"),
        ("commerce", "price adjustments and negative amounts"),
        ("technology", "offsets, settings, and calibration values"),
        ("everyday", "ratings, floors, and numbered scales"),
    ],
}


def api_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    for env_path in (
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env",
        PROJECT_ROOT.parent / ".env",
    ):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "OPENROUTER_API_KEY":
                return value.strip().strip('"').strip("'")
    return None


def request_samples(
    key: str, category: str, subgroup: str, area: str, subarea: str, count: int
) -> list[dict]:
    payload = {
        "model": MODEL,
        "temperature": 0.9,
        "max_tokens": max(1200, count * 300),
        "response_format": response_schema(category, subgroup),
        "reasoning": {"effort": "minimal", "exclude": True},
        "messages": [
            {"role": "system", "content": system_prompt(category, subgroup)},
            {"role": "user", "content": user_prompt(category, subgroup, area, subarea, count)},
        ],
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/openai/codex",
            "X-Title": "German Number Data Collection",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc

    content = body.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError("OpenRouter response has no JSON content")
    try:
        samples = json.loads(content).get("samples")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(samples, list):
        raise RuntimeError("Model JSON must contain a samples list")
    return samples


def system_prompt(category: str = "", subgroup: str = "") -> str:
    annotation_rule = (
        "Each TIME/MULTI sample must have between two and four annotations."
        if category == "TIME" and subgroup == "MULTI"
        else "Each sample must have exactly one annotation."
    )
    return f"""Generate synthetic, natural High German TTS training sentences.
Return only JSON. Every item must be grammatically clean. Use fictional examples
only: no real people, accounts, contact details, addresses, companies, or brands.

Return {{"samples":[{{"text":"...","numbers":[{{"number":"...","value":...}}]}}]}}.
{annotation_rule} Every number must be copied verbatim from text and occur
there exactly once. Do not include other digits, dates,
amounts, versions, identifiers, or mathematical expressions in the sentence."""


def response_schema(category: str = "", subgroup: str = "") -> dict:
    """JSON mode alone is not reliable enough for this model's generated data."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "number_samples",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["samples"],
                "properties": {
                    "samples": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["text", "numbers"],
                            "properties": {
                                "text": {"type": "string"},
                                "numbers": {
                                    "type": "array",
                                    "minItems": 1,
                                    "maxItems": 4 if category == "TIME" and subgroup == "MULTI" else 1,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["number", "value"],
                                        "properties": {
                                            "number": {"type": "string"},
                                            "value": {},
                                            "type": {"type": "string"},
                                            "subtype": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def user_prompt(category: str, subgroup: str, area: str, subarea: str, count: int) -> str:
    if area == "exploratory":
        context_instruction = (
            "Choose a plausible, varied context not limited to the planned "
            "areas; vary the wording, sentence structure, and vocabulary."
        )
    else:
        context_instruction = (
            f"Use the semantic area {area} and subarea {subarea}; vary the "
            "wording, sentence structure, and vocabulary within that context."
        )
    lead = f"Generate exactly {count} distinct samples in category {category}. {context_instruction}"
    if category == "CURRENCY":
        group = {"CHF": "CHF only", "EUR": "EUR only", "USD": "USD only",
                 "OTHER": "a balanced mix of GBP, JPY, CAD, AUD, SEK, NOK, DKK, PLN, and TRY"}[subgroup]
        return lead + f""" Use {group}. The single annotation value is exactly
{{"major":"digits","minor":"0-99 digits or empty","currency":"ISO code"}}.
Number includes the amount and currency marker, for example CHF 3.40. Amounts
below one major currency unit must use major "0" and the non-zero minor amount."""
    if category == "CURRENCY_BELOW_ONE":
        return lead + f""" Use {subgroup} only. Every amount must be strictly
below one major currency unit and greater than zero. The single annotation
value must copy the actual amount, for example CHF 0.40 has
{{"major":"0","minor":"40","currency":"CHF"}} and EUR 0.07 has
{{"major":"0","minor":"07","currency":"EUR"}}. Never write a pattern such
as "01-99" into the value. Use varied minor amounts, not only round tens.
Number includes the complete amount and currency marker, for example CHF 0.40,
0,37 Euro, or USD 0.85."""
    if category == "IBAN":
        group = "Swiss CH IBANs" if subgroup == "CH" else "a balanced DE, AT, FR, IT, NL, and GB IBAN mix"
        return lead + f""" Use fictional but format-plausible {group}. Number and
value are the same complete, space-grouped IBAN. Do not use a bank or person."""
    if category == "PHONE":
        group = "Swiss +41 or domestic 0 phone numbers" if subgroup == "CH" else "a balanced DE, AT, FR, IT, GB, and US mix"
        return lead + f""" Use fictional {group}. Number is formatted; value is
digits only, preserving a leading +. Do not use a person name."""
    if category == "PIN":
        return lead + """ Generate fictional four- to eight-digit PIN/access
codes that must be read strictly digit by digit. At least half of the samples
must begin with zero, and many must contain one or more internal zeros. Use
natural phrases such as "Der PIN lautet 0417" or "Geben Sie den Zugangscode
003905 ein". The annotation number and value are the identical digit string,
including every leading zero. Annotate only the digits, not the word PIN or
Code. Do not group the code as a cardinal and do not add other digits."""
    if category == "PRODUCT_ID":
        return lead + """ Create a wide mix of fictional model, article, batch,
device, component, catalogue, ISBN-like, ISO-like, Unicode-codepoint, and
compact formula-like identifiers such as A12/3, ISBN 3-85124-698-5,
ISO 3166-2, U+0339, or CO2. Number and value are identical strings. Copy the
complete identifier with its original letter case and separators. Use letters,
digits, hyphens, plus signs, or slashes; no real brands."""
    if category == "MATH_EXPRESSION":
        return lead + """ Cover arithmetic, brackets, fractions, percentages,
decimals, powers, roots, variables, functions, inequalities, and equations.
Use standard mathematical notation. Allowed operators are plus, minus,
multiplication, division, equals, less-than, greater-than, not-equal, power,
root, percent, and round or square brackets. Number and value are the complete
expression. Put that exact symbolic expression visibly in text exactly once;
never paraphrase it into words. For example, text may say "Bitte löse 7 + 5 =
12." and number/value are both "7 + 5 = 12"."""
    if category == "ZIP":
        verified_pairs = _zip_prompt_pairs(area, subarea)
        return lead + """ Use a real Swiss four-digit postal code together with
its correct locality in the sentence. Never invent a postal code/locality pair.
The annotation number and value contain only the same four ZIP digits; do not
include the locality in the annotation. Use natural contexts and varied
localities from across Switzerland. Do not add any other digits to the sentence.
Choose exclusively from these verified ZIP/locality pairs (ZIP locality):
""" + verified_pairs
    if category == "DATE":
        format_instruction = {
            "NUMERIC": "Use varied dot, slash, or hyphen dates such as 3.8.2026, 03/08/26, or 3-08-2026.",
            "WRITTEN_MONTH": "Use written or common abbreviated German month names, such as 3. August 2026, 03. Aug. 2026, or August 2026.",
            "COMPACT": "Use clearly contextual compact calendar forms such as 03082026 or 030826.",
            "ISO": "Use ISO-style calendar forms such as 2026-08-03 or the partial form 2026-08.",
            "PARTIAL": "Use day+month or month+year forms such as 3.8., 03/08, August 2026, or 08.2026.",
        }[subgroup]
        return lead + f""" {format_instruction} Every DATE must contain at least
two components: day+month, month+year, or day+month+year. Never create a
year-only annotation or a historical-event sentence. The context must make the
calendar meaning unmistakable. The annotation number is the complete date
expression verbatim and value is ["DD","MM","YYYY"], using an empty string
only for a genuinely omitted component. Use valid calendar dates and no other
digits in the sentence. Vary leading zeros, two- and four-digit years,
punctuation, written months, sentence structure, and position."""
    if category == "TIME":
        if subgroup == "MULTI":
            return lead + """ Every sentence must contain two to four distinct
clock-time expressions, each as its own annotation, especially opening hours,
service windows, shifts, departures/arrivals, lesson blocks, and appointment
changes. Prefer forms such as "von 08:30 bis 18:45 Uhr", "zwischen 7.15 und
12.05 Uhr", or a change "von 09:00 auf 10:30 Uhr". Include "Uhr" in an
annotation only when it directly follows that particular time. Each value is
["hh","mm","ss"] with empty strings for omitted components. Hours are 0-23
and minutes/seconds 0-59. Vary leading zeros, delimiters, hour combinations,
wording, and the number of times. Do not include any other digits."""
        format_instruction = {
            "COLON": "Use colon notation with varied precision, such as 8:05, 08:05, or 8:05:09.",
            "DOT": "Use clearly contextual dot notation such as 7.45 or 07.45.30.",
            "H_NOTATION": "Use h notation such as 8h, 8h05, 08h30, or 8 h 05.",
            "COMPACT": "Use clearly contextual compact notation such as 0630, 915, or 221505.",
            "HOUR_CONTEXT": "Use a bare one- or two-digit hour only when explicit words such as um, gegen, ab, bis, Punkt, Uhr, Termin, Beginn, Abfahrt, or Öffnung make it unambiguously a time.",
            "SECONDS": "Use unusual but clear forms including seconds, such as 8:05:09, 8.05.09, 080509, or 8h05m09s.",
        }[subgroup]
        return lead + f""" {format_instruction} The context must make the time
meaning unmistakable. Annotation number is the complete visible time
expression; when "Uhr" directly follows it, include "Uhr" in number. Value is
["hh","mm","ss"], with empty strings for omitted minute or second. Hours must
be 0-23 and minutes/seconds 0-59. Do not use vague written-only times and do not
include any other digits in the sentence. Vary wording, leading zeros,
punctuation, sentence position, and formal/informal scheduling contexts."""
    if category == "VERSION":
        return lead + """ Create fictional software, firmware, protocol, and API
versions with numeric dotted components, such as v2.4.1 or 3.12.0. Use a varied
mix of versions with and without a literal v/V prefix. When that prefix is shown,
it is part of the version: preserve it verbatim in both number and value.
Number and value are the same complete version."""
    if category == "NUMBER_WITH_SUFFIX":
        if subgroup == "DECADE":
            return lead + """ Create natural sentences with German decade forms
such as 1970er, 80er, or 1990er. The annotation type is NUMBER_WITH_SUFFIX,
subtype is DECADE, number is the complete compact expression, and value is
{"number":"digits","suffix":"er"}. Do not include any other digits in the
sentence."""
        return lead + """ Create fictional street, building, entrance, room,
locker, parcel, seat, or similar names containing one compact whole number plus
one directly attached letter, such as Munzachstrasse 24c. The annotated number
is only the compact expression such as "24c". Its value is exactly
{"number":"24","suffix":"c"}. Use varied whole numbers and letters. Do not
include any other digits in the sentence and do not use a real full address."""
    if category == "RANGE":
        if subgroup == "DATE":
            return lead + """ Create compact German date or year ranges,
especially slash-delimited forms such as 1998/99, 2010/2011, or
01.02.1998/03.02.1998. The annotation type is RANGE, subtype is DATE, number is
the complete range expression including the delimiter, and value contains two
DATE-style endpoints: [["DD","MM","YYYY"],["DD","MM","YYYY"]]. Use empty
strings for missing day or month. If the second year is abbreviated in the
visible text, preserve that abbreviation in value, for example 1998/99 has
[["","","1998"],["","","99"]]. Do not include other digits in the sentence."""
    if category == "RATIO":
        if subgroup == "SCALE":
            return lead + """ Create natural sentences with colon-delimited map,
drawing, model, or technical scales such as 1:1000. The annotation type is
RATIO, subtype is SCALE, number is the complete ratio expression, and value is
["first","second"]. Ratios are scales, not date ranges. Do not include other
digits in the sentence."""
    if category == "COORDINATE":
        return lead + """ Create natural German sentences with geographic
latitude or longitude in degree-minute-second notation, such as 70°40'11" O or
46°57'08" N. The annotation type is COORDINATE. Number is the complete
coordinate expression including the direction letter when present. Value is
{"degrees":"digits","minutes":"digits or empty","seconds":"digits or empty",
"direction":"N|S|O|E|W or empty"}. Do not include other digits in the sentence."""
    if category == "NEGATIVE":
        formats = {
            "NUMBER_CARDINAL": 'number "-24", value "-24", type "NUMBER_CARDINAL"',
            "DECIMAL": 'number "-3,5", value ["-3","5"], type "DECIMAL"',
            "CURRENCY": 'number "CHF -12.40", value {"major":"-12","minor":"40","currency":"CHF"}, type "CURRENCY"',
        }[subgroup]
        return lead + f""" Generate negative {subgroup} values in many natural
contexts. Use an ASCII leading minus sign that semantically means negative, not
a hyphen, range, or subtraction operator. The annotation must follow this form:
{formats}, and must additionally contain "subtype":"NEGATIVE". For negative
currencies, vary CHF, EUR, and USD. Do not include other digits in the sentence.
The symbol already means minus: never place the written word "minus" directly
before the negative expression."""
    raise ValueError(f"Unsupported category: {category}")


_MONTH_NUMBERS = {
    "januar": 1, "jan": 1, "februar": 2, "feb": 2, "märz": 3, "maerz": 3,
    "mrz": 3, "april": 4, "apr": 4, "mai": 5, "juni": 6, "jun": 6,
    "juli": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sept": 9, "sep": 9, "oktober": 10, "okt": 10, "november": 11,
    "nov": 11, "dezember": 12, "dez": 12,
}
_ZIP_LOCALITIES = None


def _zip_localities() -> dict[str, set[str]]:
    global _ZIP_LOCALITIES
    if _ZIP_LOCALITIES is not None:
        return _ZIP_LOCALITIES
    localities: dict[str, set[str]] = {}
    with ZIP_SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            zip_code = str(row.get("PLZ", "")).strip()
            locality = str(row.get("Ortschaftsname", "")).strip()
            if re.fullmatch(r"[1-9]\d{3}", zip_code) and locality:
                localities.setdefault(zip_code, set()).add(locality)
    _ZIP_LOCALITIES = localities
    return localities


def _zip_prompt_pairs(area: str, subarea: str, count: int = 80) -> str:
    """Return a stable, context-specific window of verified prompt examples."""
    pairs = sorted(
        (zip_code, locality)
        for zip_code, localities in _zip_localities().items()
        for locality in localities
    )
    digest = hashlib.sha256(f"{area}\0{subarea}".encode()).digest()
    start = int.from_bytes(digest[:8], "big") % len(pairs)
    selected = [pairs[(start + offset) % len(pairs)] for offset in range(count)]
    return "; ".join(f"{zip_code} {locality}" for zip_code, locality in selected)


def _valid_date_parts(day: str, month: str, year: str) -> list[str] | None:
    if sum(bool(part) for part in (day, month, year)) < 2 or not month:
        return None
    try:
        month_int = int(month)
        if not 1 <= month_int <= 12:
            return None
        if day:
            validation_year = int(year) if len(year) == 4 else 2000
            if not 1 <= int(day) <= calendar.monthrange(validation_year, month_int)[1]:
                return None
        if year and len(year) not in {2, 4}:
            return None
    except (ValueError, OverflowError):
        return None
    return [day.zfill(2) if day else "", month.zfill(2), year]


def _date_value_from_number(number: str, subgroup: str) -> list[str] | None:
    compact = number.strip()
    iso = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", compact)
    if iso:
        year, month, day = iso.groups()
        return _valid_date_parts(day or "", month, year)

    if subgroup == "COMPACT" and re.fullmatch(r"\d{6}|\d{8}", compact):
        day, month, year = compact[:2], compact[2:4], compact[4:]
        return _valid_date_parts(day, month, year)

    written = re.fullmatch(
        r"(?:(\d{1,2})\.? ?\s*)?"
        r"(Januar|Jan\.?|Februar|Feb\.?|März|Maerz|Mrz\.?|April|Apr\.?|Mai|"
        r"Juni|Jun\.?|Juli|Jul\.?|August|Aug\.?|September|Sept?\.?|Sep\.?|"
        r"Oktober|Okt\.?|November|Nov\.?|Dezember|Dez\.?)"
        r"(?:\s+(\d{2}|\d{4}))?",
        compact,
        re.IGNORECASE,
    )
    if written:
        day, month_name, year = written.groups()
        month_key = month_name.rstrip(".").casefold()
        month = str(_MONTH_NUMBERS[month_key])
        return _valid_date_parts(day or "", month, year or "")

    numeric = re.fullmatch(r"(\d{1,4})([./-])(\d{1,4})(?:\2(\d{2}|\d{4}))?\.?", compact)
    if numeric:
        first, _, second, third = numeric.groups()
        if third:
            return _valid_date_parts(first, second, third)
        if subgroup in {"PARTIAL", "ISO"} and len(second) == 4:
            return _valid_date_parts("", first, second)
        return _valid_date_parts(first, second, "")
    return None


def _time_value_from_number(number: str, subgroup: str) -> list[str] | None:
    compact = re.sub(r"\s+Uhr$", "", number.strip(), flags=re.IGNORECASE).strip()
    hour = minute = second = ""
    separated = re.fullmatch(r"(\d{1,2})([:.])(\d{1,2})(?:\2(\d{1,2}))?", compact)
    h_notation = re.fullmatch(
        r"(\d{1,2})\s*h(?:\s*(\d{1,2})(?:\s*m(?:\s*(\d{1,2})\s*s?)?)?)?",
        compact,
        re.IGNORECASE,
    )
    if separated:
        hour, _, minute, second = separated.groups()
        second = second or ""
    elif h_notation:
        hour, minute, second = h_notation.groups()
        minute, second = minute or "", second or ""
    elif subgroup in {"COMPACT", "SECONDS"} and re.fullmatch(r"\d{3,6}", compact):
        if len(compact) == 3:
            hour, minute = compact[0], compact[1:]
        elif len(compact) == 4:
            hour, minute = compact[:2], compact[2:]
        elif len(compact) == 5:
            hour, minute, second = compact[0], compact[1:3], compact[3:]
        else:
            hour, minute, second = compact[:2], compact[2:4], compact[4:]
    elif subgroup == "HOUR_CONTEXT" and re.fullmatch(r"\d{1,2}", compact):
        hour = compact
    else:
        return None
    if not 0 <= int(hour) <= 23:
        return None
    if minute and not 0 <= int(minute) <= 59:
        return None
    if second and not 0 <= int(second) <= 59:
        return None
    return [hour.zfill(2), minute.zfill(2) if minute else "", second.zfill(2) if second else ""]


def validate(sample: dict, category: str, subgroup: str) -> dict | None:
    if not isinstance(sample, dict):
        return None
    text, numbers = sample.get("text"), sample.get("numbers")
    expected_count = range(2, 5) if category == "TIME" and subgroup == "MULTI" else (1,)
    if (
        not isinstance(text, str)
        or not isinstance(numbers, list)
        or len(numbers) not in expected_count
    ):
        return None
    if category == "TIME" and subgroup == "MULTI":
        normalized_numbers = []
        occupied: list[tuple[int, int]] = []
        for annotation in numbers:
            if not isinstance(annotation, dict):
                return None
            number = annotation.get("number")
            if not isinstance(number, str) or not number or text.count(number) != 1:
                return None
            normalized = _time_value_from_number(number, "MULTI")
            if normalized is None:
                return None
            start = text.find(number)
            end = start + len(number)
            if any(end > old_start and start < old_end for old_start, old_end in occupied):
                return None
            occupied.append((start, end))
            annotation["value"] = normalized
            annotation["type"] = "TIME"
            annotation["start"] = start
            annotation["end"] = end
            normalized_numbers.append(annotation)
        digit_positions = {match.start() for match in re.finditer(r"\d", text)}
        covered_positions = {
            position
            for start, end in occupied
            for position in range(start, end)
            if text[position].isdigit()
        }
        if digit_positions != covered_positions:
            return None
        return {
            "text": text.strip(),
            "text_quality": "Good",
            "numbers": sorted(normalized_numbers, key=lambda item: item["start"]),
        }
    annotation = numbers[0]
    if not isinstance(annotation, dict):
        return None
    number, value = annotation.get("number"), annotation.get("value")
    if not isinstance(number, str) or not number or text.count(number) != 1:
        return None
    annotation["start"] = text.find(number)
    annotation["end"] = annotation["start"] + len(number)
    if category in {"CURRENCY", "CURRENCY_BELOW_ONE"}:
        if category == "CURRENCY_BELOW_ONE":
            amount_match = re.search(r"(?<!\d)0[.,](\d{2})(?!\d)", number)
            marker_patterns = {
                "CHF": r"(?:\bCHF\b|\bFranken\b|\bFr\.)",
                "EUR": r"(?:\bEUR\b|\bEuro\b|€)",
                "USD": r"(?:\bUSD\b|\bDollar\b|\$)",
            }
            if not amount_match or not re.search(marker_patterns[subgroup], number, re.IGNORECASE):
                return None
            minor = amount_match.group(1)
            if int(minor) == 0:
                return None
            annotation["value"] = {
                "major": "0", "minor": minor, "currency": subgroup
            }
            annotation["type"] = "CURRENCY"
            return {"text": text.strip(), "text_quality": "Good", "numbers": [annotation]}

        # Gemini occasionally serializes this nested structured field once
        # more as a JSON string. Normalize it before validation so the saved
        # label always uses the project's actual currency schema.
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if not isinstance(value, dict) or not {"major", "minor", "currency"} <= set(value):
            return None
        currency = str(value["currency"]).upper()
        allowed = {"CHF", "EUR", "USD"} if subgroup != "OTHER" else {"GBP", "JPY", "CAD", "AUD", "SEK", "NOK", "DKK", "PLN", "TRY"}
        if currency not in allowed or not str(value["major"]).isdigit() or (value["minor"] and not str(value["minor"]).isdigit()):
            return None
        annotation["value"] = {"major": str(value["major"]), "minor": str(value["minor"]), "currency": currency}
    elif category == "IBAN":
        compact = number.replace(" ", "")
        if value != number or (subgroup == "CH" and not compact.startswith("CH")) or (subgroup == "OTHER" and compact.startswith("CH")):
            return None
    elif category == "PHONE":
        if not isinstance(value, str) or not re.fullmatch(r"\+?\d+", value):
            return None
        if subgroup == "CH" and not (number.strip().startswith("+41") or number.strip().startswith("0")):
            return None
    elif category == "PIN":
        if not re.fullmatch(r"\d{4,8}", number) or str(value) != number:
            return None
    elif category == "ZIP":
        localities = _zip_localities().get(number, set())
        text_folded = text.casefold()
        if (
            str(value) != number
            or not localities
            or not any(locality.casefold() in text_folded for locality in localities)
            or re.findall(r"\d+", text) != [number]
        ):
            return None
    elif category == "DATE":
        normalized = _date_value_from_number(number, subgroup)
        if normalized is None or len(re.findall(r"\d+", text)) != len(re.findall(r"\d+", number)):
            return None
        annotation["value"] = normalized
    elif category == "TIME":
        normalized = _time_value_from_number(number, subgroup)
        if normalized is None or len(re.findall(r"\d+", text)) != len(re.findall(r"\d+", number)):
            return None
        if subgroup == "HOUR_CONTEXT" and not re.search(
            r"\b(?:um|gegen|ab|bis|punkt|uhr|termin|beginn|start|abfahrt|ankunft|öffnung)\b",
            text,
            re.IGNORECASE,
        ):
            return None
        annotation["value"] = normalized
    elif category == "VERSION":
        if str(value) != number or not re.fullmatch(r"[vV]?\d+(?:\.\d+)+", number):
            return None
    elif category == "COORDINATE":
        normalized = _coordinate_value_from_number(number)
        if normalized is None:
            return None
        annotation["value"] = normalized
    elif category == "NUMBER_WITH_SUFFIX":
        if subgroup == "DECADE":
            match = re.fullmatch(r"([1-9]\d*|[1-9]\d?)er", number)
            if not match:
                return None
            numeric_part = match.group(1)
            annotation["value"] = {"number": numeric_part, "suffix": "er"}
            annotation["subtype"] = "DECADE"
            annotation["type"] = "NUMBER_WITH_SUFFIX"
            return {"text": text.strip(), "text_quality": "Good", "numbers": [annotation]}
        match = re.fullmatch(r"([1-9]\d*)([A-Za-z])", number)
        if not match:
            return None
        numeric_part, suffix = match.groups()
        # The visible compact expression is authoritative. Models sometimes
        # repeat the illustrative value literally or serialize it as JSON.
        annotation["value"] = {"number": numeric_part, "suffix": suffix}
        annotation.pop("subtype", None)
    elif category == "RANGE":
        if subgroup != "DATE":
            return None
        normalized = _date_range_value_from_number(number)
        if normalized is None:
            return None
        annotation["value"] = normalized
        annotation["subtype"] = "DATE"
    elif category == "RATIO":
        if subgroup != "SCALE":
            return None
        match = re.fullmatch(r"([1-9]\d*)\s*:\s*([1-9]\d*)", number)
        if not match:
            return None
        annotation["value"] = [match.group(1), match.group(2)]
        annotation["subtype"] = "SCALE"
    elif category == "NEGATIVE":
        if annotation.get("type") != subgroup or annotation.get("subtype") != "NEGATIVE":
            return None
        if number.count("-") != 1 or not re.search(r"-\d", number):
            return None
        if re.search(rf"\bminus\s+{re.escape(number)}", text, re.IGNORECASE):
            return None
        if subgroup == "NUMBER_CARDINAL":
            if not re.fullmatch(r"-[1-9]\d*", number):
                return None
            annotation["value"] = number
        elif subgroup == "DECIMAL":
            decimal_match = re.fullmatch(r"-([1-9]\d*)[,.](\d+)", number)
            if not decimal_match:
                return None
            annotation["value"] = [f"-{decimal_match.group(1)}", decimal_match.group(2)]
        elif subgroup == "CURRENCY":
            amount_match = re.search(r"-([1-9]\d*)(?:[.,](\d{1,2}))?", number)
            currency = next(
                (
                    code
                    for code, pattern in {
                        "CHF": r"(?:\bCHF\b|\bFranken\b|\bFr\.)",
                        "EUR": r"(?:\bEUR\b|\bEuro\b|€)",
                        "USD": r"(?:\bUSD\b|\bDollar\b|\$)",
                    }.items()
                    if re.search(pattern, number, re.IGNORECASE)
                ),
                None,
            )
            if not amount_match or currency is None:
                return None
            annotation["value"] = {
                "major": f"-{amount_match.group(1)}",
                "minor": amount_match.group(2) or "",
                "currency": currency,
            }
        # Do not accept a partial annotation that leaves a literal version
        # prefix behind as ``vxax`` during placeholder replacement.
        if not number.startswith(("v", "V")) and re.search(
            rf"[vV]{re.escape(number)}", text
        ):
            return None
    elif category in {"PRODUCT_ID", "MATH_EXPRESSION"} and str(value) != number:
        return None
    if category not in {"CURRENCY_BELOW_ONE", "NEGATIVE"}:
        annotation["type"] = category
    return {"text": text.strip(), "text_quality": "Good", "numbers": [annotation]}


def _date_range_value_from_number(number: str) -> list[list[str]] | None:
    compact = number.strip()
    year_match = re.fullmatch(r"(\d{4})\s*/\s*(\d{2}|\d{4})", compact)
    if year_match:
        return [["", "", year_match.group(1)], ["", "", year_match.group(2)]]
    full_date_match = re.fullmatch(
        r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})\s*/\s*(\d{1,2})\.(\d{1,2})\.(\d{2,4})",
        compact,
    )
    if full_date_match:
        d1, m1, y1, d2, m2, y2 = full_date_match.groups()
        return [[d1.zfill(2), m1.zfill(2), y1], [d2.zfill(2), m2.zfill(2), y2]]
    return None


def _coordinate_value_from_number(number: str) -> dict[str, str] | None:
    match = re.fullmatch(
        r"\s*(\d{1,3})\s*°\s*(?:(\d{1,2})\s*['′]\s*)?"
        r"(?:(\d{1,2})\s*(?:\"|″)\s*)?([NSEOWeow])?\s*",
        number,
    )
    if not match:
        return None
    degrees, minutes, seconds, direction = match.groups()
    if int(degrees) > 180:
        return None
    if minutes and int(minutes) > 59:
        return None
    if seconds and int(seconds) > 59:
        return None
    return {
        "degrees": degrees,
        "minutes": minutes or "",
        "seconds": seconds or "",
        "direction": (direction or "").upper(),
    }


def split_quota(count: int, contexts: list[tuple[str, str]]):
    """Distribute a quota as evenly as possible over semantic subareas."""
    base, remainder = divmod(count, len(contexts))
    for index, (area, subarea) in enumerate(contexts):
        assigned = base + (1 if index < remainder else 0)
        if assigned:
            yield area, subarea, assigned


def contexts_for_preset(preset: str, category: str) -> list[tuple[str, str]]:
    """Keep the small review set economical; use all subareas for the full run."""
    contexts = CONTEXTS[category]
    return contexts[:4] if preset == "test" else contexts


def quota_breakdown(
    quota: int, preset: str, category: str
) -> tuple[list[tuple[str, str, int]], int]:
    """Reserve a small share for unconstrained, but still validated, contexts."""
    exploratory = round(quota * EXPLORATORY_SHARE)
    planned = quota - exploratory
    return (
        list(split_quota(planned, contexts_for_preset(preset, category))),
        exploratory,
    )


def copy_checkpoint_excluding_categories(
    source: Path, destination: Path, categories: set[str]
) -> Counter:
    """Copy a JSONL checkpoint while removing complete synthetic categories."""
    removed = Counter()
    with source.open(encoding="utf-8") as input_handle, destination.open(
        "x", encoding="utf-8"
    ) as output_handle:
        for line_number, line in enumerate(input_handle, start=1):
            try:
                record = json.loads(line)
                category = str(record["synthetic_category"])
            except (KeyError, json.JSONDecodeError) as exc:
                destination.unlink(missing_ok=True)
                raise ValueError(f"Invalid checkpoint line {line_number}: {exc}") from exc
            if category in categories:
                removed[category] += 1
            else:
                output_handle.write(line)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="test")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a previously interrupted temporary checkpoint",
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=sorted({category for preset in PRESETS.values() for category, _ in preset}),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="atomically append a preset to an existing completed checkpoint",
    )
    parser.add_argument(
        "--replace-category",
        action="append",
        choices=sorted(CONTEXTS),
        help=(
            "with --append, remove every existing sample in this category before "
            "generating the selected preset; repeat for multiple categories"
        ),
    )
    args = parser.parse_args()
    require_compute_node("synthetic number-data generation")
    if args.batch_size < 1 or args.max_retries < 1:
        parser.error("batch size and retries must be positive")
    key = api_key()
    if not key:
        parser.error("OPENROUTER_API_KEY is required in the environment or project .env")

    selected = set(args.category) if args.category else None
    replacement_categories = set(args.replace_category or [])
    if replacement_categories and not args.append:
        parser.error("--replace-category requires --append")
    preset_categories = {category for category, _ in PRESETS[args.preset]}
    if not replacement_categories <= preset_categories:
        parser.error("Every --replace-category must be generated by the selected preset")
    run_name = args.run_name or f"synthetic_{args.preset}"
    output_path = OUTPUT_DIR / f"{run_name}_results_checkpoint.jsonl"
    temporary_path = output_path.with_suffix(".tmp")
    if output_path.exists() and not args.append:
        parser.error(f"Output exists: {output_path}. Choose another run name.")
    if args.append and not output_path.exists() and not temporary_path.exists():
        parser.error(f"Cannot append because output does not exist: {output_path}")
    if temporary_path.exists() and not args.resume:
        parser.error(
            f"Temporary output exists: {temporary_path}. Use --resume, remove it, "
            "or choose another run name."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.append and output_path.exists() and not temporary_path.exists():
        if replacement_categories:
            try:
                removed = copy_checkpoint_excluding_categories(
                    output_path, temporary_path, replacement_categories
                )
            except ValueError as exc:
                parser.error(str(exc))
            print(f"Copied existing checkpoint to replacement workspace: {temporary_path}")
            for category in sorted(replacement_categories):
                print(f"  removed {removed[category]} existing {category} samples")
        else:
            shutil.copyfile(output_path, temporary_path)
            print(f"Copied existing checkpoint to append workspace: {temporary_path}")
    seen, totals, completed = set(), Counter(), Counter()
    if temporary_path.exists():
        for line_number, line in enumerate(
            temporary_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                record = json.loads(line)
                sample = json.loads(record["response"])
                record_key = (
                    record["synthetic_category"],
                    record["synthetic_subgroup"],
                    record["synthetic_area"],
                    record["synthetic_subarea"],
                )
                text = sample["text"]
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                parser.error(f"Invalid checkpoint line {line_number}: {exc}")
            if text in seen:
                parser.error(f"Duplicate text in checkpoint line {line_number}")
            seen.add(text)
            totals[record_key[:2]] += 1
            completed[record_key] += 1
        print(f"Resuming {sum(totals.values())} validated samples from {temporary_path}")
    open_mode = "a" if temporary_path.exists() else "x"
    with temporary_path.open(open_mode, encoding="utf-8", buffering=1) as output:
        for (category, subgroup), quota in PRESETS[args.preset].items():
            if selected and category not in selected:
                continue
            planned_tasks, exploratory_target = quota_breakdown(
                quota, args.preset, category
            )
            if exploratory_target:
                planned_tasks.append(
                    ("exploratory", "model-selected plausible context", exploratory_target)
                )
            for area, subarea, target in planned_tasks:
                print(f"Generating {target} {category}/{subgroup} in {area}: {subarea}")
                task_key = (category, subgroup, area, subarea)
                accepted = completed[task_key]
                if accepted > target:
                    raise RuntimeError(
                        f"Checkpoint has {accepted}, above target {target}, for {task_key}"
                    )
                if accepted:
                    print(f"  resuming at {accepted}/{target}")
                while accepted < target:
                    requested = min(args.batch_size, target - accepted)
                    valid = []
                    for attempt in range(1, args.max_retries + 1):
                        try:
                            candidates = request_samples(key, category, subgroup, area, subarea, requested)
                        except RuntimeError as exc:
                            if "reason" in str(exc).lower() or attempt == args.max_retries:
                                raise
                            print(f"  retry {attempt}: {exc}", file=sys.stderr)
                            time.sleep(attempt * 2)
                            continue
                        valid.extend(
                            item
                            for item in (
                                validate(candidate, category, subgroup)
                                for candidate in candidates
                            )
                            if item is not None
                            and item["text"] not in seen
                            and item["text"] not in {accepted_item["text"] for accepted_item in valid}
                        )
                        if len(valid) >= requested:
                            break
                        print(f"  retry {attempt}: valid {len(valid)}/{requested}", file=sys.stderr)
                    if len(valid) < requested:
                        raise RuntimeError(f"Could not validate {requested} {category}/{subgroup} samples")
                    accepted_samples = valid[:requested]
                    for sample in accepted_samples:
                        totals[(category, subgroup)] += 1
                        record = {
                            "id": f"{run_name}_{category.lower()}_{subgroup.lower()}_{totals[(category, subgroup)]:06d}",
                            "response": json.dumps(sample, ensure_ascii=False),
                            "synthetic_category": category,
                            "synthetic_subgroup": subgroup,
                            "synthetic_area": area,
                            "synthetic_subarea": subarea,
                            "model": MODEL,
                        }
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    seen.update(sample["text"] for sample in accepted_samples)
                    accepted += len(accepted_samples)
                    completed[task_key] += len(accepted_samples)
                print(f"  {accepted}/{target}")
    temporary_path.replace(output_path)
    print(f"Wrote {sum(totals.values())} labeled samples to {output_path}")
    for key, count in sorted(totals.items()):
        print(f"  {key[0]}/{key[1]}: {count}")


if __name__ == "__main__":
    main()
