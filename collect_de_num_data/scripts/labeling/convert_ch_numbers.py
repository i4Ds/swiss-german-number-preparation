"""
convert_ch_numbers.py

Convert a single LLM-classified number dict to a selectable Swiss German form.
No re-detection (no spaCy, no HeidelTime) — just num2words dispatch.

Usage:
    from convert_ch_numbers import convert_number

    sg_form = convert_number(num_info_dict, dialect="ch_sg")
"""

import sys
import os
import re
from copy import deepcopy
from pathlib import Path

# Prefer an explicit checkout when configured, otherwise use the repository's
# pinned submodule. This keeps cluster-specific paths out of the source tree.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = Path(
    os.environ.get("NUM2WORDS_CH_ROOT", _PROJECT_ROOT / "num2words_CH")
).expanduser().resolve()
if not (_REPO_ROOT / "num2words" / "num2words_CH.py").is_file():
    raise ImportError(
        f"num2words_CH was not found at {_REPO_ROOT}. "
        "Run 'git submodule update --init --recursive' or set NUM2WORDS_CH_ROOT."
    )
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from num2words.num2words_CH import num2words  # noqa: E402

UNKNOWN = "unknown"
SUPPORTED_DIALECTS = ("ch_sg", "ch_bs", "ch_zh", "ch_lu", "ch_be")

# ---------------------------------------------------------------------------
# Label mappings: LLM output → Swiss dialect converter expectations
# ---------------------------------------------------------------------------

_DEKLINATION_MAP = {
    "stark":    "stark",
    "schwach":  "schwach",
    "gemischt": "gemischt",
    "strong":   "stark",
    "weak":     "schwach",
    "mixed":    "gemischt",
}

_GENUS_MAP = {
    "maskulin": "masc",
    "feminin":  "fem",
    "neutrum":  "neut",
    "masc":     "masc",
    "fem":      "fem",
    "neut":     "neut",
    "plur":     "plur",
}

_KASUS_MAP = {
    "nominativ": "nom",
    "akkusativ": "acc",
    "dativ":     "dat",
    "genitiv":   "gen",
    "nom": "nom",
    "acc": "acc",
    "dat": "dat",
    "gen": "gen",
}

_DEFAULT_DECLENSION = {"declension": "schwach", "gender": "masc", "case": "dat"}


def _map_declension(num_info: dict) -> dict:
    """Convert LLM declension fields to the shared Swiss dialect format."""
    decl = _DEFAULT_DECLENSION.copy()
    raw_dekl  = str(num_info.get("deklination", "")).lower()
    raw_genus = str(num_info.get("genus",       "")).lower()
    raw_kasus = str(num_info.get("kasus",       "")).lower()
    if raw_dekl  in _DEKLINATION_MAP:
        decl["declension"] = _DEKLINATION_MAP[raw_dekl]
    if raw_genus in _GENUS_MAP:
        decl["gender"] = _GENUS_MAP[raw_genus]
    if raw_kasus in _KASUS_MAP:
        decl["case"] = _KASUS_MAP[raw_kasus]
    return decl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_number(num_info: dict, dialect: str = "ch_sg") -> str:
    """
    Convert one LLM number dict to a Swiss German spoken form.

    Parameters
    ----------
    num_info : dict
        One entry from the LLM 'numbers' list, e.g.:
        {
            "number": "170",
            "value":  "170",
            "type":   "NUMBER_CARDINAL",
            ...
        }
    dialect : str
        num2words dialect key: ``ch_sg`` (default), ``ch_bs``, ``ch_zh``,
        ``ch_lu`` or ``ch_be``.

    Returns
    -------
    str  – spoken form, or "unknown" on failure / unsupported type.
    """
    if dialect not in SUPPORTED_DIALECTS:
        return UNKNOWN

    if str(num_info.get("subtype", "")).upper() == "NEGATIVE":
        positive_info = _positive_copy(num_info)
        positive_form = convert_number(positive_info, dialect=dialect)
        if not positive_form or positive_form == UNKNOWN:
            return UNKNOWN
        return f"minus {positive_form}"

    ntype  = num_info.get("type", "")
    value  = num_info.get("value")
    number = num_info.get("number", "")

    try:
        if ntype == "NUMBER_CARDINAL":
            return _cardinal(value, dialect)

        elif ntype == "NUMBER_ORDINAL":
            return _ordinal(value, num_info, dialect)

        elif ntype == "DATE":
            return _date(value, dialect)

        elif ntype == "DECIMAL":
            return _decimal(value, dialect)

        elif ntype == "TIME":
            return _time(value, dialect)

        elif ntype == "PHONE":
            return _phone(number, dialect)

        elif ntype == "PIN":
            return _identifier(number, dialect)

        elif ntype in {"IBAN", "ISBN", "PRODUCT_ID", "MODEL_NUMBER"}:
            return _identifier(number, dialect)

        elif ntype == "COORDINATE":
            return _coordinate(value, number, dialect)

        elif ntype == "NUMBER_WITH_SUFFIX":
            return _number_with_suffix(value, number, dialect)

        elif ntype == "CURRENCY":
            return _currency(value, number, dialect)

        elif ntype == "ZIP":
            return _zip_code(value, number, dialect)

        elif ntype == "VERSION":
            return _version(value, number, dialect)

        elif ntype == "MATH_EXPRESSION":
            return _math_expression(value, number, dialect)

        elif ntype == "ROMAN_NUMERAL":
            return _roman_numeral(value, num_info, dialect)

        elif ntype == "FRACTION":
            return _fraction(value, number, dialect)

        elif ntype == "MIXED_FRACTION":
            return _mixed_fraction(value, number, dialect)

        elif ntype == "RANGE":
            return _range(value, num_info, dialect)

        elif ntype == "RATIO":
            return _ratio(value, num_info, dialect)

        # All other types → unknown
        return UNKNOWN

    except Exception:
        return UNKNOWN

    # safety net — should never be reached, but prevents None leaking out
    return UNKNOWN


# ---------------------------------------------------------------------------
# Type handlers
# ---------------------------------------------------------------------------

def _cardinal(value, dialect: str) -> str:
    if isinstance(value, list):
        # e.g. percentage stored as ["5", ""] — take first non-empty part
        value = next((v for v in value if v and str(v).strip()), None)
    if value is None or str(value).strip() == "":
        return UNKNOWN
    val_str = str(value).strip()
    # Sanity: must be numeric (possibly with decimal comma/dot)
    _assert_numeric(val_str)
    return num2words(val_str, lang=dialect)


def _ordinal(value, num_info: dict, dialect: str) -> str:
    if isinstance(value, list):
        value = next((v for v in value if v and str(v).strip()), None)
    if value is None or str(value).strip() == "":
        return UNKNOWN
    val_str = str(value).strip()
    _assert_numeric(val_str)
    decl = _map_declension(num_info)
    return num2words(val_str, lang=dialect, ordinal=True, declension=decl)


def _date(value, dialect: str) -> str:
    """
    value can be:
      - a plain string "2014", "2014-10", "2014-10-03"
      - a list ["", "10", "2014"]  (day, month, year — some may be empty)
    """
    day, month, year = None, None, None

    if isinstance(value, list):
        # LLM stores dates as [day_or_empty, month_or_empty, year_or_empty]
        parts = [str(v).strip() if v else "" for v in value]
        if len(parts) >= 3:
            day   = int(parts[0]) if parts[0].isdigit() else None
            month = int(parts[1]) if parts[1].isdigit() else None
            year_s = parts[2]
            year  = int(year_s) if year_s.isdigit() else None
        elif len(parts) == 1 and parts[0].isdigit():
            year = int(parts[0])
    elif isinstance(value, str):
        v = value.strip()
        if len(v) == 4 and v.isdigit():
            year = int(v)
        else:
            # Try ISO-like "YYYY-MM-DD" or "YYYY-MM"
            parts = v.split("-")
            if len(parts) >= 1 and parts[0].isdigit() and len(parts[0]) == 4:
                year = int(parts[0])
            if len(parts) >= 2 and parts[1].isdigit():
                month = int(parts[1])
            if len(parts) >= 3 and parts[2].isdigit():
                day = int(parts[2])

    if day is None and month is None and year is None:
        return UNKNOWN

    spoken_parts = []
    if day is not None:
        spoken_parts.append(
            num2words(day, lang=dialect, ordinal=True, declension=_DEFAULT_DECLENSION)
        )
    if month is not None:
        spoken_parts.append(num2words(month, lang=dialect, to="month_dates"))
    if year is not None:
        y = str(year)
        if len(y) == 4 and int(y) <= 1999:
            if y[2:] == "00":
                spoken_parts.append(num2words(y[:2], lang=dialect) + "hundert")
            else:
                spoken_parts.append(
                    num2words(y[:2], lang=dialect) + " " + num2words(y[2:], lang=dialect)
                )
        else:
            spoken_parts.append(num2words(y, lang=dialect))

    return " ".join(spoken_parts)


def _decimal(value, dialect: str) -> str:
    if isinstance(value, list):
        # [integer_part, decimal_part]
        parts = [str(v).strip() if v is not None else "" for v in value]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            val_str = parts[0] + "," + parts[1]
        else:
            return UNKNOWN
    else:
        val_str = str(value).replace(".", ",")

    _assert_numeric(val_str.replace(",", "."))
    return num2words(val_str, lang=dialect)


def _time(value, dialect: str) -> str:
    if isinstance(value, list):
        parts = [str(v).strip() if v is not None else "" for v in value]
        hour   = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else None
        minute = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        second = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
    elif isinstance(value, dict):
        hour = _as_int(value.get("HOUR"))
        minute = _as_int(value.get("MINUTE"))
        second = _as_int(value.get("SECOND"))
    else:
        return UNKNOWN

    if hour is None:
        return UNKNOWN

    # Mirror logic from detect_convert_ch_numbers.py
    if minute is not None:
        if minute == 25 or minute >= 30:
            hour += 1
    if hour > 12:
        hour -= 12
    if hour == 0:
        hour = 12

    result = num2words(hour, to="hours", lang=dialect)
    if minute is not None and minute > 0:
        result = num2words(minute, to="minutes", lang=dialect) + " " + result
    if second is not None and second > 0:
        result += " " + num2words(second, lang=dialect)
        result += " " + num2words("sek", to="lookup", lang=dialect)
    return result


def _phone(number, dialect: str) -> str:
    """Read phone numbers digit by digit, preserving a leading plus."""
    if isinstance(number, str):
        raw_number = number.strip()
    else:
        return "UNKNOWN"

    cleaned_number = "".join(raw_number.split())
    parts = ["plus"] if cleaned_number.startswith("+") else []
    parts.extend(
        num2words(digit, lang=dialect)
        for digit in cleaned_number.lstrip("+")
        if digit.isdigit()
    )
    return " ".join(parts) if parts else UNKNOWN


_LETTER_NAMES = {
    "a": "a", "b": "be", "c": "ce", "d": "de", "e": "e",
    "f": "ef", "g": "ge", "h": "ha", "i": "i", "j": "jot",
    "k": "ka", "l": "el", "m": "em", "n": "en", "o": "o",
    "p": "pe", "q": "ku", "r": "er", "s": "es", "t": "te",
    "u": "u", "v": "vau", "w": "we", "x": "iks",
    "y": "ypsilon", "z": "zett", "ä": "ä", "ö": "ö", "ü": "ü",
    "ß": "doppel-es",
}

_IDENTIFIER_SYMBOL_NAMES = {
    ".": "punkt",
    ",": "komma",
    "?": "frogezeiche",
    "!": "ausrufezeichen",
    "/": "schrägstrich",
    "\\": "rückwärtsschrägstrich",
    "+": "plus",
    "-": "bindestrich",
    "_": "unterstrich",
    ":": "doppelpunkt",
    ";": "semikolon",
    "@": "at",
    "#": "raute",
}

_IDENTIFIER_SYMBOL_NAMES_CH_SG = {
    "?": "frogezeiche",
    "!": "usruefezeiche",
    "/": "schreegstrich",
    "\\": "rückwärtsschreegstrich",
    "_": "understrich",
    "@": "ät",
}

_SENTENCE_BOUNDARY_PUNCTUATION = frozenset(".,?!;:")


def _identifier(number, dialect: str) -> str:
    """Read identifier chunks and speak symbols internal to the identifier."""
    if isinstance(number, str):
        raw_identifier = number.strip()
    else:
        return "UNKNOWN"

    tokens = re.findall(r"[A-Za-zÄÖÜäöüß]+|\d+|[^\s]", raw_identifier)
    if not tokens:
        return UNKNOWN

    # An annotation can accidentally include punctuation belonging to the
    # surrounding sentence. Keep runs at either outer edge as punctuation;
    # the same characters between identifier components are spoken aloud.
    content_start = 0
    while (
        content_start < len(tokens)
        and tokens[content_start] in _SENTENCE_BOUNDARY_PUNCTUATION
    ):
        content_start += 1
    content_end = len(tokens)
    while (
        content_end > content_start
        and tokens[content_end - 1] in _SENTENCE_BOUNDARY_PUNCTUATION
    ):
        content_end -= 1

    symbol_names = _IDENTIFIER_SYMBOL_NAMES_CH_SG if dialect == "ch_sg" else {}
    parts = []
    for token in tokens[content_start:content_end]:
        if token.isdigit():
            parts.extend(_identifier_digit_parts(token, dialect))
        elif token.isalpha():
            parts.extend(_letter_name(char) for char in token)
        else:
            parts.append(
                symbol_names.get(token, _IDENTIFIER_SYMBOL_NAMES.get(token, token))
            )

    spoken_identifier = " ".join(parts)
    boundary_prefix = "".join(tokens[:content_start])
    boundary_suffix = "".join(tokens[content_end:])
    result = f"{boundary_prefix}{spoken_identifier}{boundary_suffix}"
    return result if result else UNKNOWN


def _identifier_digit_parts(token: str, dialect: str) -> list[str]:
    if len(token) <= 2 and not (len(token) > 1 and token.startswith("0")):
        return [num2words(token, lang=dialect)]
    return [num2words(digit, lang=dialect) for digit in token]


def _letter_name(char: str) -> str:
    return _LETTER_NAMES.get(char.casefold(), char.casefold())


def _number_with_suffix(value, number_text: str, dialect: str) -> str:
    """Write the numeric part while retaining its surrounding suffix text."""
    if isinstance(value, dict):
        numeric_part = str(value.get("number", "")).strip()
        suffix = str(value.get("suffix", "")).strip()
    else:
        match = re.search(r"\d+(?:[.,]\d+)?", str(number_text).strip())
        if not match:
            return UNKNOWN
        numeric_part = match.group()
        suffix = str(number_text).strip()[match.end():]

    if re.fullmatch(r"[1-9]\d*", numeric_part):
        spoken_number = num2words(numeric_part, lang=dialect)
    elif re.fullmatch(r"\d+[.,]\d+", numeric_part):
        spoken_number = _decimal(numeric_part, dialect)
    else:
        return UNKNOWN

    if not suffix and str(number_text).strip() == numeric_part:
        return spoken_number
    if suffix.casefold() == "er":
        if len(numeric_part) == 4 and int(numeric_part) <= 1999:
            return f"{_date(['', '', numeric_part], dialect)}er"
        return f"{spoken_number}er"
    if re.fullmatch(r"[A-Za-zÄÖÜäöü]", suffix):
        letter_name = _LETTER_NAMES.get(suffix.casefold())
        if letter_name is None:
            return UNKNOWN
        if suffix.isupper():
            letter_name = letter_name.capitalize()
        return f"{spoken_number} {letter_name}"

    # Mixed tokens such as ``7ner-Seilschaft``, ``JJ3`` and
    # ``NWS-9-Sänder`` retain every non-numeric source character. Only the
    # exact numeric component is written out.
    raw_number = str(number_text).strip()
    if numeric_part not in raw_number:
        return UNKNOWN
    return raw_number.replace(numeric_part, spoken_number, 1)


_DIRECTION_NAMES = {
    "N": "Nord",
    "S": "Süd",
    "E": "Ost",
    "O": "Ost",
    "W": "West",
}


def _coordinate(value, number_text: str, dialect: str) -> str:
    """Read geographic coordinates like 70°40'11" O."""
    degrees = minutes = seconds = direction = None
    if isinstance(value, dict):
        degrees = _as_int(value.get("degrees"))
        minutes = _as_int(value.get("minutes"))
        seconds = _as_int(value.get("seconds"))
        direction = str(value.get("direction", "")).strip().upper() or None
    else:
        match = re.fullmatch(
            r"\s*(\d{1,3})\s*°\s*(?:(\d{1,2})\s*['′]\s*)?"
            r"(?:(\d{1,2})\s*(?:\"|″)\s*)?([NSEOWeow])?\s*",
            str(number_text),
        )
        if not match:
            return UNKNOWN
        degrees = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else None
        seconds = int(match.group(3)) if match.group(3) else None
        direction = match.group(4).upper() if match.group(4) else None

    if degrees is None:
        return UNKNOWN
    parts = [num2words(degrees, lang=dialect), "Grad"]
    if minutes is not None:
        parts.extend([num2words(minutes, lang=dialect), "Minute"])
    if seconds is not None:
        parts.extend([num2words(seconds, lang=dialect), "Sekunde"])
    if direction:
        direction_name = _DIRECTION_NAMES.get(direction)
        if direction_name is None:
            return UNKNOWN
        parts.append(direction_name)
    return " ".join(parts)


_CURRENCY_NAMES = {
    "CHF": "Frankä",
    "EUR": "Euro",
    "USD": "Dollar",
    "GBP": "Pfund",
    "JPY": "Yen",
    "CAD": "Kanadischi Dollar",
    "AUD": "Australischi Dollar",
    "SEK": "Schwedischi Chrone",
    "NOK": "Norwegischi Chrone",
    "DKK": "Dänischi Chrone",
    "PLN": "Zloty",
    "TRY": "Türkischi Lira",
}

_CURRENCY_MINOR_NAMES = {
    "CHF": "Rappe",
    "EUR": "Cent",
    "USD": "Cent",
    "GBP": "Pence",
    "JPY": "Sen",
    "CAD": "Cent",
    "AUD": "Cent",
    "SEK": "Öre",
    "NOK": "Öre",
    "DKK": "Öre",
    "PLN": "Groszy",
    "TRY": "Kurus",
}


def _currency(value, number_text: str, dialect: str) -> str:
    """Read major and minor units directly, e.g. ``EUR 3.50`` → ``drü Euro füfzg``."""
    major, minor, currency = _currency_parts(value, number_text)
    if major is None:
        return UNKNOWN

    currency_name = _CURRENCY_NAMES.get(currency or "CHF")
    if currency_name is None:
        return UNKNOWN

    minor_is_nonzero = minor not in (None, "", "0", "00")
    if int(major) == 0 and minor_is_nonzero:
        minor_name = _CURRENCY_MINOR_NAMES.get(currency or "CHF")
        if minor_name is None:
            return UNKNOWN
        return f"{num2words(minor, lang=dialect)} {minor_name}"

    parts = [num2words(major, lang=dialect), currency_name]
    if minor_is_nonzero:
        parts.append(num2words(minor, lang=dialect))
    return " ".join(parts)


def _zip_code(value, number_text: str, dialect: str) -> str:
    """Read Swiss four-digit postal codes in the conventional grouped form."""
    raw_zip = str(value or number_text).strip()
    if not re.fullmatch(r"\d{4}", raw_zip):
        return UNKNOWN
    if raw_zip[1:] == "000":
        return num2words(raw_zip, lang=dialect)
    if raw_zip[2] == "0":
        return " ".join((
            num2words(raw_zip[:2], lang=dialect),
            num2words("0", lang=dialect),
            num2words(raw_zip[3], lang=dialect),
        ))
    return f"{num2words(raw_zip[:2], lang=dialect)} {num2words(raw_zip[2:], lang=dialect)}"


def _version(value, number_text: str, dialect: str) -> str:
    """Read dotted software versions component by component."""
    version_text = str(number_text or value).strip()
    raw_version = version_text.lstrip("vV")
    components = re.findall(r"\d+", raw_version)
    normalized = re.sub(r"\s*([._-])\s*", r"\1", version_text)
    version_match = re.search(r"\d+(?:[._-]\d+)+", normalized.rstrip("."))
    if not components or version_match is None:
        return UNKNOWN
    spoken_version = " punkt ".join(
        num2words(component, lang=dialect) for component in components
    )
    return f"v {spoken_version}" if version_text.startswith(("v", "V")) else spoken_version


_MATH_TOKEN_RE = re.compile(
    r"\d+(?:[,.]\d+)?|<=|>=|!=|==|\*\*|[A-Za-zÄÖÜäöüß]+|[+\-×*/÷=:<>^√%()\[\]{}]"
)
_MATH_WORDS = {
    "+": "plus", "-": "minus", "×": "mal", "*": "mal",
    "/": "geteilt durch", "÷": "geteilt durch", ":": "geteilt durch",
    "=": "isch gliich", "==": "isch gliich", "<": "isch chliner als",
    ">": "isch grösser als", "<=": "isch chliner oder gliich",
    ">=": "isch grösser oder gliich", "!=": "isch nöd gliich",
    "^": "hoch", "**": "hoch", "√": "Wurzle us", "%": "Prozent",
    "(": "Klammer uf", ")": "Klammer zue", "[": "eckigi Klammer uf",
    "]": "eckigi Klammer zue", "{": "gschwungni Klammer uf",
    "}": "gschwungni Klammer zue",
}
_MATH_FUNCTIONS = {"sin": "Sinus", "cos": "Cosinus", "tan": "Tangens", "log": "Logarithmus"}


def _math_expression(value, number_text: str, dialect: str) -> str:
    """Convert a restricted mathematical expression without evaluating it."""
    expression = str(value or number_text).strip()
    tokens = _MATH_TOKEN_RE.findall(expression)
    if not tokens or "".join(tokens).replace(",", ".") != expression.replace(" ", "").replace(",", "."):
        return UNKNOWN

    parts = []
    for token in tokens:
        if re.fullmatch(r"\d+(?:[,.]\d+)?", token):
            parts.append(num2words(token.replace(".", ","), lang=dialect))
        elif token in _MATH_WORDS:
            parts.append(_MATH_WORDS[token])
        else:
            parts.append(_MATH_FUNCTIONS.get(token.lower(), token))
    return " ".join(parts)


def _roman_numeral(value, num_info: dict, dialect: str) -> str:
    """Support legacy ROMAN_NUMERAL labels while newer data uses semantic types."""
    if isinstance(value, dict):
        arabic = value.get("arabic") or value.get("value")
        reading_type = value.get("reading_type", "NUMBER_CARDINAL")
    else:
        arabic = value
        reading_type = num_info.get("reading_type", "NUMBER_CARDINAL")
    if reading_type == "NUMBER_ORDINAL":
        return _ordinal(arabic, num_info, dialect)
    return _cardinal(arabic, dialect)


_FRACTION_DENOMINATORS = {
    "ch_sg": {2: "halb", 3: "Drittel", 4: "Viertel"},
    "ch_bs": {2: "halb", 3: "Drittel", 4: "Viertel"},
    "ch_zh": {2: "halb", 3: "Drittel", 4: "Viertel"},
}


def _fraction(value, number_text: str, dialect: str) -> str:
    """Handle simple fractions such as ``1/3`` as ``eis Drittel``."""
    mixed_match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*", str(number_text))
    if mixed_match is not None:
        whole, numerator, denominator = mixed_match.groups()
        return _mixed_fraction(
            {"whole": whole, "numerator": numerator, "denominator": denominator},
            number_text,
            dialect,
        )
    if (
        isinstance(value, list)
        and len(value) == 2
        and str(value[0]).isdigit()
        and re.fullmatch(r"\d+\s*/\s*\d+", str(value[1]))
    ):
        numerator, denominator = re.split(r"\s*/\s*", str(value[1]))
        return _mixed_fraction(
            {"whole": str(value[0]), "numerator": numerator, "denominator": denominator},
            number_text,
            dialect,
        )
    if isinstance(value, list) and len(value) == 2:
        num_part, den_part = str(value[0]).strip(), str(value[1]).strip()
        if num_part.isdigit() and den_part.isdigit():
            numerator = num2words(num_part, lang=dialect)
            if num_part == "1" and dialect == "ch_sg":
                numerator = "ei"
            denominator = _FRACTION_DENOMINATORS.get(dialect, {}).get(int(den_part))
            if denominator is None:
                denominator = num2words(
                    den_part,
                    lang=dialect,
                    ordinal=True,
                    declension=_DEFAULT_DECLENSION,
                )
            return f"{numerator} {denominator}"
    return UNKNOWN


def _mixed_fraction(value, number_text: str, dialect: str) -> str:
    """Handle mixed fractions; halves use the requested idiomatic ``ähalb``."""
    if not isinstance(value, dict):
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*", str(number_text))
        if match is None:
            return UNKNOWN
        whole, numerator, denominator = match.groups()
    else:
        whole = str(value.get("whole", "")).strip()
        numerator = str(value.get("numerator", "")).strip()
        denominator = str(value.get("denominator", "")).strip()
    if not all(part.isdigit() for part in (whole, numerator, denominator)):
        return UNKNOWN
    if numerator == "1" and denominator == "2":
        return f"{num2words(whole, lang=dialect)} ähalb"
    fraction = _fraction([numerator, denominator], number_text, dialect)
    if fraction == UNKNOWN:
        return UNKNOWN
    return f"{num2words(whole, lang=dialect)} {fraction}"


def _range(value, num_info: dict, dialect: str) -> str:
    """Convert a numeric range such as ["3", "5"] to "three to five"."""
    if not isinstance(value, list) or len(value) != 2:
        return UNKNOWN

    subtype = str(num_info.get("subtype", "")).upper()
    number_text = str(num_info.get("number", ""))
    slash_year_range = re.fullmatch(r"\s*\d{4}\s*/\s*(?:\d{2}|\d{4})\s*", number_text)
    if subtype in {"DATE", "DATE_RANGE", "YEAR", "YEAR_RANGE"} or slash_year_range:
        return _date_range(value, number_text, dialect)

    start, end = (str(part).strip() for part in value)
    if not start or not end:
        return UNKNOWN

    try:
        _assert_numeric(start.replace(",", "."))
        _assert_numeric(end.replace(",", "."))
        return f"{num2words(start, lang=dialect)} bis {num2words(end, lang=dialect)}"
    except Exception:
        return UNKNOWN


def _ratio(value, num_info: dict, dialect: str) -> str:
    """Convert a ratio such as ["1", "1000"] to "one to one thousand"."""
    if not isinstance(value, list) or len(value) != 2:
        return UNKNOWN

    first, second = (str(part).strip() for part in value)
    if not first or not second:
        return UNKNOWN

    try:
        _assert_numeric(first.replace(",", "."))
        _assert_numeric(second.replace(",", "."))
        return f"{num2words(first, lang=dialect)} zu {num2words(second, lang=dialect)}"
    except Exception:
        return UNKNOWN


def _date_range(value: list, number_text: str, dialect: str) -> str:
    """Use a compact reading for consecutive years and ``bis`` for real gaps."""
    start = _range_endpoint(value[0], dialect)
    end = _range_endpoint(value[1], dialect)
    if not start or not end:
        return UNKNOWN
    joiner = " bis "
    if "/" in number_text:
        start_year = _endpoint_year(value[0])
        end_year = _endpoint_year(value[1], reference=start_year)
        if start_year is not None and end_year == start_year + 1:
            joiner = " "
    return f"{start}{joiner}{end}"


def _endpoint_year(value, reference: int | None = None) -> int | None:
    if isinstance(value, dict) and "value" in value:
        return _endpoint_year(value["value"], reference=reference)
    if isinstance(value, list) and len(value) >= 3:
        text = str(value[2]).strip()
    elif isinstance(value, dict):
        text = str(value.get("year", "") or value.get("YYYY", "")).strip()
    else:
        text = str(value).strip()
    if not text.isdigit():
        return None
    if len(text) == 2 and reference is not None:
        candidate = (reference // 100) * 100 + int(text)
        if candidate < reference:
            candidate += 100
        return candidate
    return int(text) if len(text) == 4 else None


def _range_endpoint(value, dialect: str) -> str:
    if isinstance(value, dict) and "type" in value:
        spoken = convert_number(value, dialect=dialect)
        return "" if spoken == UNKNOWN else spoken
    return _date_endpoint(value, dialect)


def _date_endpoint(value, dialect: str) -> str:
    if isinstance(value, list):
        if len(value) >= 3:
            return _date(value, dialect)
        if len(value) == 1:
            return _date_endpoint(value[0], dialect)
        return UNKNOWN
    if isinstance(value, dict):
        day = str(value.get("day", "") or value.get("DD", "")).strip()
        month = str(value.get("month", "") or value.get("MM", "")).strip()
        year = str(value.get("year", "") or value.get("YYYY", "")).strip()
        return _date([day, month, year], dialect)
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit() and len(text) in {1, 2}:
        return num2words(text, lang=dialect)
    if text.isdigit() and len(text) == 4:
        return _date(["", "", text], dialect)
    return _date(text, dialect)


def _positive_copy(num_info: dict) -> dict:
    """Copy a negative annotation and remove its one semantic minus sign."""
    positive = deepcopy(num_info)
    positive.pop("subtype", None)
    positive["number"] = re.sub(r"[-−]", "", str(positive.get("number", "")), count=1)
    value = positive.get("value")

    if isinstance(value, str):
        positive["value"] = value.lstrip("-−")
    elif isinstance(value, list):
        cleaned = list(value)
        for index, part in enumerate(cleaned):
            if isinstance(part, str) and part.startswith(("-", "−")):
                cleaned[index] = part[1:]
                break
        positive["value"] = cleaned
    elif isinstance(value, dict):
        cleaned = dict(value)
        for key in ("major", "number", "integer_part", "numerator"):
            part = cleaned.get(key)
            if isinstance(part, str) and part.startswith(("-", "−")):
                cleaned[key] = part[1:]
                break
        positive["value"] = cleaned
    return positive


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_int(value):
    """Return an integer for an integer-like value, otherwise ``None``."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None




def _currency_parts(value, number_text: str):
    """Extract major/minor units and an ISO currency code from labeled data."""
    if isinstance(value, dict):
        major = str(value.get("major", "")).strip()
        minor = str(value.get("minor", "")).strip()
        currency = str(value.get("currency", "")).strip().upper()
        if major.isdigit() and (not minor or minor.isdigit()) and currency:
            return major, minor, currency

    if isinstance(value, list) and len(value) >= 2:
        major, minor = (str(part).strip() for part in value[:2])
        if major.isdigit() and (not minor or minor.isdigit()):
            return major, minor, _currency_from_text(number_text) or "CHF"

    if isinstance(value, (int, float)):
        value = str(value)
    if isinstance(value, str):
        candidate = value.strip()
    else:
        candidate = number_text.strip() if isinstance(number_text, str) else ""

    # Supports source forms such as "Fr. 3.40", "CHF 3,40", and "3.40 Fr.".
    match = re.search(r"(\d+)(?:[.,](\d{1,2}))?", candidate)
    if match:
        return match.group(1), match.group(2) or "", _currency_from_text(number_text)
    return None, None, None


def _currency_from_text(number_text: str) -> str | None:
    normalized = str(number_text).upper()
    currency_markers = {
        "CHF": "CHF", "FR.": "CHF", "SFR": "CHF", "₣": "CHF",
        "EUR": "EUR", "€": "EUR", "USD": "USD", "$": "USD",
        "GBP": "GBP", "£": "GBP", "JPY": "JPY", "¥": "JPY",
        "CAD": "CAD", "AUD": "AUD", "SEK": "SEK", "NOK": "NOK",
        "DKK": "DKK", "PLN": "PLN", "TRY": "TRY", "₺": "TRY",
    }
    return next((code for marker, code in currency_markers.items() if marker in normalized), None)

def _assert_numeric(val_str: str):
    """Raise ValueError if val_str cannot be parsed as a number."""
    cleaned = val_str.replace(",", ".").replace("'", "").replace(" ", "")
    float(cleaned)  # raises ValueError if not numeric


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        {"type": "NUMBER_CARDINAL", "value": "170",  "number": "170"},
        {"type": "NUMBER_CARDINAL", "value": "1000000000", "number": "eine Milliarde"},
        {"type": "NUMBER_ORDINAL",  "value": "20",   "number": "20.", "deklination": "stark", "genus": "maskulin", "kasus": "Dativ"},
        {"type": "DATE",            "value": ["", "10", "2014"], "number": "Oktober 2014"},
        {"type": "DATE",            "value": "1989",  "number": "1989"},
        {"type": "DECIMAL",         "value": ["1", "5"],  "number": "1,5"},
        {"type": "DECIMAL",         "value": ["6", "4"],  "number": "6,4"},
        {"type": "TIME",            "value": ["13", "15", ""], "number": "13.15"},
        {"type": "FRACTION",        "value": ["2", "3"],  "number": "zwei Drittel"},
        {"type": "OTHER_NUMERIC",   "value": ["5", ""],   "number": "5%"},
    ]
    for s in samples:
        result = convert_number(s)
        print(f"  {s['number']:<20} ({s['type']:<20}) → {result}")
