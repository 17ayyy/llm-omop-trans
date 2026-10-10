"""Validate all fourteen medical attribute families deterministically."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Final


class ValidationStatus(StrEnum):
    """Represent the aggregate deterministic validation outcome."""

    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"


class CheckStatus(StrEnum):
    """Represent one attribute-level comparison outcome."""

    PASS = "pass"
    REVIEW = "review"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class CandidateAttribute(StrEnum):
    """Identify each independently audited medical attribute family."""

    NUMERIC_VALUE = "numeric_value"
    UNIT = "unit"
    POLARITY_NEGATION = "polarity_negation"
    LATERALITY = "laterality"
    DOSAGE_CONCENTRATION = "dosage_concentration"
    DOSE_FORM = "dose_form"
    ADMINISTRATION_ROUTE = "administration_route"
    ANATOMICAL_SITE = "anatomical_site"
    SEX = "sex"
    TIME_FREQUENCY = "time_frequency"
    BRAND = "brand"
    PACKAGE = "package"
    GENE_SYMBOL = "gene_symbol"
    HGVS = "hgvs"


@dataclass(frozen=True, slots=True)
class ValidationEvidence:
    """Store observable text and deterministic normalized values."""

    source_matches: tuple[str, ...]
    candidate_matches: tuple[str, ...]
    normalized_source: tuple[str, ...]
    normalized_candidate: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable evidence mapping."""
        return {
            "candidate_matches": list(self.candidate_matches),
            "normalized_candidate": list(self.normalized_candidate),
            "normalized_source": list(self.normalized_source),
            "source_matches": list(self.source_matches),
        }


@dataclass(frozen=True, slots=True)
class AttributeCheck:
    """Describe one deterministic critical-attribute comparison."""

    attribute: CandidateAttribute
    status: CheckStatus
    flags: tuple[str, ...]
    evidence: ValidationEvidence

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable check mapping."""
        return {
            "attribute": self.attribute.value,
            "evidence": self.evidence.as_dict(),
            "flags": list(self.flags),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class CandidateValidation:
    """Contain the aggregate status and all auditable checks."""

    validation_status: ValidationStatus
    flags: tuple[str, ...]
    checks: tuple[AttributeCheck, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable validation mapping."""
        return {
            "checks": [check.as_dict() for check in self.checks],
            "flags": list(self.flags),
            "validation_status": self.validation_status.value,
        }


@dataclass(frozen=True, slots=True)
class _Extracted:
    """Hold internal normalized values and their observable evidence."""

    values: tuple[str, ...] = ()
    matches: tuple[str, ...] = ()
    review_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _QuantityExtraction:
    """Hold normalized quantities and spans consumed by number checks."""

    extracted: _Extracted
    dimensions: _Extracted
    spans: tuple[tuple[int, int], ...]


type _PatternMap = tuple[tuple[str, re.Pattern[str]], ...]


_NUMBER_TEXT: Final = r"(?:\d+(?:\.\d+)?|\.\d+)"
_PLAIN_NUMBER_RE: Final = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<first>{_NUMBER_TEXT})"
    rf"(?:\s*(?:-|–|—|~|至|to)\s*(?P<second>{_NUMBER_TEXT}))?"
    r"(?![A-Za-z0-9_.])",
    re.IGNORECASE,
)

_UNIT_ALIASES: Final[dict[str, tuple[str, Decimal]]] = {
    "%": ("percent", Decimal("1")),
    "百分比": ("percent", Decimal("1")),
    "bpm": ("rate", Decimal("1")),
    "beats/min": ("rate", Decimal("1")),
    "次/分": ("rate", Decimal("1")),
    "cfu": ("count", Decimal("1")),
    "dl": ("volume", Decimal("100")),
    "g": ("mass", Decimal("1000")),
    "iu": ("international_unit", Decimal("1")),
    "kg": ("mass", Decimal("1000000")),
    "l": ("volume", Decimal("1000")),
    "meq": ("equivalent", Decimal("1")),
    "mg": ("mass", Decimal("1")),
    "ml": ("volume", Decimal("1")),
    "mmhg": ("pressure", Decimal("1")),
    "mmol": ("substance", Decimal("1")),
    "mol": ("substance", Decimal("1000")),
    "ng": ("mass", Decimal("0.000001")),
    "nmol": ("substance", Decimal("0.000001")),
    "pg": ("mass", Decimal("0.000000001")),
    "ug": ("mass", Decimal("0.001")),
    "ul": ("volume", Decimal("0.001")),
    "umol": ("substance", Decimal("0.001")),
    "克": ("mass", Decimal("1000")),
    "分升": ("volume", Decimal("100")),
    "国际单位": ("international_unit", Decimal("1")),
    "微克": ("mass", Decimal("0.001")),
    "微升": ("volume", Decimal("0.001")),
    "微摩尔": ("substance", Decimal("0.001")),
    "毫克": ("mass", Decimal("1")),
    "毫升": ("volume", Decimal("1")),
    "毫米汞柱": ("pressure", Decimal("1")),
    "毫摩尔": ("substance", Decimal("1")),
    "摩尔": ("substance", Decimal("1000")),
    "纳克": ("mass", Decimal("0.000001")),
    "纳摩尔": ("substance", Decimal("0.000001")),
    "皮克": ("mass", Decimal("0.000000001")),
    "升": ("volume", Decimal("1000")),
}

_UNIT_TOKEN_TEXT: Final = (
    "(?:"
    + "|".join(
        re.escape(alias)
        for alias in sorted(_UNIT_ALIASES, key=len, reverse=True)
        if alias != "%"
    )
    + r"|%|mcg|μg|µg|μl|µl|μmol|µmol)"
)

_QUANTITY_RE: Final = re.compile(
    rf"(?P<number>{_NUMBER_TEXT})\s*(?P<unit>{_UNIT_TOKEN_TEXT})"
    rf"(?:\s*(?:/|per|每)\s*(?:(?P<denominator>{_NUMBER_TEXT})\s*)?"
    rf"(?P<denominator_unit>{_UNIT_TOKEN_TEXT}))?",
    re.IGNORECASE,
)
_UNIT_RE: Final = re.compile(_UNIT_TOKEN_TEXT, re.IGNORECASE)


def _compile_patterns(
    definitions: tuple[tuple[str, str], ...],
) -> _PatternMap:
    """Compile a stable canonical-value pattern mapping."""
    return tuple(
        (canonical, re.compile(pattern, re.IGNORECASE))
        for canonical, pattern in definitions
    )


_DOSE_FORM_PATTERNS: Final[_PatternMap] = _compile_patterns(
    (
        ("tablet", r"\btablets?\b|片剂|片(?=$|[\s,，;；])"),
        ("capsule", r"\bcapsules?\b|胶囊剂?|胶囊"),
        (
            "solution",
            r"\bsolutions?\b(?!\s+for\s+injection\b)|溶液剂?|溶液",
        ),
        ("suspension", r"\bsuspensions?\b|混悬剂|混悬液"),
        ("injection", r"\binjections?\b|\binjectable\b|注射剂|注射液"),
        ("cream", r"\bcreams?\b|乳膏剂?|乳膏"),
        ("ointment", r"\bointments?\b|软膏剂?|软膏"),
        ("gel", r"\bgels?\b|凝胶剂?|凝胶"),
        ("patch", r"\bpatch(?:es)?\b|贴剂"),
        ("spray", r"\bsprays?\b|喷雾剂"),
        ("drops", r"\bdrops?\b|滴剂|滴眼液|滴耳液"),
        ("suppository", r"\bsuppositor(?:y|ies)\b|栓剂"),
        ("powder", r"\bpowders?\b|粉剂|散剂|粉末"),
        ("granules", r"\bgranules?\b|颗粒剂"),
        ("syrup", r"\bsyrups?\b|糖浆剂?|糖浆"),
        ("lozenge", r"\blozenges?\b|含片"),
        ("implant", r"\bimplants?\b|植入剂"),
    ),
)

_ROUTE_PATTERNS: Final[_PatternMap] = _compile_patterns(
    (
        ("oral", r"\boral(?:ly)?\b|\bby mouth\b|口服"),
        ("intravenous", r"\bintravenous(?:ly)?\b|\biv\b|静脉(?:给药|注射)?"),
        (
            "intramuscular",
            r"\bintramuscular(?:ly)?\b|\bim\b|肌内(?:注射)?|肌肉注射",
        ),
        ("subcutaneous", r"\bsubcutaneous(?:ly)?\b|\bsc\b|皮下(?:注射)?"),
        ("topical", r"\btopical(?:ly)?\b|局部(?:外用|给药)|外用"),
        ("inhalation", r"\binhal(?:ed|ation)\b|吸入(?:给药)?"),
        ("intranasal", r"\bintranasal(?:ly)?\b|鼻内(?:给药)?"),
        ("ophthalmic", r"\bophthalmic\b|眼用|滴眼"),
        ("otic", r"\botic\b|耳用|滴耳"),
        ("rectal", r"\brectal(?:ly)?\b|直肠(?:给药)?"),
        ("vaginal", r"\bvaginal(?:ly)?\b|阴道(?:给药)?"),
        (
            "transdermal",
            r"\btransdermal(?:ly)?\b|透皮|"
            r"经皮(?!入路|内镜|途径|穿刺|手术|置入)(?:给药)?",
        ),
        ("sublingual", r"\bsublingual(?:ly)?\b|舌下(?:给药)?"),
        ("buccal", r"\bbuccal(?:ly)?\b|颊黏膜(?:给药)?"),
        ("intraarticular", r"\bintra-?articular\b|关节腔内(?:给药)?"),
        ("intrathecal", r"\bintrathecal(?:ly)?\b|鞘内(?:给药)?"),
    ),
)

_ANATOMY_PATTERNS: Final[_PatternMap] = _compile_patterns(
    (
        ("lymph_node", r"\blymph nodes?\b|淋巴结"),
        ("thyroid", r"\bthyroid\b|甲状腺"),
        ("adrenal", r"\badrenal\b|肾上腺"),
        ("pancreas", r"\bpancrea(?:s|tic)\b|胰腺"),
        ("prostate", r"\bprostat(?:e|ic)\b|前列腺"),
        ("bladder", r"\bbladder\b|膀胱"),
        ("uterus", r"\buter(?:us|ine)\b|子宫"),
        ("ovary", r"\bovar(?:y|ian)\b|卵巢"),
        ("breast", r"\bbreasts?\b|乳腺|乳房"),
        ("heart", r"\bheart\b|\bcardiac\b|心脏"),
        ("lung", r"\blungs?\b|\bpulmonary\b|肺部|肺脏|肺"),
        ("liver", r"\bliver\b|\bhepatic\b|肝脏|肝(?!素)"),
        ("kidney", r"\bkidneys?\b|\brenal\b|肾脏|肾"),
        ("brain", r"\bbrain\b|\bcerebral\b|脑部|脑"),
        ("stomach", r"\bstomach\b|\bgastric\b|胃部|胃"),
        ("colon", r"\bcolon(?:ic)?\b|结肠"),
        ("rectum", r"\brect(?:um|al)\b|直肠"),
        ("spleen", r"\bspleen\b|\bsplenic\b|脾脏|脾"),
        ("skin", r"\bskin\b|\bcutaneous\b|皮肤"),
        ("bone", r"\bbones?\b|\bosseous\b|骨骼|骨"),
        ("muscle", r"\bmuscles?\b|\bmuscular\b|肌肉"),
        ("eye", r"\beyes?\b|\bocular\b|眼部|眼"),
        ("ear", r"\bears?\b|耳部|耳"),
        ("nose", r"\bnose\b|\bnasal\b|鼻部|鼻"),
        ("pharynx", r"\bpharyn(?:x|geal)\b|咽部|咽"),
        ("larynx", r"\blaryn(?:x|geal)\b|喉部|喉"),
        ("spine", r"\bspine\b|\bspinal\b|脊柱"),
        ("shoulder", r"\bshoulders?\b|肩部|肩"),
        ("elbow", r"\belbows?\b|肘部|肘"),
        ("wrist", r"\bwrists?\b|腕部|腕"),
        ("hip", r"\bhips?\b|髋部|髋"),
        ("knee", r"\bknees?\b|膝部|膝"),
        ("ankle", r"\bankles?\b|踝部|踝"),
        ("chest", r"\bchest\b|\bthoracic\b|胸部|胸腔"),
        ("abdomen", r"\babdom(?:en|inal)\b|腹部|腹腔"),
        ("pelvis", r"\bpelv(?:is|ic)\b|盆腔|骨盆"),
    ),
)
_AMBIGUOUS_ANATOMY_RE: Final = re.compile(
    r"\bcervical\b|\bcervix\b|宫颈|子宫颈|颈部",
    re.IGNORECASE,
)

_SEX_PATTERNS: Final[_PatternMap] = _compile_patterns(
    (
        ("male", r"\bmales?\b|\bmen\b|男性|男(?!女)"),
        ("female", r"\bfemales?\b|\bwomen\b|女性|女"),
        ("intersex", r"\bintersex\b|间性|双性"),
    ),
)

_PACKAGE_PATTERNS: Final[_PatternMap] = _compile_patterns(
    (
        ("pack", r"\bpacks?\b|\bpackages?\b|包装|(?<=\d)\s*装"),
        ("box", r"\bbox(?:es)?\b|盒装|盒"),
        ("bottle", r"\bbottles?\b|瓶装|瓶"),
        ("vial", r"\bvials?\b|西林瓶"),
        ("ampoule", r"\bampoules?\b|\bampules?\b|安瓿"),
        ("blister", r"\bblisters?\b|泡罩|板装"),
        ("syringe", r"\bsyringes?\b|预充式注射器|注射器"),
        ("tube", r"\btubes?\b|管装"),
        ("bag", r"\bbags?\b|袋装|袋"),
    ),
)

_POLARITY_NEGATIVE_RE: Final = re.compile(
    r"\bnegative\b|\bnon[- ]?reactive\b|\bnot\s+detected\b|"
    r"\bno\s+evidence\s+of\b|\bwithout\b|\babsence\s+of\b|"
    # ``无菌`` is the standard Chinese equivalent of "sterile", not a
    # semantic negation.  Keep generic ``无`` detection for true additions,
    # while excluding this lexicalized dm+d device/drug attribute.
    r"阴性|未检出|未见|缺如|无(?!菌)",
    re.IGNORECASE,
)
_POLARITY_POSITIVE_RE: Final = re.compile(
    r"\bpositive\b|\breactive\b|\bdetected\b|阳性|已检出",
    re.IGNORECASE,
)

_LATERALITY_PATTERNS: Final[_PatternMap] = _compile_patterns(
    (
        ("bilateral", r"\bbilateral(?:ly)?\b|双侧|两侧"),
        ("unilateral", r"\bunilateral(?:ly)?\b|单侧"),
        ("ipsilateral", r"\bipsilateral(?:ly)?\b|同侧"),
        ("contralateral", r"\bcontralateral(?:ly)?\b|对侧"),
        ("left", r"\bleft\b|左侧|左(?!旋|氧)"),
        ("right", r"\bright\b|右侧|右(?!旋)"),
    ),
)

_FREQUENCY_PATTERNS: Final[_PatternMap] = _compile_patterns(
    (
        ("daily:1", r"\bonce daily\b|\bq\.?d\.?\b|每日一次|每天一次"),
        ("daily:2", r"\btwice daily\b|\bb\.?i\.?d\.?\b|每日两次|每天两次"),
        (
            "daily:3",
            r"\bthree times daily\b|\bt\.?i\.?d\.?\b|每日三次|每天三次",
        ),
        ("daily:4", r"\bfour times daily\b|\bq\.?i\.?d\.?\b|每日四次|每天四次"),
        ("weekly", r"\bweekly\b|每周"),
        ("monthly", r"\bmonthly\b|每月"),
        ("as_needed", r"\bas needed\b|\bp\.?r\.?n\.?\b|按需"),
        ("continuous", r"\bcontinuous(?:ly)?\b|持续(?:给药)?"),
        ("acute", r"\bacute\b|急性"),
        ("chronic", r"\bchronic\b|慢性"),
        ("morning", r"\bin the morning\b|\bmorning\b|早晨|晨间"),
        ("evening", r"\bin the evening\b|\bevening\b|晚间|傍晚"),
        ("bedtime", r"\bat bedtime\b|\bbedtime\b|睡前"),
        ("before_meals", r"\bbefore meals?\b|餐前|饭前"),
        ("after_meals", r"\bafter meals?\b|餐后|饭后"),
    ),
)
_EVERY_INTERVAL_RE: Final = re.compile(
    rf"\bevery\s+(?P<number>{_NUMBER_TEXT})\s*"
    r"(?P<unit>hours?|days?|weeks?|months?|years?)\b|"
    rf"\bq(?P<qnumber>{_NUMBER_TEXT})(?P<qunit>h|d|w)\b|"
    rf"每(?:隔)?\s*(?P<cnumber>{_NUMBER_TEXT})\s*"
    r"(?P<cunit>小时|天|日|周|月|年)",
    re.IGNORECASE,
)

_BRACKET_RE: Final = re.compile(r"[\[【](?P<brand>[^\]】]{1,80})[\]】]")
_BRAND_MARKER_RE: Final = re.compile(
    r"\bbrand(?:ed)?\s*[:：]?\s*(?P<en>[A-Za-z][A-Za-z0-9 .'-]{1,50})|"
    r"品牌\s*[:：]?\s*(?P<zh>[\u4e00-\u9fffA-Za-z0-9 .'-]{1,50})",
    re.IGNORECASE,
)

_HGVS_RE: Final = re.compile(
    r"(?:(?:N[CMRPTG]|LRG)_\d+(?:\.\d+)?:)?"
    r"(?:c|g|m|n|o|p|r)\."
    r"(?:\[[^\]\s,，；]+\]|[A-Za-z0-9_*+?>=().:-]+)",
)
_GENE_WITH_DIGIT_RE: Final = re.compile(
    r"(?<![A-Za-z0-9])(?P<gene>[A-Z][A-Z0-9-]{1,14}\d[A-Z0-9-]*)"
    r"(?![A-Za-z0-9])"
)
_GENE_CONTEXT_RE: Final = re.compile(
    r"(?<![A-Za-z0-9])(?P<gene>[A-Z][A-Z0-9-]{2,14})"
    r"(?=\s+(?:gene|mutation|variant|expression)\b)"
)
_GENE_STOPWORDS: Final = frozenset(
    {"B12", "COVID19", "H1N1", "H5N1", "ICD10", "PHQ9"}
)


def _normalize_text(value: str) -> str:
    """Normalize Unicode width and surrounding whitespace for matching."""
    return unicodedata.normalize("NFKC", value).strip()


def _normalize_decimal(value: str) -> str:
    """Return a stable non-exponent decimal representation."""
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"Invalid numeric token: {value!r}") from error
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _unit_spec(token: str) -> tuple[str, Decimal]:
    """Return the dimension and base-unit multiplier for a unit token."""
    normalized = token.casefold().replace("μ", "u").replace("µ", "u")
    aliases = {
        "mcg": "ug",
        "ul": "ul",
        "umol": "umol",
    }
    normalized = aliases.get(normalized, normalized)
    return _UNIT_ALIASES[normalized]


def _overlaps(
    span: tuple[int, int],
    occupied: tuple[tuple[int, int], ...],
) -> bool:
    """Return whether a span overlaps any already consumed text."""
    return any(span[0] < end and start < span[1] for start, end in occupied)


def _extract_quantities(text: str) -> _QuantityExtraction:
    """Extract dosage, concentration, and measured quantity expressions."""
    values: list[str] = []
    matches: list[str] = []
    dimensions: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in _QUANTITY_RE.finditer(text):
        numerator_dimension, numerator_scale = _unit_spec(match["unit"])
        numerator = Decimal(match["number"]) * numerator_scale
        value = f"{numerator_dimension}:{_normalize_decimal(str(numerator))}"
        dimensions.append(numerator_dimension)
        denominator_unit = match["denominator_unit"]
        if denominator_unit is not None:
            denominator_dimension, denominator_scale = _unit_spec(
                denominator_unit,
            )
            denominator_text = match["denominator"] or "1"
            denominator = Decimal(denominator_text) * denominator_scale
            value += (
                f"/{denominator_dimension}:"
                f"{_normalize_decimal(str(denominator))}"
            )
            dimensions.append(denominator_dimension)
        values.append(value)
        matches.append(match.group(0))
        spans.append(match.span())
    return _QuantityExtraction(
        extracted=_Extracted(tuple(sorted(values)), tuple(matches)),
        dimensions=_Extracted(
            tuple(sorted(set(dimensions))),
            tuple(matches),
        ),
        spans=tuple(spans),
    )


def _extract_numbers(
    text: str,
    excluded_spans: tuple[tuple[int, int], ...],
) -> _Extracted:
    """Extract non-quantity numbers, preserving ranges deterministically."""
    values: list[str] = []
    matches: list[str] = []
    for match in _PLAIN_NUMBER_RE.finditer(text):
        if _overlaps(match.span(), excluded_spans):
            continue
        first = _normalize_decimal(match["first"])
        second = match["second"]
        value = first
        if second is not None:
            value += "-" + _normalize_decimal(second)
        values.append(value)
        matches.append(match.group(0))
    return _Extracted(tuple(sorted(values)), tuple(matches))


def _extract_pattern_map(text: str, patterns: _PatternMap) -> _Extracted:
    """Extract canonical values from a curated bilingual pattern map."""
    values: set[str] = set()
    matches: list[str] = []
    for canonical, pattern in patterns:
        for match in pattern.finditer(text):
            values.add(canonical)
            matches.append(match.group(0))
    return _Extracted(tuple(sorted(values)), tuple(matches))


def _extract_polarity(text: str) -> _Extracted:
    """Extract high-precision positive and negative semantics."""
    values: set[str] = set()
    matches: list[str] = []
    negative_spans: list[tuple[int, int]] = []
    for match in _POLARITY_NEGATIVE_RE.finditer(text):
        values.add("negative")
        matches.append(match.group(0))
        negative_spans.append(match.span())
    occupied = tuple(negative_spans)
    for match in _POLARITY_POSITIVE_RE.finditer(text):
        if _overlaps(match.span(), occupied):
            continue
        values.add("positive")
        matches.append(match.group(0))
    return _Extracted(tuple(sorted(values)), tuple(matches))


def _extract_laterality(text: str) -> _Extracted:
    """Extract and normalize side qualifiers."""
    extracted = _extract_pattern_map(text, _LATERALITY_PATTERNS)
    values = set(extracted.values)
    if "left" in values and "right" in values:
        values.difference_update({"left", "right"})
        values.add("bilateral")
    return _Extracted(tuple(sorted(values)), extracted.matches)


def _extract_anatomy(text: str) -> _Extracted:
    """Extract high-precision sites and flag inherently ambiguous wording."""
    extracted = _extract_pattern_map(text, _ANATOMY_PATTERNS)
    ambiguous = tuple(
        match.group(0) for match in _AMBIGUOUS_ANATOMY_RE.finditer(text)
    )
    flags = ("anatomical_site_ambiguous",) if ambiguous else ()
    return _Extracted(
        extracted.values,
        (*extracted.matches, *ambiguous),
        flags,
    )


def _extract_time_frequency(text: str) -> _Extracted:
    """Extract named and numeric temporal/frequency qualifiers."""
    named = _extract_pattern_map(text, _FREQUENCY_PATTERNS)
    values = set(named.values)
    matches = list(named.matches)
    unit_map = {
        "d": "day",
        "day": "day",
        "days": "day",
        "h": "hour",
        "hour": "hour",
        "hours": "hour",
        "month": "month",
        "months": "month",
        "w": "week",
        "week": "week",
        "weeks": "week",
        "year": "year",
        "years": "year",
        "天": "day",
        "年": "year",
        "日": "day",
        "小时": "hour",
        "月": "month",
        "周": "week",
    }
    for match in _EVERY_INTERVAL_RE.finditer(text):
        number = match["number"] or match["qnumber"] or match["cnumber"]
        unit = match["unit"] or match["qunit"] or match["cunit"]
        assert number is not None and unit is not None
        canonical = unit_map[unit.casefold()]
        values.add(f"every:{_normalize_decimal(number)}:{canonical}")
        matches.append(match.group(0))
    return _Extracted(tuple(sorted(values)), tuple(matches))


def _normalize_brand(value: str) -> str:
    """Normalize spacing and case without translating a brand."""
    return " ".join(value.casefold().split()).strip(" .'-")


def _extract_brands(text: str) -> _Extracted:
    """Extract explicit bracketed or labelled brand evidence only."""
    values: set[str] = set()
    matches: list[str] = []
    for match in _BRACKET_RE.finditer(text):
        brand = _normalize_brand(match["brand"])
        if brand and not re.fullmatch(r"[\d\s.,%/-]+", brand):
            values.add(brand)
            matches.append(match.group(0))
    for match in _BRAND_MARKER_RE.finditer(text):
        brand_text = match["en"] or match["zh"]
        assert brand_text is not None
        brand = _normalize_brand(brand_text)
        if brand:
            values.add(brand)
            matches.append(match.group(0))
    return _Extracted(tuple(sorted(values)), tuple(matches))


def _extract_hgvs(text: str) -> _Extracted:
    """Extract HGVS-like expressions exactly, because case is meaningful."""
    matches = tuple(match.group(0) for match in _HGVS_RE.finditer(text))
    return _Extracted(tuple(sorted(matches)), matches)


def _extract_genes(text: str) -> _Extracted:
    """Extract conservative gene symbols that must remain verbatim."""
    genes: set[str] = set()
    matches: list[str] = []
    for pattern in (_GENE_WITH_DIGIT_RE, _GENE_CONTEXT_RE):
        for match in pattern.finditer(text):
            gene = match["gene"]
            if gene in _GENE_STOPWORDS:
                continue
            genes.add(gene)
            matches.append(gene)
    return _Extracted(tuple(sorted(genes)), tuple(matches))


def _evidence(source: _Extracted, candidate: _Extracted) -> ValidationEvidence:
    """Build public evidence from two internal extractions."""
    return ValidationEvidence(
        source_matches=source.matches,
        candidate_matches=candidate.matches,
        normalized_source=source.values,
        normalized_candidate=candidate.values,
    )


def _compare(
    attribute: CandidateAttribute,
    source: _Extracted,
    candidate: _Extracted,
    *,
    missing_is_review: bool = False,
) -> AttributeCheck:
    """Compare one attribute family with fail-closed deterministic rules."""
    review_flags = tuple(
        sorted(set((*source.review_flags, *candidate.review_flags)))
    )
    if source.values == candidate.values:
        if review_flags:
            return AttributeCheck(
                attribute,
                CheckStatus.REVIEW,
                review_flags,
                _evidence(source, candidate),
            )
        status = (
            CheckStatus.PASS if source.values else CheckStatus.NOT_APPLICABLE
        )
        return AttributeCheck(
            attribute,
            status,
            (),
            _evidence(source, candidate),
        )
    prefix = attribute.value
    if source.values and not candidate.values:
        status = CheckStatus.REVIEW if missing_is_review else CheckStatus.FAIL
        suffix = "not_verifiable" if missing_is_review else "missing"
        flag = f"{prefix}_{suffix}"
    elif candidate.values and not source.values:
        status = CheckStatus.REVIEW if missing_is_review else CheckStatus.FAIL
        suffix = "addition_not_verifiable" if missing_is_review else "added"
        flag = f"{prefix}_{suffix}"
    else:
        status = CheckStatus.FAIL
        flag = f"{prefix}_conflict"
    return AttributeCheck(
        attribute,
        status,
        tuple(sorted(set((*review_flags, flag)))),
        _evidence(source, candidate),
    )


def _brand_check(source_text: str, candidate_text: str) -> AttributeCheck:
    """Require exact brand preservation or defer translations to review."""
    source = _extract_brands(source_text)
    candidate = _extract_brands(candidate_text)
    if source.values and all(
        brand in candidate_text.casefold() for brand in source.values
    ):
        candidate = _Extracted(
            source.values,
            candidate.matches or source.values,
        )
    if not source.values and not candidate.values:
        status = CheckStatus.NOT_APPLICABLE
        flags: tuple[str, ...] = ()
    elif source.values == candidate.values:
        status = CheckStatus.PASS
        flags = ()
    elif source.values and candidate.values:
        status = CheckStatus.FAIL
        flags = ("brand_conflict",)
    elif source.values:
        status = CheckStatus.REVIEW
        flags = ("brand_not_verifiable",)
    else:
        status = CheckStatus.FAIL
        flags = ("brand_added",)
    return AttributeCheck(
        CandidateAttribute.BRAND,
        status,
        flags,
        _evidence(source, candidate),
    )


def validate_candidate(
    source_name: str,
    candidate_name_zh: str,
) -> CandidateValidation:
    """Compare critical medical attributes in source and candidate text.

    Clear additions, losses, or substitutions in hard attributes produce
    ``fail``.  Brand translations, ambiguous anatomy, or terms without any
    checkable attribute produce ``review``.  ``pass`` means only that all
    detected critical attributes were preserved; it is not proof of overall
    translation correctness.

    Args:
        source_name: Original authoritative concept name, usually English.
        candidate_name_zh: Chinese candidate to validate.

    Returns:
        An immutable, JSON-serializable validation result with evidence.
    """
    source_text = _normalize_text(source_name)
    candidate_text = _normalize_text(candidate_name_zh)
    source_quantities = _extract_quantities(source_text)
    candidate_quantities = _extract_quantities(candidate_text)

    checks = (
        _compare(
            CandidateAttribute.NUMERIC_VALUE,
            _extract_numbers(source_text, source_quantities.spans),
            _extract_numbers(candidate_text, candidate_quantities.spans),
        ),
        _compare(
            CandidateAttribute.UNIT,
            source_quantities.dimensions,
            candidate_quantities.dimensions,
        ),
        _compare(
            CandidateAttribute.POLARITY_NEGATION,
            _extract_polarity(source_text),
            _extract_polarity(candidate_text),
        ),
        _compare(
            CandidateAttribute.LATERALITY,
            _extract_laterality(source_text),
            _extract_laterality(candidate_text),
        ),
        _compare(
            CandidateAttribute.DOSAGE_CONCENTRATION,
            source_quantities.extracted,
            candidate_quantities.extracted,
        ),
        _compare(
            CandidateAttribute.DOSE_FORM,
            _extract_pattern_map(source_text, _DOSE_FORM_PATTERNS),
            _extract_pattern_map(candidate_text, _DOSE_FORM_PATTERNS),
        ),
        _compare(
            CandidateAttribute.ADMINISTRATION_ROUTE,
            _extract_pattern_map(source_text, _ROUTE_PATTERNS),
            _extract_pattern_map(candidate_text, _ROUTE_PATTERNS),
        ),
        _compare(
            CandidateAttribute.ANATOMICAL_SITE,
            _extract_anatomy(source_text),
            _extract_anatomy(candidate_text),
            missing_is_review=True,
        ),
        _compare(
            CandidateAttribute.SEX,
            _extract_pattern_map(source_text, _SEX_PATTERNS),
            _extract_pattern_map(candidate_text, _SEX_PATTERNS),
        ),
        _compare(
            CandidateAttribute.TIME_FREQUENCY,
            _extract_time_frequency(source_text),
            _extract_time_frequency(candidate_text),
        ),
        _brand_check(source_text, candidate_text),
        _compare(
            CandidateAttribute.PACKAGE,
            _extract_pattern_map(source_text, _PACKAGE_PATTERNS),
            _extract_pattern_map(candidate_text, _PACKAGE_PATTERNS),
        ),
        _compare(
            CandidateAttribute.GENE_SYMBOL,
            _extract_genes(source_text),
            _extract_genes(candidate_text),
        ),
        _compare(
            CandidateAttribute.HGVS,
            _extract_hgvs(source_text),
            _extract_hgvs(candidate_text),
        ),
    )

    flags = [flag for check in checks for flag in check.flags]
    if not source_text:
        flags.append("source_name_empty")
    if not candidate_text:
        flags.append("candidate_name_empty")

    applicable = tuple(
        check for check in checks if check.status != CheckStatus.NOT_APPLICABLE
    )
    if any(check.status == CheckStatus.FAIL for check in checks):
        status = ValidationStatus.FAIL
    elif not candidate_text:
        status = ValidationStatus.FAIL
    elif any(check.status == CheckStatus.REVIEW for check in checks):
        status = ValidationStatus.REVIEW
    elif not source_text:
        status = ValidationStatus.REVIEW
    elif not applicable:
        status = ValidationStatus.REVIEW
        flags.append("no_checkable_critical_attributes")
    else:
        status = ValidationStatus.PASS

    return CandidateValidation(
        validation_status=status,
        flags=tuple(sorted(set(flags))),
        checks=checks,
    )
