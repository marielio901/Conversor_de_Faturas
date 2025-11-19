"""
Conversor de Faturas de Energia

"""

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import tempfile
import re
import unicodedata
from datetime import datetime
import camelot
import pandas as pd
import pdfplumber
import streamlit as st

# Mapas auxiliares para meses (permitindo várias grafias).
MONTH_LOOKUP = {
    "JAN": "JAN", "JANEIRO": "JAN",
    "FEV": "FEV", "FEVEREIRO": "FEV",
    "MAR": "MAR", "MARCO": "MAR", "MARÇO": "MAR",
    "ABR": "ABR", "ABRIL": "ABR",
    "MAI": "MAI", "MAIO": "MAI",
    "JUN": "JUN", "JUNHO": "JUN",
    "JUL": "JUL", "JULHO": "JUL",
    "AGO": "AGO", "AGOSTO": "AGO",
    "SET": "SET", "SETEMBRO": "SET", "SEP": "SET",
    "OUT": "OUT", "OUTUBRO": "OUT", "OCT": "OUT",
    "NOV": "NOV", "NOVEMBRO": "NOV",
    "DEZ": "DEZ", "DEZEMBRO": "DEZ", "DEC": "DEZ",
}
MONTH_ORDER = {abbr: idx for idx, abbr in enumerate(
    ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"], start=1
)}

DATE_STD_REGEX = re.compile(r"(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{2,4})")
DATE_TEXT_REGEX = re.compile(r"(\d{1,2})\s+([A-Za-zÇç]{3,9})\s+(\d{4})")
CURRENCY_REGEX = re.compile(r"(-?(?:\d{1,3}(?:\.\d{3})*(?:,\d{2,})|\d+,\d{2,}))")
TRAILING_NUMBER_REGEX = re.compile(
    r"(-?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:[.,]\d+)*)(?:\s*)$"
)
UNIT_KEYWORDS = {
    "KW",
    "KWH",
    "KVAR",
    "KVARH",
    "MWH",
    "WH",
    "KVA",
    "KVAH",
    "MW",
    "MWD",
    "DIAS",
    "DIA",
    "UN",
    "UND",
    "M3",
    "M2",
}
CNPJ_REGEX = re.compile(r"(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}|\d{14})")
PDF_DEFAULT_PASSWORD = "0604"


@dataclass
class InvoiceItem:
    descricao: str
    unidade: str
    quantidade: str
    preco_unitario: str
    valor: str


@dataclass
class EnergyBill:
    instalacao: str = ""
    mes_ano_referencia: str = ""
    data_vencimento: str = ""
    data_emissao: str = ""
    valor_total: Optional[float] = None
    cliente: str = ""
    cnpj: str = ""
    arquivo_origem: str = ""
    datas_leitura: str = ""
    itens_fatura: List[InvoiceItem] = field(default_factory=list)
    pis_cofins_base: str = ""
    pis_cofins_aliq: str = ""
    icms_base_calc: str = ""
    icms_aliq: str = ""
    icms_tarifa_unit: str = ""

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "instalacao": self.instalacao,
            "mes_ano_referencia": self.mes_ano_referencia,
            "data_vencimento": self.data_vencimento,
            "data_emissao": self.data_emissao,
            "datas_leitura": self.datas_leitura,
            "valor_total": self.valor_total,
            "cliente": self.cliente,
            "cnpj": self.cnpj,
            "arquivo_origem": self.arquivo_origem,
            "itens_descricao": join_item_attribute(self.itens_fatura, "descricao"),
            "itens_unidade": join_item_attribute(self.itens_fatura, "unidade"),
            "itens_quantidade": join_item_attribute(self.itens_fatura, "quantidade"),
            "itens_preco_unitario": join_unit_prices(self.itens_fatura),
            "itens_valor": join_item_attribute(self.itens_fatura, "valor"),
            "pis_cofins_base_calc": self.pis_cofins_base,
            "pis_cofins_aliquota": self.pis_cofins_aliq,
            "pis_cofins_resumo": summarize_pis_cofins(self.pis_cofins_base, self.pis_cofins_aliq),
            "icms_base_calc": self.icms_base_calc,
            "icms_aliquota": self.icms_aliq,
            "icms_tarifa_unitaria": self.icms_tarifa_unit,
        }


def join_item_attribute(items: List[InvoiceItem], attr: str) -> str:
    values = []
    for item in items:
        value = getattr(item, attr, "")
        if attr in {"quantidade", "preco_unitario", "valor"}:
            values.append(ensure_numeric_text(value))
            continue
        if value:
            clean = value.strip()
            if clean:
                values.append(clean)
    return "; ".join(values)


def join_unit_prices(items: List[InvoiceItem]) -> str:
    values = []
    for item in items:
        price = compute_unit_price_text(item)
        if price:
            values.append(price)
    return "; ".join(values)


def summarize_pis_cofins(base: str, aliq: str) -> str:
    if not base and not aliq:
        return ""
    base_txt = f"Base: {base}" if base else ""
    aliq_txt = f"Aliq.: {aliq}" if aliq else ""
    return " ".join(part for part in [base_txt, aliq_txt] if part)


def remove_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def normalize_reference(token: str) -> Optional[str]:
    token = token.replace(" ", "")
    if "/" not in token:
        return None
    month_part, year = token.split("/", 1)
    clean_month = remove_accents(month_part).upper()
    abbr = MONTH_LOOKUP.get(clean_month)
    if not abbr or not year.isdigit():
        return None
    return f"{abbr}/{year}"


def extract_reference_year(reference: str) -> Optional[int]:
    if not reference or "/" not in reference:
        return None
    parts = reference.split("/", 1)
    if len(parts) != 2:
        return None
    year = parts[1]
    if year.isdigit():
        return int(year)
    return None


def normalize_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    match = DATE_STD_REGEX.search(raw)
    if match:
        day, month, year = match.groups()
        year = f"20{year}" if len(year) == 2 else year
        try:
            date_obj = datetime(int(year), int(month), int(day))
            return date_obj.strftime("%d/%m/%Y")
        except ValueError:
            return None
    match = DATE_TEXT_REGEX.search(remove_accents(raw).upper())
    if match:
        day, month_name, year = match.groups()
        abbr = MONTH_LOOKUP.get(month_name.upper())
        if not abbr:
            return None
        month = MONTH_ORDER.get(abbr)
        try:
            date_obj = datetime(int(year), month, int(day))
            return date_obj.strftime("%d/%m/%Y")
        except ValueError:
            return None
    return None


def build_date_from_parts(day: str, month: str, year: int) -> Optional[str]:
    try:
        date_obj = datetime(int(year), int(month), int(day))
        return date_obj.strftime("%d/%m/%Y")
    except ValueError:
        return None


def parse_currency(raw: str) -> Optional[float]:
    if not raw:
        return None
    raw = raw.replace("R$", "").replace(" ", "")
    raw = raw.replace(".", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def first_currency_fragment(text: str) -> str:
    if not text:
        return ""
    match = CURRENCY_REGEX.search(text)
    if match:
        return match.group(1)
    return text.strip()


def format_currency(value: Optional[float]) -> str:
    if value is None:
        return ""
    formatted = f"{value:,.2f}"
    # Ajuste para formato PT-BR
    return "R$ " + formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def format_decimal_br(value: Optional[float], decimals: int = 8) -> str:
    if value is None:
        return ""
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace(".", ",")


def open_pdf_document(file_path: str):
    """
    Abre o PDF tentando primeiro sem senha e, se necessario, com a senha padrao.
    """
    try:
        return pdfplumber.open(file_path)
    except Exception as first_exc:
        try:
            return pdfplumber.open(file_path, password=PDF_DEFAULT_PASSWORD)
        except Exception:
            raise first_exc


def read_pdf_lines(file_path: str) -> List[str]:
    lines: List[str] = []
    with open_pdf_document(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines.extend(line.strip() for line in text.splitlines() if line.strip())
    return lines


def extract_instalacao(text: str) -> str:
    normalized = remove_accents(text).upper().replace("º", "")
    patterns = [
        r"N\s*O?\s*DA\s+INSTALACAO\s*[:\-]?\s*([\d\.\-\s_/]+)",
        r"INSTALACAO\s*[:\-]?\s*([\d\.\-\s_/]+)",
    ]
    candidates: List[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            digits = re.sub(r"\D", "", match.group(1))
            if digits:
                candidates.append(digits)
    if candidates:
        return max(candidates, key=len)
    numbers = re.findall(r"\b\d{6,}\b", normalized)
    return numbers[0] if numbers else ""


def extract_reference(text: str) -> Optional[str]:
    normalized = remove_accents(text).upper()
    match = re.search(r"REFERENTE\s+A[:\s]*([A-ZÇ]{3,9}\s*/\s*\d{4})", normalized)
    if match:
        ref = normalize_reference(match.group(1))
        if ref:
            return ref
    # fallback: first occurrence of "MÊS/AAAA"
    match = re.search(r"([A-ZÇ]{3,9})\s*/\s*(\d{4})", normalized)
    if match:
        ref = normalize_reference("/".join(match.groups()))
        if ref:
            return ref
    return None


def extract_date_by_label(text: str, labels: List[str], fallback_year: Optional[int] = None) -> Optional[str]:
    for label in labels:
        regex_standard = re.compile(
            label + r".{0,40}?(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})",
            re.IGNORECASE,
        )
        match = regex_standard.search(text)
        if match:
            formatted = normalize_date(match.group(1))
            if formatted:
                return formatted

        regex_textual = re.compile(
            label + r".{0,60}?(\d{1,2}\s+[A-Za-zÇç]{3,9}\s+\d{4})",
            re.IGNORECASE,
        )
        match = regex_textual.search(text)
        if match:
            formatted = normalize_date(match.group(1))
            if formatted:
                return formatted

        if fallback_year:
            regex_partial = re.compile(
                label + r".{0,40}?(\d{1,2})[\/\.\-](\d{1,2})(?!\d)",
                re.IGNORECASE,
            )
            match = regex_partial.search(text)
            if match:
                formatted = build_date_from_parts(match.group(1), match.group(2), fallback_year)
                if formatted:
                    return formatted
    return None


def extract_value_total(text: str) -> Optional[float]:
    for label in [
        r"Valor\s+a\s+pagar",
        r"Total\s+a\s+pagar",
        r"Valor\s+total\s+R\$",
        r"Total\s+geral",
    ]:
        regex = re.compile(label + r"[^\d]*(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+,\d{2})", re.IGNORECASE)
        match = regex.search(text)
        if match:
            value = parse_currency(match.group(1))
            if value is not None:
                return value
    # fallback: look for the first currency-like value after "R$"
    match = re.search(r"R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+,\d{2})", text)
    if match:
        return parse_currency(match.group(1))
    return None


def extract_cliente(lines: List[str]) -> str:
    for line in lines:
        clean = line.strip()
        lowered = clean.lower()
        if "cliente" in lowered and ":" in clean:
            candidate = clean.split(":", 1)[1].strip()
            if candidate and "cliente" not in candidate.lower():
                return candidate
    return ""


def extract_cnpj(text: str) -> str:
    match = CNPJ_REGEX.search(text)
    if match:
        cnpj = match.group(1)
        if len(cnpj) == 14:
            return "{}{}.{}{}{}.{}{}{}{}/{}{}{}{}-{}{}".format(*cnpj)
        return cnpj
    return ""


def camelot_read_pdf_with_password(file_path: str, **kwargs):
    """
    Le tabelas tentando novamente com a senha padrao caso necessario.
    """
    try:
        return camelot.read_pdf(file_path, **kwargs)
    except Exception as first_exc:
        fallback_kwargs = dict(kwargs)
        fallback_kwargs["password"] = PDF_DEFAULT_PASSWORD
        try:
            return camelot.read_pdf(file_path, **fallback_kwargs)
        except Exception:
            raise first_exc


def extract_itens_camelot(file_path: str) -> List[InvoiceItem]:
    camelot_options = [
        {"flavor": "stream", "strip_text": "\n"},
        {"flavor": "lattice", "strip_text": "\n", "line_scale": 40},
    ]

    for options in camelot_options:
        try:
            tables = camelot_read_pdf_with_password(
                file_path,
                pages="1-end",
                **options,
            )
        except Exception:
            continue

        for table in tables:
            df = table.df
            header_idx = find_header_row(df)
            if header_idx is None:
                continue

            headers, header_rows = build_combined_headers(df, header_idx)
            column_map = map_header_columns(headers)
            if "descricao" not in column_map or "valor" not in column_map:
                continue

            start_row = header_idx + header_rows
            items: List[InvoiceItem] = []
            for row_idx in range(start_row, len(df)):
                row_values = [clean_cell(df.iat[row_idx, col]) for col in range(df.shape[1])]
                if row_is_total(row_values):
                    break
                descricao = get_value_from_row(row_values, column_map, "descricao")
                if not descricao:
                    continue
                unidade = get_value_from_row(row_values, column_map, "unidade")
                quantidade = get_value_from_row(row_values, column_map, "quantidade")
                preco = first_currency_fragment(get_value_from_row(row_values, column_map, "preco_unitario"))
                valor = first_currency_fragment(get_value_from_row(row_values, column_map, "valor"))

                # ignorar linhas vazias
                if not any([unidade, quantidade, preco, valor]):
                    continue

                items.append(
                    InvoiceItem(
                        descricao=descricao,
                        unidade=unidade,
                        quantidade=quantidade,
                        preco_unitario=preco,
                        valor=valor,
                    )
                )

            if items:
                return items
    return []


def clean_cell(cell: object) -> str:
    if cell is None:
        return ""
    return str(cell).strip()


def find_header_row(df: pd.DataFrame) -> Optional[int]:
    for idx in range(len(df)):
        row_text = " ".join(clean_cell(df.iat[idx, col]) for col in range(df.shape[1]))
        normalized = remove_accents(row_text).upper()
        if "ITENS" in normalized and "FATURA" in normalized:
            return idx
        keyword_hits = sum(
            1
            for keyword in [
                "ITEM",
                "DESCRI",
                "SERVICO",
                "SERVI",
                "QTDE",
                "QUANT",
                "QTD",
                "UNID",
                "UND",
                "PRECO",
                "VALOR",
                "TARIFA",
            ]
            if keyword in normalized
        )
        if keyword_hits >= 3:
            return idx
    return None


def extract_vencimento_from_referente(lines: List[str], fallback_year: Optional[int] = None) -> Optional[str]:
    for idx, line in enumerate(lines):
        normalized = remove_accents(line).upper()
        if "REFERENTE" in normalized and "VENC" in normalized:
            window = " ".join(lines[idx: idx + 3])
            match = DATE_STD_REGEX.search(window)
            if match:
                raw = "/".join(match.groups())
                formatted = normalize_date(raw)
                if formatted:
                    return formatted
            if fallback_year:
                partial = re.search(r"(\d{1,2})[\/\.\-](\d{1,2})(?!\d)", window)
                if partial:
                    formatted = build_date_from_parts(partial.group(1), partial.group(2), fallback_year)
                    if formatted:
                        return formatted
            match = DATE_TEXT_REGEX.search(remove_accents(window).upper())
            if match:
                raw = " ".join(match.groups())
                formatted = normalize_date(raw)
                if formatted:
                    return formatted
            # headers in one line and values on the next
            if idx + 1 < len(lines):
                next_line = lines[idx + 1]
                match = DATE_STD_REGEX.search(next_line)
                if match:
                    raw = "/".join(match.groups())
                    formatted = normalize_date(raw)
                    if formatted:
                        return formatted
                if fallback_year:
                    partial = re.search(r"(\d{1,2})[\/\.\-](\d{1,2})(?!\d)", next_line)
                    if partial:
                        formatted = build_date_from_parts(partial.group(1), partial.group(2), fallback_year)
                        if formatted:
                            return formatted
    return None


def looks_like_header_row(cells: List[str]) -> bool:
    keywords = ["UNID", "QUANT", "PRECO", "VALOR", "PIS", "COFINS", "ICMS"]
    for cell in cells:
        normalized = remove_accents(cell).upper()
        if any(keyword in normalized for keyword in keywords):
            return True
    return False


def build_combined_headers(df: pd.DataFrame, header_idx: int) -> Tuple[List[str], int]:
    first = [clean_cell(df.iat[header_idx, col]) for col in range(df.shape[1])]
    header_rows = 1
    headers = first
    if header_idx + 1 < len(df):
        second = [clean_cell(df.iat[header_idx + 1, col]) for col in range(df.shape[1])]
        if looks_like_header_row(second):
            headers = [
                " ".join(filter(None, [first[col], second[col]])).strip()
                for col in range(df.shape[1])
            ]
            header_rows = 2
    return headers, header_rows


def map_header_columns(headers: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, header in enumerate(headers):
        normalized = remove_accents(header).upper()
        if not normalized:
            continue
        if any(word in normalized for word in ["ITEM", "DESCRI", "FATURA"]):
            mapping.setdefault("descricao", idx)
        elif "UNID" in normalized or "UND" in normalized:
            mapping["unidade"] = idx
        elif any(word in normalized for word in ["QUANT", "QTD", "QTE"]):
            mapping["quantidade"] = idx
        elif "PRECO" in normalized or "UNIT" in normalized:
            mapping["preco_unitario"] = idx
        elif "VALOR" in normalized and "PIS" not in normalized and "COFINS" not in normalized:
            mapping["valor"] = idx
    return mapping


def get_value_from_row(row_values: List[str], column_map: Dict[str, int], field: str) -> str:
    idx = column_map.get(field)
    if idx is None or idx >= len(row_values):
        return ""
    return row_values[idx].strip()


def row_is_total(row_values: List[str]) -> bool:
    return any("TOTAL" in remove_accents(value).upper() for value in row_values)


def parse_item_line(line: str) -> Optional[InvoiceItem]:
    line_clean = re.sub(r"\s+", " ", line.strip())
    if not line_clean:
        return None
    currency_matches = list(CURRENCY_REGEX.finditer(line_clean))
    if len(currency_matches) < 2:
        return None

    valor_match = currency_matches[-1]
    preco_match = currency_matches[-2]

    before_preco = line_clean[:preco_match.start()]
    quant_match = TRAILING_NUMBER_REGEX.search(before_preco)
    quant_start = quant_end = None
    if quant_match:
        quant_text = quant_match.group(1)
        quant_start = quant_match.start()
        quant_end = quant_match.end()
        before_quant = before_preco[:quant_start].rstrip()
    elif len(currency_matches) >= 3:
        quant_match = currency_matches[-3]
        quant_text = quant_match.group(1)
        quant_start = quant_match.start()
        quant_end = quant_match.end()
        before_quant = line_clean[:quant_start].rstrip()
    else:
        return None

    tokens = before_quant.split()
    if tokens:
        unidade = tokens[-1]
        descricao = " ".join(tokens[:-1]).strip()
    else:
        unidade = ""
        descricao = before_quant.strip()

    if not descricao:
        descricao = before_quant.strip()

    if quant_end is None:
        return None
    after_quant = line_clean[quant_end:].strip()
    currency_after_quant = list(CURRENCY_REGEX.finditer(after_quant))
    if len(currency_after_quant) < 2:
        return None
    preco_text = currency_after_quant[0].group(1)
    valor_text = currency_after_quant[1].group(1)

    return InvoiceItem(
        descricao=descricao or "",
        unidade=unidade or "",
        quantidade=quant_text,
        preco_unitario=preco_text,
        valor=valor_text,
    )


def extract_itens_fatura_text(lines: List[str]) -> List[InvoiceItem]:
    items: List[InvoiceItem] = []
    start = None
    for idx, line in enumerate(lines):
        normalized = remove_accents(line).upper()
        if "ITENS DA FATURA" in normalized:
            start = idx + 1
            break
        header_hits = any(key in normalized for key in ["DESCR", "SERVIC", "ITEM"])
        value_hits = any(key in normalized for key in ["VALOR", "VLR", "R$", "PRECO", "TARIFA"])
        if header_hits and value_hits:
            start = idx + 1
            break

    iterable = lines[start:] if start is not None else lines

    for line in iterable:
        normalized = remove_accents(line).upper()
        if not line.strip():
            if items:
                break
            continue
        if any(stop in normalized for stop in ["TOTAL", "PIS", "COFINS"]):
            if items:
                break
        parsed = parse_item_line(line)
        if parsed:
            items.append(parsed)
    return items


def sanitize_items(itens: List[InvoiceItem]) -> List[InvoiceItem]:
    for item in itens:
        clean_item_unit(item)
    return itens


def clean_item_unit(item: InvoiceItem) -> None:
    if item.unidade and item.unidade.strip():
        item.unidade = item.unidade.strip()
        return
    desc = item.descricao.strip()
    if not desc:
        return
    result = extract_unit_from_description(desc)
    if result:
        new_desc, unit = result
        item.descricao = new_desc
        item.unidade = unit


def extract_unit_from_description(desc: str) -> Optional[Tuple[str, str]]:
    tokens = desc.split()
    if not tokens:
        return None
    candidate = tokens[-1]
    normalized = normalize_unit_token(candidate)
    if normalized in UNIT_KEYWORDS:
        new_desc = " ".join(tokens[:-1]).strip()
        return (new_desc or desc, candidate)
    return None


def normalize_unit_token(token: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9/]", "", token)
    return remove_accents(cleaned).upper()


def extract_pis_cofins_info(lines: List[str]) -> Tuple[str, str]:
    for line in lines:
        normalized = remove_accents(line).upper()
        if "PIS" in normalized and "COFINS" in normalized:
            base_match = re.search(
                r"Base\s*Calc\.?\s*[:\-]?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+,\d{2})",
                line,
                re.IGNORECASE,
            )
            aliq_match = re.search(r"Aliq\.?\s*[:\-]?\s*([\d\.,]+%?)", line, re.IGNORECASE)
            base = base_match.group(1) if base_match else ""
            aliq = aliq_match.group(1) if aliq_match else ""
            if not base:
                curr = CURRENCY_REGEX.search(line)
                if curr:
                    base = curr.group(1)
            return base, aliq
    return "", ""


def extract_icms_tarifa_unit(text: str) -> str:
    match = re.search(
        r"ICMS\s+Tarifa\s+Uni?t\.?\s*[:\-]?\s*([\d\.,]+)",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def extract_icms_base_aliq(lines: List[str]) -> Tuple[str, str]:
    for line in lines:
        normalized = remove_accents(line).upper()
        if "ICMS" not in normalized:
            continue
        base_match = re.search(
            r"Base\s*Calc\.?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+,\d{2})",
            line,
            re.IGNORECASE,
        )
        aliq_match = re.search(
            r"Ali[qq]\.?\s*[:\-]?\s*([\d\.,]+%?)",
            line,
            re.IGNORECASE,
        )
        if base_match or aliq_match:
            base = base_match.group(1) if base_match else ""
            aliq = aliq_match.group(1) if aliq_match else ""
            return base, aliq
    return "", ""


def extract_datas_leitura(lines: List[str], fallback_year: Optional[int] = None) -> str:
    for idx, line in enumerate(lines):
        normalized = remove_accents(line).upper()
        if "DATA" in normalized and "LEITURA" in normalized:
            combined = " ".join(lines[idx:idx + 3])
            dates = DATE_STD_REGEX.findall(combined)
            if dates:
                primeira = normalize_date("/".join(dates[0]))
                segunda = normalize_date("/".join(dates[1])) if len(dates) > 1 else None
                return segunda or primeira or ""
            textual = DATE_TEXT_REGEX.findall(remove_accents(combined).upper())
            if textual:
                primeira = normalize_date(" ".join(textual[0]))
                segunda = normalize_date(" ".join(textual[1])) if len(textual) > 1 else None
                return segunda or primeira or ""
            if fallback_year:
                partial = re.findall(r"(\d{1,2})[\/\.\-](\d{1,2})(?!\d)", combined)
                if partial:
                    primeira = build_date_from_parts(partial[0][0], partial[0][1], fallback_year)
                    segunda = (
                        build_date_from_parts(partial[1][0], partial[1][1], fallback_year)
                        if len(partial) > 1
                        else None
                    )
                    return segunda or primeira or ""
    return ""


def parse_pdf(file_path: str) -> EnergyBill:
    lines = read_pdf_lines(file_path)
    if not lines:
        raise ValueError("Não foi possível extrair texto do PDF.")
    text = "\n".join(lines)

    instalacao = extract_instalacao(text)
    referencia = extract_reference(text) or ""
    reference_year = extract_reference_year(referencia)
    data_venc = (
        extract_vencimento_from_referente(lines, reference_year)
        or extract_date_by_label(
            text,
            [r"Vencimento", r"Vence", r"Data\s+de\s+vencimento"],
            fallback_year=reference_year,
        )
        or ""
    )
    data_emissao = extract_date_by_label(
        text,
        [r"Emiss[aã]o", r"Data\s+de\s+emiss[aã]o", r"Data\s+Atual"],
        fallback_year=reference_year,
    ) or ""
    valor_total_extracted = extract_value_total(text)
    cliente = extract_cliente(lines)
    cnpj = extract_cnpj(text)
    datas_leitura = extract_datas_leitura(lines, reference_year)
    itens = extract_itens_camelot(file_path)
    if not itens:
        itens = extract_itens_fatura_text(lines)
    itens = sanitize_items(itens)
    valor_total = compute_items_total(itens) if itens else valor_total_extracted
    pis_base, pis_aliq = extract_pis_cofins_info(lines)
    icms_base, icms_aliq = extract_icms_base_aliq(lines)
    icms_tarifa = extract_icms_tarifa_unit(text)

    return EnergyBill(
        instalacao=instalacao,
        mes_ano_referencia=referencia,
        data_vencimento=data_venc,
        data_emissao=data_emissao,
        valor_total=valor_total,
        cliente=cliente,
        cnpj=cnpj,
        arquivo_origem=file_path,
        datas_leitura=datas_leitura,
        itens_fatura=itens,
        pis_cofins_base=pis_base,
        pis_cofins_aliq=pis_aliq,
        icms_base_calc=icms_base,
        icms_aliq=icms_aliq,
        icms_tarifa_unit=icms_tarifa,
    )


def build_bill_key(bill: EnergyBill) -> Tuple[str, str, str]:
    filename = Path(bill.arquivo_origem).name.lower()
    return (filename, bill.mes_ano_referencia or "", bill.data_vencimento or "")


def compose_display_row(bill: EnergyBill, item: Optional[InvoiceItem]) -> Dict[str, str]:
    descricao = item.descricao if item else ""
    unidade = item.unidade if item else ""
    quantidade = ensure_numeric_text(item.quantidade if item else "", use_decimal=True)
    preco_base = compute_unit_price_text(item) if item else "0"
    preco_value = parse_currency(preco_base)
    preco = format_currency(preco_value) if preco_value is not None else ensure_numeric_text(preco_base, use_decimal=True)
    valor_raw = ensure_numeric_text(item.valor if item else "", use_decimal=True)
    valor_value = parse_currency(valor_raw)
    valor = format_currency(valor_value) if valor_value is not None else valor_raw
    return {
        "Instalação": bill.instalacao,
        "Mês/Ano": bill.mes_ano_referencia,
        "Vencimento": bill.data_vencimento,
        "Datas de Leitura": bill.datas_leitura,
        "Item da Fatura": descricao,
        "Unidade": unidade,
        "Quantidade": quantidade,
        "Preço Unitário": preco,
        "Valor do Item": valor,
        "Arquivo": Path(bill.arquivo_origem).name,
    }


def build_display_rows(records: List[EnergyBill]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for bill in records:
        if bill.itens_fatura:
            for item in bill.itens_fatura:
                rows.append(compose_display_row(bill, item))
        else:
            rows.append(compose_display_row(bill, None))
    return rows


def build_export_row(bill: EnergyBill, item: Optional[InvoiceItem]) -> Dict[str, object]:
    descricao = item.descricao if item else ""
    unidade = item.unidade if item else ""
    quantidade_val = parse_decimal_value(item.quantidade) if item and item.quantidade else None
    if quantidade_val is None:
        quantidade_val = 0.0
    preco_val = compute_unit_price_value(item) or 0.0
    valor_val = parse_currency(item.valor) if item and item.valor else None
    if valor_val is None:
        valor_val = 0.0
    return {
        "instalacao": bill.instalacao,
        "mes_ano_referencia": bill.mes_ano_referencia,
        "data_vencimento": bill.data_vencimento,
        "datas_leitura": bill.datas_leitura,
        "item_descricao": descricao,
        "unidade": unidade,
        "quantidade": quantidade_val,
        "preco_unitario": preco_val,
        "valor": valor_val,
        "arquivo_origem": Path(bill.arquivo_origem).name,
    }


def build_export_rows(records: List[EnergyBill]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for bill in records:
        if bill.itens_fatura:
            for item in bill.itens_fatura:
                rows.append(build_export_row(bill, item))
        else:
            rows.append(build_export_row(bill, None))
    return rows


def build_export_dataframe(records: List[EnergyBill]) -> pd.DataFrame:
    return pd.DataFrame(build_export_rows(records))


def save_uploaded_pdf(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def process_uploaded_files(
    uploaded_files,
    records: List[EnergyBill],
    known_keys: set[Tuple[str, str, str]],
) -> Tuple[int, List[str], List[str]]:
    added = 0
    duplicates: List[str] = []
    errors: List[str] = []

    for uploaded in uploaded_files:
        temp_path = save_uploaded_pdf(uploaded)
        try:
            bill = parse_pdf(temp_path)
            bill.arquivo_origem = uploaded.name
        except Exception as exc:
            errors.append(f"{uploaded.name}: {exc}")
            continue
        finally:
            Path(temp_path).unlink(missing_ok=True)

        key = build_bill_key(bill)
        if key in known_keys:
            duplicates.append(uploaded.name)
            continue

        records.append(bill)
        known_keys.add(key)
        added += 1

    return added, duplicates, errors


def render_app() -> None:
    st.set_page_config(page_title="Consolidador de Contas de Energia", layout="wide")
    st.title("Consolidador de Contas de Energia")
    st.write("Envie os PDFs das faturas de energia, consolide os itens e exporte para Excel.")

    if "records" not in st.session_state:
        st.session_state.records = []
    if "known_keys" not in st.session_state:
        st.session_state.known_keys = set()

    with st.sidebar:
        st.header("Arquivos")
        uploaded_files = st.file_uploader(
            "Selecione arquivos PDF",
            type=["pdf"],
            accept_multiple_files=True,
            help="Você pode arrastar vários PDFs de uma vez.",
        )
        process_clicked = st.button("Processar PDFs")
        if st.button("Limpar dados"):
            st.session_state.records = []
            st.session_state.known_keys = set()
            st.experimental_rerun()

    if process_clicked:
        if not uploaded_files:
            st.info("Envie pelo menos um PDF para processar.")
        else:
            added, duplicates, errors = process_uploaded_files(
                uploaded_files,
                st.session_state.records,
                st.session_state.known_keys,
            )
            if added:
                st.success(f"{added} conta(s) incluída(s) com sucesso.")
            if duplicates:
                st.warning(
                    "Arquivos ignorados por já estarem carregados:\n- " + "\n- ".join(duplicates)
                )
            if errors:
                st.error("Falha ao processar alguns PDFs:\n- " + "\n- ".join(errors))

    records: List[EnergyBill] = st.session_state.records
    if not records:
        st.info("Nenhum PDF processado ainda.")
        return

    instalacoes_values = sorted({bill.instalacao or "" for bill in records})
    selected_instalacoes = st.multiselect(
        "Filtrar por instalação",
        options=instalacoes_values,
        default=instalacoes_values,
        format_func=lambda value: value or "Sem instalação",
    )

    filtered_records = [
        bill for bill in records if (bill.instalacao or "") in selected_instalacoes
    ]

    active_records = filtered_records if selected_instalacoes else records

    total_contas = len(active_records)
    total_itens = sum(len(bill.itens_fatura) or 1 for bill in active_records)
    total_valor = sum(bill.valor_total or 0.0 for bill in active_records)

    col1, col2, col3 = st.columns(3)
    col1.metric("Contas processadas", total_contas)
    col2.metric("Itens consolidados", total_itens)
    col3.metric("Valor total", format_currency(total_valor))

    st.subheader("Itens consolidados")
    display_rows = build_display_rows(active_records)
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    export_df_all = build_export_dataframe(records)
    buffer_all = BytesIO()
    export_df_all.to_excel(buffer_all, index=False, engine="openpyxl")
    buffer_all.seek(0)
    st.download_button(
        "Baixar Excel (todos)",
        buffer_all,
        file_name="contas_consolidadas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    if filtered_records and len(filtered_records) != len(records):
        export_df_filtered = build_export_dataframe(filtered_records)
        buffer_filtered = BytesIO()
        export_df_filtered.to_excel(buffer_filtered, index=False, engine="openpyxl")
        buffer_filtered.seek(0)
        st.download_button(
            "Baixar Excel (filtrado)",
            buffer_filtered,
            file_name="contas_filtradas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def parse_treeview_date(value: str) -> Optional[datetime]:
    try:
        return datetime.strptime(value, "%d/%m/%Y")
    except ValueError:
        return None


def parse_reference_order(value: str):
    if not value or "/" not in value:
        return (9999, 13)
    month, year = value.split("/", 1)
    order = MONTH_ORDER.get(month.upper(), 13)
    try:
        year_int = int(year)
    except ValueError:
        year_int = 9999
    return (year_int, order)


def parse_datas_leitura_value(value: str) -> Optional[datetime]:
    if not value:
        return None
    first_part = value.split("-")[0].strip()
    if not first_part:
        return None
    return parse_treeview_date(first_part)


def parse_decimal_value(value: str) -> Optional[float]:
    if not value:
        return None
    normalized = value.replace(".", "").replace(",", ".")
    try:
        return float(normalized)
    except ValueError:
        return None


def compute_items_total(itens: List[InvoiceItem]) -> float:
    total = 0.0
    for item in itens:
        value = parse_currency(item.valor)
        if value is not None:
            total += value
    return total


def compute_unit_price_value(item: Optional[InvoiceItem]) -> Optional[float]:
    if not item:
        return None
    quantity = parse_decimal_value(item.quantidade)
    total = parse_currency(item.valor)
    if quantity in (None, 0) or total is None:
        return parse_currency(item.preco_unitario)
    return total / quantity


def compute_unit_price_text(item: Optional[InvoiceItem]) -> str:
    if not item:
        return "0"
    price_value = compute_unit_price_value(item)
    if price_value is None:
        return ensure_numeric_text(item.preco_unitario)
    return ensure_numeric_text(format_decimal_br(price_value))


def ensure_numeric_text(value: Optional[str], use_decimal: bool = False) -> str:
    if value is None:
        return "0"
    text = value.strip()
    if use_decimal:
        normalized = text.replace(".", "").replace(",", ".")
        try:
            number = float(normalized)
        except ValueError:
            return text or "0"
        return format_decimal_br(number)
    return text if text else "0"


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
