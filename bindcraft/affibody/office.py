"""Small, dependency-free readers for the tabular parts of XLSX and PPTX files."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath


_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_SS_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _xml(archive: zipfile.ZipFile, name: str) -> ET.Element:
    try:
        payload = archive.read(name)
    except KeyError as exc:
        raise ValueError(f"Office archive is missing {name}") from exc
    return ET.fromstring(payload)


def _cell_column(reference: str) -> int:
    match = re.match(r"([A-Za-z]+)", reference)
    if not match:
        raise ValueError(f"invalid worksheet cell reference: {reference!r}")
    index = 0
    for character in match.group(1).upper():
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = _xml(archive, "xl/sharedStrings.xml")
    return [
        "".join(node.text or "" for node in item.iter(f"{{{_SS_NS}}}t"))
        for item in root.findall(f"{{{_SS_NS}}}si")
    ]


def read_xlsx_tables(path: str | Path) -> dict[str, list[list[str]]]:
    """Return worksheet rows while preserving empty cells between values."""

    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        workbook = _xml(archive, "xl/workbook.xml")
        rel_root = _xml(archive, "xl/_rels/workbook.xml.rels")
        relationships = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rel_root.findall(f"{{{_PKG_REL_NS}}}Relationship")
        }
        output: dict[str, list[list[str]]] = {}
        sheets = workbook.find(f"{{{_SS_NS}}}sheets")
        if sheets is None:
            return output
        for sheet in sheets:
            name = sheet.attrib.get("name", "Sheet")
            relationship_id = sheet.attrib.get(f"{{{_REL_NS}}}id")
            if relationship_id not in relationships:
                raise ValueError(f"worksheet {name!r} has no workbook relationship")
            target = relationships[relationship_id]
            worksheet_path = str(
                PurePosixPath("xl") / PurePosixPath(target.lstrip("/"))
            )
            if target.startswith("/xl/"):
                worksheet_path = target.lstrip("/")
            root = _xml(archive, worksheet_path)
            rows: list[list[str]] = []
            sheet_data = root.find(f"{{{_SS_NS}}}sheetData")
            if sheet_data is None:
                output[name] = rows
                continue
            for row_node in sheet_data.findall(f"{{{_SS_NS}}}row"):
                values: dict[int, str] = {}
                for cell in row_node.findall(f"{{{_SS_NS}}}c"):
                    column = _cell_column(cell.attrib.get("r", "A1"))
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find(f"{{{_SS_NS}}}v")
                    if cell_type == "inlineStr":
                        value = "".join(
                            node.text or ""
                            for node in cell.iter(f"{{{_SS_NS}}}t")
                        )
                    elif value_node is None:
                        value = ""
                    elif cell_type == "s":
                        shared_index = int(value_node.text or "0")
                        value = shared[shared_index]
                    elif cell_type == "b":
                        value = "TRUE" if value_node.text == "1" else "FALSE"
                    else:
                        value = value_node.text or ""
                    values[column] = value
                if values:
                    width = max(values) + 1
                    rows.append([values.get(index, "") for index in range(width)])
                else:
                    rows.append([])
            output[name] = rows
        return output


def read_pptx_tables(
    path: str | Path,
    *,
    slide_number: int,
) -> list[list[list[str]]]:
    """Return all DrawingML tables on one 1-based PowerPoint slide."""

    if int(slide_number) < 1:
        raise ValueError("slide_number must be 1-based and positive")
    with zipfile.ZipFile(Path(path)) as archive:
        root = _xml(archive, f"ppt/slides/slide{int(slide_number)}.xml")
    tables: list[list[list[str]]] = []
    for table_node in root.iter(f"{{{_DRAWING_NS}}}tbl"):
        table: list[list[str]] = []
        for row_node in table_node.findall(f"{{{_DRAWING_NS}}}tr"):
            row: list[str] = []
            for cell in row_node.findall(f"{{{_DRAWING_NS}}}tc"):
                paragraphs: list[str] = []
                for paragraph in cell.iter(f"{{{_DRAWING_NS}}}p"):
                    text = "".join(
                        node.text or ""
                        for node in paragraph.iter(f"{{{_DRAWING_NS}}}t")
                    )
                    if text:
                        paragraphs.append(text)
                row.append("\n".join(paragraphs).strip())
            table.append(row)
        tables.append(table)
    return tables
