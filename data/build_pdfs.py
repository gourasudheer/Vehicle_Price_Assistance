"""
Builds two spec-sheet PDFs (bikes_data.pdf, cars_data.pdf) from the sample
dataset in generate_dataset.py.

Each vehicle is written as ONE self-contained paragraph block with clearly
labelled fields (Brand, Model, Variant, Fuel Type, Engine, Mileage,
Ex-Showroom Price, On-Road Price, City). This "one block = one vehicle"
layout is intentional: it makes the PDF easy for a human to read AND easy
to chunk cleanly for RAG (see backend/ingest.py) because each chunk maps
1:1 to a complete, unambiguous vehicle record.

Run:
    python build_pdfs.py
Outputs:
    bikes_data.pdf, cars_data.pdf  (in this same folder)
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)

from generate_dataset import BIKES, CARS, format_price

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20)
heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"],
                                textColor=colors.HexColor("#1a3c6e"))
entry_title_style = ParagraphStyle("EntryTitle", parent=styles["Heading3"],
                                    spaceBefore=14, spaceAfter=4,
                                    textColor=colors.HexColor("#0f5132"))
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15)


def vehicle_block(v, kind):
    """One clean text block per vehicle -> one clean RAG chunk per vehicle."""
    name = f"{v['brand']} {v['model']} ({v['variant']})"
    engine = f"{v['engine_cc']} cc" if v.get("engine_cc") else "Electric motor"
    mileage = v["mileage_kmpl"]
    mileage_str = f"{mileage} km/l" if isinstance(mileage, (int, float)) else str(mileage)

    lines = [
        f"<b>{kind} Record: {name}</b>",
        f"Brand: {v['brand']} | Model: {v['model']} | Variant: {v['variant']}",
        f"Fuel Type: {v['fuel']} | Engine/Power: {engine}",
        f"Mileage: {mileage_str}",
        f"Ex-Showroom Price: {format_price(v['ex_showroom'])}",
        f"On-Road Price ({v['city']}): {format_price(v['on_road'])}",
        f"Summary: The {name} is a {v['fuel']} {kind.lower()} with a mileage of "
        f"{mileage_str}. Ex-showroom price is {format_price(v['ex_showroom'])} and the "
        f"on-road price in {v['city']} is {format_price(v['on_road'])}.",
    ]
    return "<br/>".join(lines)


def build_pdf(filename, title, intro, items, kind):
    path = os.path.join(OUT_DIR, filename)
    doc = SimpleDocTemplate(path, pagesize=A4,
                             topMargin=2 * cm, bottomMargin=2 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    story = [Paragraph(title, title_style), Spacer(1, 10),
             Paragraph(intro, body_style), Spacer(1, 16)]

    # Quick overview table
    table_data = [["Brand", "Model", "Fuel", "Mileage", "On-Road Price"]]
    for v in items:
        mileage = v["mileage_kmpl"]
        mileage_str = f"{mileage} km/l" if isinstance(mileage, (int, float)) else str(mileage)
        table_data.append([v["brand"], v["model"], v["fuel"], mileage_str,
                            format_price(v["on_road"])])
    tbl = Table(table_data, repeatRows=1, colWidths=[3.2*cm, 4.2*cm, 2.3*cm, 3.6*cm, 3.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c6e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(Paragraph("Quick Overview", heading_style))
    story.append(tbl)
    story.append(PageBreak())

    story.append(Paragraph("Detailed Records", heading_style))
    for v in items:
        story.append(Paragraph(vehicle_block(v, kind), body_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    print(f"Wrote {path}")


if __name__ == "__main__":
    build_pdf(
        "bikes_data.pdf",
        "Two-Wheeler Price & Mileage Guide",
        "Sample dataset of popular bikes/scooters/EVs in India with ex-showroom price, "
        "on-road price (Bangalore) and mileage. Demo data for practising RAG pipelines — "
        "replace with live/verified data before using in production.",
        BIKES, "Bike",
    )
    build_pdf(
        "cars_data.pdf",
        "Car Price & Mileage Guide",
        "Sample dataset of popular cars in India with ex-showroom price, on-road price "
        "(Bangalore) and mileage. Demo data for practising RAG pipelines — replace with "
        "live/verified data before using in production.",
        CARS, "Car",
    )
