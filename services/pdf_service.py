from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
import os
import re
from pathlib import Path


GEN_TMP_DIR = Path("/tmp/generate_report")


def generate_pdf(report_text: str, file_name: str) -> str:
    upload_dir = GEN_TMP_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = os.path.join(upload_dir, file_name)

    doc = SimpleDocTemplate(
        pdf_path,
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        alignment=TA_CENTER,
        textColor=HexColor("#1E3A8A"),
        spaceAfter=20,
    ))

    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=HexColor("#0F172A"),
        spaceBefore=15,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=18,
        textColor=HexColor("#475569"),
        alignment=TA_LEFT,
        spaceAfter=8,
    ))

    story = []

    for line in report_text.split("\n"):
        line = line.strip()
        line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)

        if not line:
            story.append(Spacer(1, 12))
            continue

        if line.startswith("##"):
            story.append(Paragraph(f"<b>{line[2:].strip()}</b>", styles["SectionTitle"]))
            story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#CBD5E1")))
        elif line.startswith("#"):
            story.append(Paragraph(f"<b>{line[1:].strip()}</b>", styles["ReportTitle"]))
            story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#CBD5E1")))
        elif line.startswith("- "):
            story.append(Paragraph(f"\u2022 {line[2:]}", styles["Body"]))
        elif line[0].isdigit() and "." in line:
            story.append(Paragraph(line, styles["Body"]))
        else:
            story.append(Paragraph(line, styles["Body"]))

        story.append(Spacer(1, 6))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    return pdf_path

def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(HexColor("#64748B"))
    canvas.drawRightString(560, 20, f"Page {doc.page}")
    canvas.restoreState()