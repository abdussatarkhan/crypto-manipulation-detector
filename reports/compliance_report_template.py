"""
Automated Regulatory Compliance & Market Abuse Audit PDF Report Generator
========================================================================
Generates institutional-grade Suspicious Activity Reports (SAR) and Market Abuse
Regulation (MAR) compliance audit documents detailing detected wash-trading syndicates,
Benford's Law distribution violations, and statistical volume manipulation.
"""

import os
import sys
import time
import datetime
import argparse
from typing import Dict, Any, List, Optional

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.utils import load_config, setup_logger, get_project_root, generate_synthetic_trade_stream
from scripts.preprocessing import TradePreprocessor
from scripts.alert_scoring import CompositeAlertEngine

# ReportLab Imports for Professional PDF Layouts
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ComplianceReportGenerator:
    """Produces regulatory compliance PDFs from surveillance audit logs."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[Any] = None):
        self.config = config or load_config()
        self.logger = logger or setup_logger("compliance_reporter")
        self.comp_cfg = self.config.get("compliance_report", {})
        self.agency = self.comp_cfg.get("agency_header", "FINANCIAL INTELLIGENCE UNIT - SURVEILLANCE DIVISION")

    def generate_pdf_report(self, audit_data: Dict[str, Any], output_path: str) -> str:
        """
        Builds institutional PDF report summarizing manipulation findings.

        Args:
            audit_data: Audit summary dictionary produced by CompositeAlertEngine.
            output_path: Target destination for the PDF file.

        Returns:
            Path to the written PDF file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if not REPORTLAB_AVAILABLE:
            self.logger.warning("ReportLab not installed. Generating structured Markdown compliance report instead.")
            md_path = output_path.replace(".pdf", ".md")
            return self._generate_markdown_fallback(audit_data, md_path)

        self.logger.info(f"Generating PDF compliance report at: {output_path}")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        normal = styles["Normal"]

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0d1b2a"),
            fontName="Helvetica-Bold",
            spaceAfter=6
        )

        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=normal,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#415a77"),
            fontName="Helvetica-Bold",
            spaceAfter=14
        )

        section_heading = ParagraphStyle(
            "SectionH2",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1b263b"),
            fontName="Helvetica-Bold",
            spaceBefore=12,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            "Body",
            parent=normal,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#2b2d42")
        )

        story = []

        # 1. Header & Agency Banner
        story.append(Paragraph(self.agency.upper(), subtitle_style))
        story.append(Paragraph("CRYPTOCURRENCY MARKET SURVEILLANCE & ABUSE AUDIT REPORT", title_style))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0d1b2a"), spaceAfter=14))

        # 2. Executive Metadata Box
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        mpi = audit_data.get("manipulation_probability_index", 0.0)
        severity = audit_data.get("overall_severity", "UNKNOWN")

        sev_color = colors.HexColor("#d90429") if severity in ("CRITICAL", "HIGH") else colors.HexColor("#f77f00")

        meta_data = [
            [Paragraph("<b>Audit ID:</b>", body_style), Paragraph(f"SAR-{int(time.time())}", body_style),
             Paragraph("<b>Date Generated:</b>", body_style), Paragraph(now_utc, body_style)],
            [Paragraph("<b>Target Symbol:</b>", body_style), Paragraph(str(audit_data.get("symbol", "ALL")), body_style),
             Paragraph("<b>Trades Examined:</b>", body_style), Paragraph(f"{audit_data.get('total_trades_analyzed', 0):,}", body_style)],
            [Paragraph("<b>Total Notional Analyzed:</b>", body_style), Paragraph(f"${audit_data.get('total_volume_usd', 0):,.2f}", body_style),
             Paragraph("<b>Risk Severity Tier:</b>", body_style), Paragraph(f"<font color='{sev_color.hexval()}'><b>{severity}</b></font>", body_style)],
            [Paragraph("<b>Manipulation Prob Index:</b>", body_style), Paragraph(f"<b>{mpi*100:.1f}%</b>", body_style),
             Paragraph("<b>Active Alerts Raised:</b>", body_style), Paragraph(str(audit_data.get("active_alerts_count", 0)), body_style)]
        ]

        meta_table = Table(meta_data, colWidths=[1.8*inch, 1.8*inch, 1.8*inch, 1.8*inch])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ced4da")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e9ecef")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 14))

        # 3. Surveillance Sub-Signal Vector
        story.append(Paragraph("1. Multi-Modal Surveillance Signal Breakdown", section_heading))
        sub_sig = audit_data.get("sub_signals", {})
        signal_data = [
            ["Analytical Dimension", "Detection Technique", "Signal Score", "Risk Evaluation"],
            ["Graph Wash-Trading", "Johnson Cycle Detection & Louvain Communities", f"{sub_sig.get('graph_cycle_score', 0)*100:.1f}%", "CIRCULAR WASH RINGS DETECTED" if sub_sig.get('graph_cycle_score', 0) > 0.5 else "NOMINAL"],
            ["Volume Surges", "EWMA Control Chart (+3σ) & Dynamic Z-Score", f"{sub_sig.get('volume_anomaly_score', 0)*100:.1f}%", "ABNORMAL SPIKES IDENTIFIED" if sub_sig.get('volume_anomaly_score', 0) > 0.5 else "NOMINAL"],
            ["Benford's Law Forensic", "First-Digit Kolmogorov-Smirnov & MAD Test", f"{sub_sig.get('benfords_law_score', 0)*100:.1f}%", "ARTIFICIAL TRADE SIZES" if sub_sig.get('benfords_law_score', 0) > 0.5 else "CONFORMING"],
            ["Order Book Spoofing", "Cancel-to-Trade Ratio & Spread Flash Monitors", f"{sub_sig.get('spoofing_order_score', 0)*100:.1f}%", "LAYERED QUOTES SUSPECTED" if sub_sig.get('spoofing_order_score', 0) > 0.5 else "NOMINAL"]
        ]

        signal_table = Table(signal_data, colWidths=[1.7*inch, 2.5*inch, 1.2*inch, 1.8*inch])
        signal_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b263b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ("ALIGN", (2, 0), (2, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(signal_table)
        story.append(Spacer(1, 14))

        # 4. Actionable Regulatory Alerts Log
        story.append(Paragraph("2. Top Flagged Market Abuse Incidents", section_heading))
        alerts = audit_data.get("alerts", [])[:6]
        if alerts:
            alert_table_data = [["Alert ID", "Severity", "Pattern", "Description", "Notional (USD)"]]
            for a in alerts:
                alert_table_data.append([
                    a.get("alert_id", "N/A"),
                    a.get("severity", "MED"),
                    a.get("type", "N/A")[:20],
                    Paragraph(a.get("description", "N/A")[:90], body_style),
                    f"${a.get('volume_usd', 0):,.2f}"
                ])

            alert_table = Table(alert_table_data, colWidths=[1.1*inch, 0.9*inch, 1.6*inch, 2.6*inch, 1.0*inch])
            alert_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2d42")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(alert_table)
        else:
            story.append(Paragraph("No anomalous abuse events recorded during the surveillance period.", body_style))

        story.append(Spacer(1, 16))

        # 5. Regulatory Certification & Signature Block
        story.append(Paragraph("3. Officer Attestation & Evidentiary Certification", section_heading))
        disclaimer = (
            "This report is generated automatically by the algorithmic Market Abuse Detection Engine. "
            "The statistical, graph, and forensic metrics contained herein satisfy evidentiary guidelines "
            "for referral to relevant legal enforcement divisions (e.g., CFTC, SEC, ESMA). "
            "Preservation of raw Kafka ingest offsets and cryptographic address mappings is mandated."
        )
        story.append(Paragraph(disclaimer, body_style))
        story.append(Spacer(1, 24))

        sig_data = [
            [Paragraph("<b>Lead Compliance Investigator:</b>", body_style), Paragraph("___________________________", body_style),
             Paragraph("<b>Date:</b>", body_style), Paragraph(now_utc[:10], body_style)],
            [Paragraph("<b>Chief Compliance Officer (CCO):</b>", body_style), Paragraph("___________________________", body_style),
             Paragraph("<b>Status:</b>", body_style), Paragraph("PENDING FORMAL FILING", body_style)]
        ]
        sig_table = Table(sig_data, colWidths=[2.2*inch, 2.2*inch, 1.0*inch, 1.8*inch])
        sig_table.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(sig_table)

        # Build document
        doc.build(story)
        self.logger.info(f"Compliance PDF report compiled successfully: {output_path}")
        return output_path

    def _generate_markdown_fallback(self, audit_data: Dict[str, Any], output_path: str) -> str:
        """Fallback markdown summary report if ReportLab is missing."""
        content = f"""# {self.agency}
## CRYPTOCURRENCY MARKET ABUSE & WASH-TRADE COMPLIANCE AUDIT REPORT

**Audit ID:** SAR-{int(time.time())}  
**Date:** {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Target Asset:** {audit_data.get('symbol', 'BTCUSDT')}  
**Total Examined Trades:** {audit_data.get('total_trades_analyzed', 0):,}  
**Total Notional USD:** ${audit_data.get('total_volume_usd', 0):,.2f}  
**Manipulation Probability Index (MPI):** {audit_data.get('manipulation_probability_index', 0)*100:.1f}%  
**Regulatory Risk Classification:** **{audit_data.get('overall_severity', 'UNKNOWN')}**  

---

### Surveillance Sub-Signal Breakdown
- **Graph Wash-Trading (Cycles & Syndicates):** {audit_data.get('sub_signals', {}).get('graph_cycle_score', 0)*100:.1f}%
- **Volume Surge Outliers (EWMA Control Charts):** {audit_data.get('sub_signals', {}).get('volume_anomaly_score', 0)*100:.1f}%
- **Benford's Law Forensic Non-Conformity:** {audit_data.get('sub_signals', {}).get('benfords_law_score', 0)*100:.1f}%
- **Order Book Spoofing & Quote Layering:** {audit_data.get('sub_signals', {}).get('spoofing_order_score', 0)*100:.1f}%

---

### Top Active Market Abuse Incidents
"""
        for a in audit_data.get("alerts", [])[:10]:
            content += f"- **[{a.get('severity')}] {a.get('type')}** ({a.get('alert_id')}): {a.get('description')} (Volume: ${a.get('volume_usd', 0):,.2f})\n"

        content += "\n\n*Certified by Compliance Division Algorithmic Surveillance Unit.*"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate PDF compliance audit report for crypto market abuse.")
    parser.add_argument("--output", type=str, default=None, help="Output destination for report")
    args = parser.parse_args()

    root = get_project_root()
    output_dir = os.path.join(root, "reports", "generated")
    os.makedirs(output_dir, exist_ok=True)
    pdf_filename = f"compliance_audit_{int(time.time())}.pdf"
    output_path = args.output or os.path.join(output_dir, pdf_filename)

    # Generate test audit data
    generator = ComplianceReportGenerator()
    generator.logger.info("Synthesizing audit dataset for report compilation...")
    stream = generate_synthetic_trade_stream(num_trades=600, inject_wash_trading=True)
    prep = TradePreprocessor()
    trades_df = prep.normalize_trade_dataframe(stream)

    engine = CompositeAlertEngine()
    audit_data = engine.evaluate_dataset(trades_df)

    res_path = generator.generate_pdf_report(audit_data, output_path)
    print(f"\nReport generated successfully: {res_path}")


if __name__ == "__main__":
    main()
