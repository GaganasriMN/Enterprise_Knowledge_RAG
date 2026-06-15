from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "hr_policy_handbook.pdf"


PAGES = [
    (
        "Remote Work Policy",
        "Employees may work remotely up to three days per week with manager approval. "
        "Remote work must be recorded in the workforce portal before 10 AM local time. "
        "Customer-facing teams need coverage approval before taking remote days. "
        "Employees must use managed devices, approved VPN, and company collaboration tools."
    ),
    (
        "Leave Policy",
        "Full-time employees receive twenty paid leave days per calendar year. Sick leave is "
        "available for medical needs and should be reported through the HR portal. Leave requests "
        "longer than five working days require manager and HR approval."
    ),
    (
        "Benefits Policy",
        "Employees are eligible for health insurance, wellness reimbursement, and retirement plan "
        "contributions after onboarding completion. Benefits changes are processed during open "
        "enrollment or after qualifying life events."
    ),
    (
        "Performance Review Policy",
        "Performance reviews occur twice per year. Managers must document goals, feedback, growth "
        "areas, and promotion readiness. Calibration discussions must avoid protected-class bias."
    ),
    (
        "Travel and Expense Policy",
        "Business travel requires manager approval and must use preferred vendors where available. "
        "Expense claims must include receipts, business purpose, cost center, and approval trail."
    ),
    (
        "Code of Conduct",
        "Employees must maintain respectful communication, protect confidential information, and "
        "avoid conflicts of interest. Concerns can be reported to HR, Legal, or Compliance."
    ),
    (
        "Data Handling for HR Records",
        "HR records include sensitive personal information and must be accessed only for approved "
        "business purposes. Exports require HR approval and must be stored in approved systems."
    ),
    (
        "Onboarding Policy",
        "New hires must complete identity verification, security awareness training, benefits setup, "
        "and role-specific access approvals before receiving production system access."
    ),
    (
        "Offboarding Policy",
        "Managers must submit offboarding requests at least five business days before the last workday. "
        "Access removal, asset return, and knowledge transfer must be completed before departure."
    ),
    (
        "Hybrid Meeting Policy",
        "Hybrid meetings should include remote-friendly agendas, written decisions, and accessible "
        "recordings when appropriate. Sensitive HR discussions must not be recorded without approval."
    ),
    (
        "Compensation Confidentiality",
        "Individual compensation records are confidential and available only to authorized HR, Finance, "
        "and Admin users. General employees can view only their own compensation documents."
    ),
    (
        "Policy Exceptions",
        "Exceptions to HR policies require documented business justification, manager approval, and HR "
        "review. Exceptions involving sensitive data require additional Compliance approval."
    ),
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=letter, title="HR Employee Policy Handbook 2026")
    styles = getSampleStyleSheet()
    story = []

    for page_number, (title, body) in enumerate(PAGES, start=1):
        story.append(Paragraph("HR Employee Policy Handbook 2026", styles["Title"]))
        story.append(Spacer(1, 18))
        story.append(Paragraph(f"Page {page_number}: {title}", styles["Heading1"]))
        story.append(Spacer(1, 12))
        for _ in range(5):
            story.append(Paragraph(body, styles["BodyText"]))
            story.append(Spacer(1, 10))
        story.append(
            Paragraph(
                "Metadata: department=HR; sensitivity=internal; allowed_roles=Employee,HR,Admin",
                styles["Italic"],
            )
        )
        if page_number != len(PAGES):
            story.append(PageBreak())

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
