from pathlib import Path
import pandas as pd


def generate_synthetic_dataset(raw_csv_path: Path) -> pd.DataFrame:
    """
    Creates a realistic, explicitly labeled synthetic dataset of human perceptions of AI.
    Includes intentional duplicates, missing entries, and varied phrasing to test preprocessing.
    """
    raw_csv_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        # Group A: Technical / Systems-Oriented Mental Models
        ("P01", "Software Engineer", "AI relies on algorithms and massive data to perform learning and pattern recognition for automation.", True),
        ("P02", "Data Analyst", "Artificial intelligence combines statistical algorithms, data pipelines, and machine learning to improve decision-making.", True),
        ("P03", "CS Student", "AI systems use neural networks and algorithms trained on data to simulate human intelligence and automate tasks.", True),
        ("P04", "IT Administrator", "Automation and data processing drive AI, helping organizations streamline decision-making and reduce manual effort.", True),
        ("P05", "Robotics Tech", "Intelligence in machines comes from sensors, data, and control algorithms enabling physical automation.", True),
        ("P06", "Product Manager", "AI transforms raw data into predictive decision-making and workflow automation using learning models.", True),
        ("P07", "Backend Developer", "Algorithms and compute power allow AI learning systems to optimize software automation and data retrieval.", True),
        ("P08", "Research Intern", "Machine intelligence depends on training data, optimization algorithms, and human oversight for reliable decision-making.", True),

        # Group B: Socio-Economic & Ethics-Focused Mental Models
        ("P09", "Policy Researcher", "AI decision-making often reflects historical bias in data, threatening fairness, trust, and human jobs.", True),
        ("P10", "Journalist", "Artificial intelligence impacts jobs through workplace automation while raising serious concerns about algorithmic bias and privacy.", True),
        ("P11", "Labor Union Rep", "Corporate automation driven by AI threatens worker jobs and widens inequality when humans lose decision-making control.", True),
        ("P12", "Law Student", "AI governance must address bias in training data, lack of transparency, and accountability when algorithms affect humans.", True),
        ("P13", "Sociologist", "Human values and social bias are encoded into AI data, reshaping employment jobs and institutional trust.", True),
        ("P14", "HR Specialist", "Automated screening algorithms can introduce hiring bias and anxiety over jobs unless humans oversee final decision-making.", True),
        ("P15", "Civil Servant", "Public trust in AI requires regulation against bias, protection of jobs, and clear human accountability.", True),
        ("P16", "Educator", "AI challenges academic integrity and future jobs while requiring humans to question bias in automated answers.", True),

        # Group C: Human-Centric & Creative Collaboration Mental Models
        ("P17", "Graphic Designer", "AI is a tool that augments human creativity, helping designers brainstorm ideas while leaving emotion to humans.", True),
        ("P18", "Content Writer", "Artificial intelligence assists human creativity and drafting, though true intelligence and empathy remain uniquely human.", True),
        ("P19", "Marketing Lead", "AI accelerates creative ideation and campaign personalization by connecting audience data with human creativity.", True),
        ("P20", "Musician", "Generative AI mimics patterns in music data, but authentic creativity and artistic intent come from humans.", True),
        ("P21", "UX Researcher", "AI should empower humans rather than replace creativity, supporting intuitive collaboration and accessible tools.", True),
        ("P22", "Architect", "Design automation frees humans from repetitive drafting so we can focus on spatial creativity and human experience.", True),
        ("P23", "Illustrator", "AI learns from existing art data, creating tension between rapid creativity tools and fair credit for humans.", True),

        # Group D: Everyday Pragmatists / Mixed General Public
        ("P24", "Retail Manager", "AI is basically smart software that automates customer chat and inventory data to save time on daily jobs.", True),
        ("P25", "Healthcare Nurse", "AI helps doctors spot patterns in medical data faster, though humans must still make caring decision-making choices.", True),
        ("P26", "High School Teacher", "Artificial intelligence feels like a fast search assistant that learns from internet data, sometimes with bias or errors.", True),
        ("P27", "Small Business Owner", "AI tools automate emails and marketing creativity for small teams without needing deep knowledge of algorithms.", True),
        ("P28", "Accountant", "Automation and AI handle repetitive spreadsheet data accurately, shifting human jobs toward advisory decision-making.", True),
        ("P29", "Retired Librarian", "AI organizes vast information and mimics human intelligence, yet still needs human judgment to verify truth and trust.", True),
        ("P30", "Operations Coordinator", "AI connects scheduling data and automation to speed up routine decision-making alongside human staff.", True),

        # Intentional edge cases to validate data cleaning pipeline:
        ("P02", "Data Analyst", "Artificial intelligence combines statistical algorithms, data pipelines, and machine learning to improve decision-making.", True),  # Exact duplicate
        ("P31", "Unknown", "", True),  # Empty response
        ("P32", "Student", None, True),  # Missing value
    ]

    df = pd.DataFrame(
        records,
        columns=["participant_id", "role_background", "open_response", "is_synthetic"],
    )
    df.to_csv(raw_csv_path, index=False)
    return df


def load_dataset(raw_csv_path: Path) -> pd.DataFrame:
    """Loads existing raw dataset or generates the synthetic dataset if not present."""
    if not raw_csv_path.exists():
        return generate_synthetic_dataset(raw_csv_path)
    return pd.read_csv(raw_csv_path)