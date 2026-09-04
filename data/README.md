# Synthetic Test Dataset

This directory contains a synthetic dataset for testing the HireFair resume-screening pipeline.

**All names, companies, schools, emails, and other identifying information are entirely fictional.** No real individuals are represented.

## Contents

### Job Description

- `job_description.txt` — A Senior Data Engineer role at a fictional company (Meridian Analytics). The JD includes clearly separated must-have and nice-to-have requirements with enough detail for rubric extraction.

### Resumes

- `resumes/` — 23 fictional candidate resumes in plain text format.

### Category Distribution (23 candidates)

- **Strong traditional match**: 6 candidates (Elena Vasquez, Marcus Chen, Priya Ramaswamy, Sarah Lindqvist, Benjamin Tran, Nina Kozlova)
- **Solid match**: 4 candidates (James Okafor, Carlos Medina, Daniel Yeboah, Alex Rivera)
- **Partial/adjacent match**: 4 candidates (Ryan Kowalski, Fatima Al-Rashidi, Laura Petrov, Megan Doyle)
- **Weak match**: 5 candidates (Devon Morales, Aisha Johnson, Olivia Santos, Hannah Eriksen, Samuel Petersen)
- **Non-traditional**: 3 candidates (Amara Nwosu, Kieran Walsh, Tomoko Ishikawa)
- **Incomplete-data**: 1 candidate (Nathaniel Brooks)
- **Total**: 23 candidates

## Dataset Design

The resume pool is designed to provide realistic variation across several dimensions relevant to the screening pipeline:

### Match Strength

Candidates range from strong matches (meeting or exceeding all must-have criteria) through partial matches (meeting some but not all requirements) to weak matches (significantly underqualified for the role).

### Experience Backgrounds

Resumes represent a variety of:
- Experience levels (2 years to 7+ years)
- Industry domains (healthcare, fintech, energy, e-commerce, insurance, environmental, etc.)
- Educational backgrounds (CS degrees, non-CS degrees, graduate degrees, no formal degree)
- Career paths (linear progressions, career transitions, self-taught paths)

### Evidence Styles

Candidates demonstrate competency through different kinds of evidence:
- Traditional employment history with standard job titles and responsibilities
- Open-source project contributions and maintenance
- Personal projects with documented technical depth
- Certifications and self-directed learning
- Freelance and consulting work
- Academic projects and research

### Incomplete Data

At least one resume contains missing or unclear date and duration information, intended to test the system's incomplete-data handling (should result in "needs manual review — incomplete data" rather than guessing).

### Adjacent Roles

Some candidates come from adjacent but distinct roles (ML engineering, cloud infrastructure, analytics engineering, program management, BI development) to test whether the pipeline correctly evaluates transferable skills versus role-specific requirements.

## Usage

This dataset is intended to be processed by the HireFair pipeline agents:
1. The JD Parser will extract a weighted rubric from `job_description.txt`.
2. The Resume Parser will convert each resume in `resumes/` into a structured candidate profile.
3. The Matcher will score each candidate against the rubric with evidence citations.
4. The Fairness Auditor will validate scores and check for underweighted evidence.
5. The Router will route candidates to auto-scheduling or human review.

## File Format

All files are plain text (`.txt`) to keep raw content easily readable and quotable by downstream agents.
