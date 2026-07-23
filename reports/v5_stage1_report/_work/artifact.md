# Reference template contract

## Reference

- Source copy: `D:\DM_local\reports\v5_stage1_report\_work\reference.docx`
- Original: `C:\Users\mila2\Desktop\7.10寒潮案例调研&特征归纳总结.docx`
- SHA-256: `3903AAF39E014D5CDE1120F00EFA0496BAC3A57FFA61530B0128007E36A0EF86`
- Package: 14 parts; no drawings, headers, footers, comments, or content controls.
- Cached page count: 3 pages.
- Structure audit: 1 portrait section.
- Render status: packaged LibreOffice renderer unavailable on this host; structural
  evidence is in `reference_evidence.json` and `template-style-evidence.json`.

## Page system

- US Letter portrait, 8.5 x 11.0 in.
- Margins: 1.0 in on all sides.
- One column; one continuous section.
- No first-page distinction, running header, or running footer.

## Typography and paragraph roles

- Chinese text: SimSun (`宋体`); Latin/numerals: Times New Roman.
- Title: centered, 20 pt, black, 1.25 line spacing.
- Heading 1: left, 18 pt, black, 1.25 line spacing; numbered in text.
- Heading 2: left, approximately 13-14 pt, black, 1.25 line spacing; numbered in text.
- Body: 12 pt, black, 1.25 line spacing, first-line indent 0.333 in.
- List paragraphs: 12 pt, 1.25 line spacing; compact hanging/list indentation.
- Figure captions are not present in the reference; use 10.5 pt centered SimSun,
  keeping the caption with the preceding figure.

## Tables

- One `Table Grid` example, 3 columns and 5 rows.
- Use simple black borders, a concise header row, centered short values, and
  left-aligned explanatory text.
- New report tables may clone this grid pattern and use content-specific widths.
- Never use fixed row height.

## Components and content flow

- Centered one-line report title.
- Numbered Heading 1 sections.
- Numbered Heading 2 subsections.
- Short analytical paragraphs with explicit interpretation.
- One compact comparison/definition table.
- Final section uses numbered conclusion paragraphs.
- No cover artwork, decorative color system, or complex page furniture.

## Slot map for the new report

- Replace the entire body with a new title and V5 Stage-1 report sections.
- Preserve the source package styles, theme, numbering definitions, page geometry,
  and default document settings where possible.
- Clone the source heading/body/table patterns as needed for a longer report.
- Add inline figures with centered captions; this is an intentional extension
  required by the user and not present in the source.
- A static manual contents list may be added after the title; no field-based TOC
  is required.

## Fidelity gates

- Retained reference must remain byte-for-byte unchanged.
- Final document must retain Letter portrait geometry, 1 in margins, SimSun /
  Times New Roman typography, black hierarchy, 1.25 line spacing, and simple
  report styling.
- Tables must remain readable and figures must not clip or overlap.
- Every final page must be visually inspected if a renderer becomes available;
  otherwise run structural, image, section, and style audits and disclose the
  renderer limitation.
