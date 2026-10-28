# Data Directory

Raw and processed data files are git-ignored (except documentation).

## Raw Corpus (`data/raw/`)

Official English translations of key legislative acts from the Nepal Law Commission (and Constituent Assembly Secretariat):

| File | Document | Pages | Description |
|---|---|---|---|
| `Constitution-of-Nepal.pdf` | Constitution of Nepal (2015 / 2072 BS) | 240 | Supreme law of Nepal; Part 3 Fundamental Rights & Duties, federal structure, judiciary. |
| `The-Labour-Act-2017.pdf` | Labour Act, 2017 (2074 BS) | 82 | Employment relationships, leave provisions, remuneration, collective disputes. |
| `National-Civil-Code.pdf` | Muluki Civil Code, 2017 (2074 BS) | 409 | Property rights, contracts, family law, torts, and civil legal principles. |
| `National-Penal-code.pdf` | Muluki Penal Code, 2017 (2074 BS) | 200 | Criminal offenses, punishments, public order, and state security. |
| `National-Criminal-Procedure-Code.pdf` | Muluki Criminal Procedure Code, 2017 | 258 | Investigation, arrest, bail, trial procedure, and appellate jurisdiction. |
| `Industrial-Enterprises-Act-2020.pdf` | Industrial Enterprises Act, 2020 (2076 BS) | 68 | Registration, classification, incentives, and concessions for industries. |
| `Foreign-Investment-and-Technology-Transfer-Act-.pdf` | FITTA, 2019 (2075 BS) | 36 | Foreign investment approvals, repatriation, and technology transfer regulations. |
| `Environment-Protection-Act.pdf` | Environment Protection Act, 2019 (2076 BS) | 28 | EIA requirements, pollution control, climate change provisions, and penalties. |
| `National-Human-Rights-Commission-Act.pdf` | NHRC Act, 2012 (2068 BS) | 21 | Jurisdiction, powers, inquiry procedure, and recommendations of the NHRC. |
| `Prevention-of-Corruption_Act.pdf` | Prevention of Corruption Act, 2002 (2059 BS) | 32 | Offenses related to public servants, bribery, property investigation, penalties. |

**Source & License:** Official publications of the Government of Nepal (Nepal Law Commission / Constituent Assembly). Government legislative enactments are public documents published in the Nepal Gazette for public information and enforcement.

---

## 1-Hour Nepali PDF Extraction Spike Findings

An extraction test was conducted on official Nepal Law Commission Nepali PDFs (e.g. `नेपालको संविधान unicode भाद्र २०८१_mtbuyjt.pdf` and `विद्युतीय कारोवार ऐन २०६३_jkxllj0.pdf`) using standard PDF text extractors (`pypdf`, `pdfplumber`):

- **Finding:** The government PDFs were compiled using legacy/proprietary Devanagari font encodings (Preeti / Kantipur derivatives) with corrupted `ToUnicode` CMap tables.
- **Observed Artifacts:** 
  - Ligatures and vowels decompose into incorrect characters: `मिति` extracts as `मममि`, `हामी` extracts as `िामी`, `ऐन` extracts as `ऐि`, `बमोजिम` extracts as `बमोक्षजम`, and conjunct letters (like `ह्र`, `द्ध`, `त्र`) fail or drop glyphs.
- **Architectural Conclusion:** Direct indexing of raw Nepali PDFs results in severely degraded lexical (BM25) and semantic retrieval. 
- **Adopted Design:** We index clean official English translations while supporting **cross-lingual bilingual queries** (English and Nepali) via `BAAI/bge-m3` embeddings, which map Devanagari queries into the shared multilingual embedding space out-of-the-box.

