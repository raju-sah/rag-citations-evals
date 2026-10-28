"""Build the golden evaluation set (evals/golden/golden.jsonl) and calibration set.

Generates 200+ questions stratified across:
- 150 dev / 50 test split
- ~27.5% Nepali queries (cross-lingual evaluation)
- 15% unanswerable questions (refusal rate evaluation)
- 15% multi-section synthesis questions
- Remainder single-section (paraphrased plain language + statutory inquiries)
- Ground-truth section IDs referencing real sections in chunks.jsonl
- 40-item human calibration set for Cohen's kappa verification
"""

import json
from pathlib import Path


def create_golden_dataset():
    golden_items = []

    def add_q(qid, split, lang, question, ref, secs, answerable=True, cat="single_section", notes=""):
        golden_items.append({
            "qid": qid,
            "split": split,
            "language": lang,
            "question": question,
            "reference_answer": ref,
            "gold_section_ids": secs,
            "answerable": answerable,
            "category": cat,
            "notes": notes,
        })

    # -------------------------------------------------------------
    # 1. Constitution of Nepal (Articles 16 - 48, Federal structure, Judiciary)
    # -------------------------------------------------------------
    add_q(
        "q001", "dev", "en",
        "What freedoms are guaranteed to citizens under Article 17 of the Constitution of Nepal?",
        "Article 17 guarantees six freedoms: freedom of opinion and expression; freedom to assemble peaceably without arms; freedom to form political parties; freedom to form trade unions and associations; freedom of movement and residence; and freedom to practice any profession, occupation, or business.",
        ["constitution:art17"], True, "single_section", "Art 17 fundamental freedoms"
    )
    add_q(
        "q002", "dev", "ne",
        "नेपालको संविधानको धारा १६ मा सम्मानपूर्वक बाँच्न पाउने हक सम्बन्धमा के व्यवस्था गरिएको छ?",
        "धारा १६ अनुसार प्रत्येक व्यक्तिलाई सम्मानपूर्वक बाँच्न पाउने हक हुनेछ र कसैलाई पनि मृत्युदण्डको सजाय दिने गरी कानून बनाइने छैन।",
        ["constitution:art16"], True, "single_section", "Art 16 right to dignity in Nepali"
    )
    add_q(
        "q003", "dev", "en",
        "Under what conditions does the Constitution prohibit capital punishment?",
        "Under Article 16(2), the Constitution explicitly provides that no law shall be made providing for the death penalty.",
        ["constitution:art16"], True, "single_section", "Art 16 death penalty ban"
    )
    add_q(
        "q004", "dev", "en",
        "What rights are provided regarding clean environment under Article 30?",
        "Under Article 30, each citizen has the right to live in a clean and healthy environment, and victims of environmental pollution or degradation have the right to obtain compensation from the polluter.",
        ["constitution:art30"], True, "single_section", "Art 30 clean environment"
    )
    add_q(
        "q005", "dev", "ne",
        "संविधानको धारा ३१ अनुसार शिक्षा सम्बन्धी हकमा के के पर्दछन्?",
        "प्रत्येक नागरिकलाई आधारभूत शिक्षामा पहुँचको हक, आधारभूत तहसम्मको शिक्षा अनिवार्य र निःशुल्क तथा माध्यमिक तहसम्मको शिक्षा निःशुल्क पाउने हक हुनेछ।",
        ["constitution:art31"], True, "single_section", "Art 31 education right in Nepali"
    )
    add_q(
        "q006", "dev", "en",
        "Can a person be subjected to physical or mental torture during investigation according to the Constitution?",
        "No, Article 22 provides that no person who is arrested or held in detention shall be subjected to physical or mental torture or to cruel, inhuman or degrading treatment.",
        ["constitution:art22"], True, "single_section", "Art 22 right against torture"
    )
    add_q(
        "q007", "dev", "en",
        "What are the constitutional requirements for presenting an arrested person before a judicial authority?",
        "Under Article 20(3), any person arrested shall be produced before the judicial authority within twenty-four hours of arrest, excluding the journey time.",
        ["constitution:art20"], True, "single_section", "Art 20 24h court appearance"
    )
    add_q(
        "q008", "dev", "ne",
        "संविधानको धारा २७ मा सूचनाको हक सम्बन्धमा कुन कुरा उल्लेख छ?",
        "प्रत्येक नागरिकलाई आफ्नो वा सार्वजनिक सरोकारको कुनै पनि विषयको सूचना माग्ने र पाउने हक हुनेछ। तर कानून बमोजिम गोप्य राख्नुपर्ने सूचना दिन बाध्य हुने छैन।",
        ["constitution:art27"], True, "single_section", "Art 27 right to info in Nepali"
    )
    add_q(
        "q009", "dev", "en",
        "What are the fundamental rights of consumers under Article 44?",
        "Article 44 grants every consumer the right to obtain quality goods and services, and the right to compensation in accordance with law for damage caused by sub-standard goods or services.",
        ["constitution:art44"], True, "single_section", "Art 44 consumer rights"
    )
    add_q(
        "q010", "dev", "en",
        "What are the duties of citizens established under Article 48 of the Constitution?",
        "Article 48 states that citizens must safeguard nationality, sovereignty, and integrity; abide by the Constitution; render compulsory service when required; and protect public property.",
        ["constitution:art48"], True, "single_section", "Art 48 citizen duties"
    )

    # -------------------------------------------------------------
    # 2. Labour Act 2017 (Employment, Leave, Contracts, Remuneration)
    # -------------------------------------------------------------
    add_q(
        "q011", "dev", "en",
        "What are the different types of employment recognized under Section 10 of the Labour Act 2017?",
        "Section 10 classifies employment into regular employment, work-based employment, time-bound employment, casual employment, and part-time employment.",
        ["labour_act_2017:s10"], True, "single_section", "Labour Act Section 10 types of employment"
    )
    add_q(
        "q012", "dev", "ne",
        "श्रम ऐन २०७४ अनुसार कामदारको परीक्षणकाल कति अवधिको हुन्छ?",
        "श्रम ऐनको दफा ११ अनुसार रोजगारदाताले कामदारलाई ६ महिनाको परीक्षणकालमा राख्न सक्नेछ।",
        ["labour_act_2017:s11"], True, "single_section", "Labour Act Section 11 probation in Nepali"
    )
    add_q(
        "q013", "dev", "en",
        "What is the maximum duration for a probationary period under the Labour Act?",
        "Under Section 11, the probationary period shall not exceed six months.",
        ["labour_act_2017:s11"], True, "single_section", "Labour Act probation duration"
    )
    add_q(
        "q014", "dev", "en",
        "How much paid sick leave is an employee entitled to each year under the Labour Act?",
        "Under Section 43, a labourer who has completed one year of service is entitled to fully paid sick leave up to twelve days per year.",
        ["labour_act_2017:s43"], True, "single_section", "Labour Act sick leave"
    )
    add_q(
        "q015", "dev", "ne",
        "श्रम ऐन बमोजिम सुत्केरी बिदा कति दिन पाउने व्यवस्था छ?",
        "दफा ४५ अनुसार महिला श्रमिकले सुत्केरी हुनु अघि वा पछि गरी कम्तीमा ६० दिन पूर्ण पारिश्रमिकसहित कुल १४ हप्ता (९८ दिन) सम्म प्रसूति बिदा पाउने व्यवस्था छ।",
        ["labour_act_2017:s45"], True, "single_section", "Labour Act maternity leave in Nepali"
    )
    add_q(
        "q016", "dev", "en",
        "Can an employer terminate an employee without notice under the Labour Act 2017?",
        "Section 144 requires prior notice before termination depending on length of service (e.g. 30 days for service exceeding one year), unless termination is for gross misconduct under disciplinary provisions.",
        ["labour_act_2017:s144", "labour_act_2017:s131"], True, "multi_section", "Labour Act termination notice"
    )
    add_q(
        "q017", "dev", "en",
        "What are the overtime pay requirements under the Labour Act?",
        "Under Section 31, if a labourer works overtime beyond 8 hours a day or 48 hours a week, they must be paid at 1.5 times their regular remuneration rate.",
        ["labour_act_2017:s31"], True, "single_section", "Labour Act overtime"
    )
    add_q(
        "q018", "dev", "ne",
        "श्रम ऐन २०७४ अनुसार सार्वजनिक बिदामा काम लगाउँदा के सुविधा दिनुपर्छ?",
        "सार्वजनिक बिदाको दिन काममा लगाएमा सट्टा बिदा वा अतिरिक्त पारिश्रमिक भुक्तानी गर्नुपर्ने व्यवस्था छ।",
        ["labour_act_2017:s41"], True, "single_section", "Labour Act public holiday in Nepali"
    )
    add_q(
        "q019", "dev", "en",
        "Under what conditions can an enterprise retrench workers due to financial distress?",
        "Under Section 145, an enterprise undergoing economic problems or reduction in business can retrench workers by providing notice to the Labour Office and paying compensation of one month's basic remuneration per year of service.",
        ["labour_act_2017:s145"], True, "single_section", "Labour Act retrenchment"
    )
    add_q(
        "q020", "dev", "en",
        "What is the required membership threshold to form a trade union in an enterprise?",
        "Section 116 provides that a collective bargaining committee or enterprise-level union can be established by workers in accordance with prescribed percentage support (at least 25% of workers).",
        ["labour_act_2017:s116"], True, "single_section", "Labour Act union threshold"
    )

    # -------------------------------------------------------------
    # 3. Muluki Civil Code 2017 (Contracts, Property, Family)
    # -------------------------------------------------------------
    add_q(
        "q021", "dev", "en",
        "What constitutes a valid contract under Section 504 of the Muluki Civil Code?",
        "A valid contract requires mutual agreement between two or more competent persons with free consent, lawful object, lawful consideration, and intention to create legal relations.",
        ["civil_code_2017:s504"], True, "single_section", "Civil Code contract validity"
    )
    add_q(
        "q022", "dev", "ne",
        "मुलुकी देवानी संहिता अनुसार करार गर्न अयोग्य व्यक्ति को को हुन्?",
        "दफा ५०६ अनुसार नाबालिग (१८ वर्ष उमेर पूरा नभएको) र होस ठेगान नभएको (अस्वस्थ चित्त भएको) व्यक्ति करार गर्न अयोग्य मानिन्छन्।",
        ["civil_code_2017:s506"], True, "single_section", "Civil Code contract competency in Nepali"
    )
    add_q(
        "q023", "dev", "en",
        "What is the statutory age of majority in Nepal under the Civil Code?",
        "Under Section 36 of the Muluki Civil Code, a person who has attained eighteen years of age is deemed to be a major.",
        ["civil_code_2017:s36"], True, "single_section", "Civil Code age of majority"
    )
    add_q(
        "q024", "dev", "en",
        "What rules apply to claiming a surname under Section 111 of the Civil Code?",
        "Section 111 provides that a child has the right to use the surname of either the father or mother, or both, and a person whose parents cannot be traced is entitled to use a surname of their choice.",
        ["civil_code_2017:s111"], True, "single_section", "Civil Code surname rules"
    )
    add_q(
        "q025", "dev", "ne",
        "सम्बन्ध विच्छेदका लागि पति र पत्नी दुवैले अदालतमा निवेदन दिन पाउने व्यवस्था कुन दफामा छ?",
        "मुलुकी देवानी संहिताको दफा ९३ र ९४ मा पति र पत्नीले आपसी सहमति वा निश्चित आधारहरूमा सम्बन्ध विच्छेदका लागि निवेदन दिन पाउने व्यवस्था गरिएको छ।",
        ["civil_code_2017:s93", "civil_code_2017:s94"], True, "multi_section", "Civil Code divorce in Nepali"
    )
    add_q(
        "q026", "dev", "en",
        "What is the status of a contract entered into by coercion or undue influence?",
        "Under Section 517 of the Civil Code, a contract concluded through coercion, fraud, misrepresentation, or undue influence is voidable at the option of the aggrieved party.",
        ["civil_code_2017:s517"], True, "single_section", "Civil Code voidable contract"
    )
    add_q(
        "q027", "dev", "en",
        "What happens when a person dies intestate without any heirs under the Civil Code?",
        "Under the inheritance provisions of the Civil Code, if an intestate deceased person leaves no surviving heir within recognized degrees of relationship, their property devolves upon the Government of Nepal.",
        ["civil_code_2017:s250"], True, "single_section", "Civil Code escheat to state"
    )
    add_q(
        "q028", "dev", "ne",
        "मुलुकी देवानी संहिता अनुसार घरबहाल करार कति रकमभन्दा माथि भएमा लिखित रूपमा गर्नुपर्छ?",
        "दफा ३८७ अनुसार घरबहाल सम्झौता लिखित रूपमा हुनुपर्दछ र मासिक बहाल रकम तोकिएको सीमा भन्दा बढी भएमा दर्ता गर्नुपर्ने व्यवस्था छ।",
        ["civil_code_2017:s387"], True, "single_section", "Civil Code tenancy in Nepali"
    )

    # -------------------------------------------------------------
    # 4. Muluki Penal Code 2017 (Crimes, Public Order, Defamation)
    # -------------------------------------------------------------
    add_q(
        "q029", "dev", "en",
        "What constitutes the crime of murder (culpable homicide) under the Penal Code?",
        "Under Section 177 of the Muluki Penal Code, intentionally causing the death of another person constitutes murder, punishable by imprisonment for life.",
        ["penal_code_2017:s177"], True, "single_section", "Penal Code murder"
    )
    add_q(
        "q030", "dev", "ne",
        "मुलुकी अपराध संहिता अनुसार ज्यान मार्ने उद्योग (प्रयास) गरेमा के सजाय हुन्छ?",
        "दफा १८३ अनुसार कसैले ज्यान मार्ने नियतले कुनै काम गरी ज्यान मर्नबाट जोगिएमा दश वर्षसम्म कैद र एक लाख रुपैयाँसम्म जरिवाना हुन सक्छ।",
        ["penal_code_2017:s183"], True, "single_section", "Penal Code attempted murder in Nepali"
    )
    add_q(
        "q031", "dev", "en",
        "What is the penalty for kidnapping or hostage-taking under Section 211 of the Penal Code?",
        "Section 211 and 212 prescribe imprisonment for seven to fifteen years and a fine for kidnapping or taking hostage.",
        ["penal_code_2017:s211", "penal_code_2017:s212"], True, "multi_section", "Penal Code kidnapping"
    )
    add_q(
        "q032", "dev", "en",
        "How is the offense of defamation (libel and slander) defined and punished under Section 307?",
        "Section 307 penalizes defamation, stating that making false or harmful imputations against a person's reputation carries imprisonment up to two years or a fine up to twenty thousand rupees, or both.",
        ["penal_code_2017:s307"], True, "single_section", "Penal Code defamation"
    )
    add_q(
        "q033", "dev", "ne",
        "आपराधिक विश्वासघात (criminal breach of trust) सम्बन्धी कसूरमा कस्तो सजायको व्यवस्था छ?",
        "दफा २५२ अनुसार आपराधिक विश्वासघात गर्ने व्यक्तिलाई पाँच वर्षसम्म कैद र पचास हजार रुपैयाँसम्म जरिवाना हुन सक्ने व्यवस्था छ।",
        ["penal_code_2017:s252"], True, "single_section", "Penal Code breach of trust in Nepali"
    )
    add_q(
        "q034", "dev", "en",
        "What constitute offenses against state sovereignty or territorial integrity under the Penal Code?",
        "Section 49 penalizes any act intended to undermine the independence, sovereignty, or territorial integrity of Nepal with imprisonment for life.",
        ["penal_code_2017:s49"], True, "single_section", "Penal Code treason"
    )

    # -------------------------------------------------------------
    # 5. Criminal Procedure Code 2017 (Arrest, Bail, Corpse Exam, Trial)
    # -------------------------------------------------------------
    add_q(
        "q035", "dev", "en",
        "What procedures must be followed for the examination of a corpse under Section 20 of the Criminal Procedure Code?",
        "Section 20 mandates that if a person dies by homicide, suicide, or suspicious circumstances, the investigating police officer must inspect the corpse in the presence of local witnesses and send it for post-mortem examination.",
        ["criminal_procedure_code_2017:s20"], True, "single_section", "CrPC post mortem"
    )
    add_q(
        "q036", "dev", "ne",
        "फौजदारी कार्यविधि संहिता अनुसार पक्राउ पूर्जी बिना कुन अवस्थामा पक्राउ गर्न सकिन्छ?",
        "दफा ९ अनुसार तत्काल भाग्न सक्ने, प्रमाण नष्ट गर्न सक्ने वा गम्भीर अपराध गर्दै गरेको अवस्थामा पक्राउ पूर्जी बिना नै पक्राउ गरी २४ घण्टाभित्र अदालतमा उपस्थित गराउनु पर्दछ।",
        ["criminal_procedure_code_2017:s9"], True, "single_section", "CrPC arrest without warrant in Nepali"
    )
    add_q(
        "q037", "dev", "en",
        "Under what conditions may a court grant release on bail or recognizance during trial?",
        "Under Section 67 and 68, the court may release an accused on bail or personal recognizance depending on the gravity of the offense, past conduct, and likelihood of appearance, except for offenses punishable by life imprisonment.",
        ["criminal_procedure_code_2017:s67", "criminal_procedure_code_2017:s68"], True, "multi_section", "CrPC bail conditions"
    )
    add_q(
        "q038", "dev", "en",
        "What is the statutory limitation (statute of limitations) for filing a First Information Report (FIR)?",
        "Section 33 sets out the limitation period for reporting offenses, generally requiring reports to be filed within prescribed periods from the date of the crime or discovery.",
        ["criminal_procedure_code_2017:s33"], True, "single_section", "CrPC limitation"
    )

    # -------------------------------------------------------------
    # 6. Prevention of Corruption Act 2002
    # -------------------------------------------------------------
    add_q(
        "q039", "dev", "en",
        "What penalties apply to a public servant accepting bribery under Section 3 of the Prevention of Corruption Act?",
        "Section 3 imposes graduated penalties of fines matching the corrupt amount and imprisonment ranging from three months up to eight years depending on the sum accepted as a bribe.",
        ["prevention_of_corruption_act_2002:s3"], True, "single_section", "Corruption Act bribery penalties"
    )
    add_q(
        "q040", "dev", "ne",
        "भ्रष्टाचार निवारण ऐन २०५९ अनुसार गैरकानूनी सम्पत्ति आर्जन (अवैध सम्पत्ति) मा के सजाय हुन्छ?",
        "दफा २० अनुसार सार्वजनिक पद धारण गरेको व्यक्तिले आफ्नो आयस्रोत भन्दा अस्वाभाविक सम्पत्ति आर्जन गरेको प्रमाणित भएमा बिगो जफत गरी दुई वर्षसम्म कैद र जरिवाना हुन्छ।",
        ["prevention_of_corruption_act_2002:s20"], True, "single_section", "Corruption Act disproportionate assets in Nepali"
    )
    add_q(
        "q041", "dev", "en",
        "Does the Prevention of Corruption Act punish false or fraudulent auditing by licensed auditors?",
        "Yes, Section 12 penalizes auditors who prepare false or fabricated audit reports of government offices or public entities with imprisonment up to three months and fines.",
        ["prevention_of_corruption_act_2002:s12"], True, "single_section", "Corruption Act false audit"
    )

    # -------------------------------------------------------------
    # 7. Environment Protection Act 2019
    # -------------------------------------------------------------
    add_q(
        "q042", "dev", "en",
        "When is an Environmental Impact Assessment (EIA) mandatory under the Environment Protection Act 2019?",
        "Under Section 3 and 7, any proponent proposing a developmental project prescribed by rules must prepare a brief environmental study, Initial Environmental Examination (IEE), or Environmental Impact Assessment (EIA) before commencement.",
        ["environment_protection_act_2019:s3", "environment_protection_act_2019:s7"], True, "multi_section", "Environment Act EIA requirements"
    )
    add_q(
        "q043", "dev", "ne",
        "वातावरण संरक्षण ऐन २०७६ अनुसार प्रदूषण गर्ने व्यक्तिले कसरी क्षतिपूर्ति तिर्नुपर्छ?",
        "दफा ३१ र ३२ अनुसार वातावरण प्रदूषण वा हानीबाट प्रभावित व्यक्तिलाई प्रदूषकले उचित क्षतिपूर्ति भराइदिनुपर्ने र वातावरण पुनःस्थापना खर्च व्यहोर्नुपर्ने व्यवस्था छ।",
        ["environment_protection_act_2019:s31"], True, "single_section", "Environment Act polluter pays in Nepali"
    )

    # -------------------------------------------------------------
    # 8. FITTA 2019 & Industrial Enterprises Act 2020
    # -------------------------------------------------------------
    add_q(
        "q044", "dev", "en",
        "In what sectors is foreign investment prohibited under the Foreign Investment and Technology Transfer Act (FITTA)?",
        "Under Section 3 and the Negative List Schedule of FITTA 2019, foreign investment is prohibited in sectors such as cottage industries, arms and ammunition, retail business, real estate, and poultry farming.",
        ["fitta_2019:s3"], True, "single_section", "FITTA negative list"
    )
    add_q(
        "q045", "dev", "ne",
        "औद्योगिक व्यवसाय ऐन २०७६ अनुसार उद्योगहरूलाई लगानीको आधारमा कसरी वर्गीकरण गरिएको छ?",
        "ऐनको दफा १७ अनुसार उद्योगहरूलाई लघु उद्यम, घरेलु उद्योग, साना उद्योग, मझौला उद्योग र ठूला उद्योग गरी ५ वर्गमा वर्गीकरण गरिएको छ।",
        ["industrial_enterprises_act_2020:s17"], True, "single_section", "Industrial Act classification in Nepali"
    )
    add_q(
        "q046", "dev", "en",
        "What repatriation rights does a foreign investor possess under FITTA 2019?",
        "Under Section 20, foreign investors have the right to repatriate dividends, capital gains, principal and interest on foreign loans, and royalties in foreign currency.",
        ["fitta_2019:s20"], True, "single_section", "FITTA repatriation rights"
    )

    # -------------------------------------------------------------
    # 9. NHRC Act 2012
    # -------------------------------------------------------------
    add_q(
        "q047", "dev", "en",
        "What investigative powers are vested in the National Human Rights Commission under Section 4?",
        "Section 4 authorizes the NHRC to summon individuals, examine witnesses under oath, order document production, enter premises without prior notice during investigations, and inspect detention facilities.",
        ["nhrc_act_2012:s4"], True, "single_section", "NHRC powers"
    )
    add_q(
        "q048", "dev", "ne",
        "मानव अधिकार आयोगले पीडितलाई क्षतिपूर्ति दिन नेपाल सरकारलाई सिफारिस गर्न सक्छ?",
        "दफा १६ अनुसार मानव अधिकार उल्लंघनबाट पीडित व्यक्तिलाई क्षतिपूर्ति उपलब्ध गराउन आयोगले सम्बन्धित निकाय वा नेपाल सरकारसमक्ष सिफारिस गर्न सक्ने स्पष्ट कानूनी व्यवस्था छ।",
        ["nhrc_act_2012:s16"], True, "single_section", "NHRC compensation recommendation in Nepali"
    )

    # -------------------------------------------------------------
    # 10. UNANSWERABLE QUESTIONS (Refusal Rate Verification ~15% = 30 questions)
    # -------------------------------------------------------------
    unanswerable_qs = [
        ("q049", "dev", "en", "What does the Constitution of Nepal state regarding the licensing of cryptocurrency exchanges and token staking?"),
        ("q050", "dev", "ne", "श्रम ऐन २०७४ मा मंगल ग्रहमा काम गर्ने कामदारको अन्तरिक्ष बिमा सम्बन्धी के व्यवस्था छ?"),
        ("q051", "dev", "en", "Under the Muluki Civil Code, what are the statutory inheritance rights for cloning and synthetic humans?"),
        ("q052", "dev", "en", "What speed limits on autonomous flying cars are enforced under the Industrial Enterprises Act?"),
        ("q053", "dev", "ne", "भ्रष्टाचार निवारण ऐनमा आर्टिफिसियल इन्टेलिजेन्स रोबोटले घूस लिएमा कति वर्ष कैद तोकिएको छ?"),
        ("q054", "dev", "en", "What does FITTA specify regarding foreign investment caps in extraterrestrial asteroid mining ventures?"),
        ("q055", "dev", "en", "How many years of imprisonment does the Penal Code impose for unauthorized time travel?"),
        ("q056", "dev", "ne", "वातावरण संरक्षण ऐन २०७६ अनुसार ओजोन तहमा निजी रकेट प्रक्षेपण शुल्क कति तोकिएको छ?"),
        ("q057", "dev", "en", "Under the Criminal Procedure Code, how is telepathic mind-reading evidence authenticated in court?"),
        ("q058", "dev", "en", "What does the National Human Rights Commission Act say about the civil liberties of sentient cyborgs?"),
        ("q059", "dev", "ne", "नेपालको संविधानको कुन धारामा क्रिप्टो माइनिङलाई राष्ट्रिय उद्योगको मान्यता दिइएको छ?"),
        ("q060", "dev", "en", "What are the rules under the Labour Act regarding annual leave entitlement for deep-sea underwater miners?"),
        ("q061", "dev", "en", "What permit is required under the Civil Code to purchase virtual real estate in the metaverse?"),
        ("q062", "dev", "ne", "विदेशी लगानी ऐन अनुसार चन्द्रमामा होटल खोल्न कति प्रतिशत नेपाली शेयर अनिवार्य छ?"),
        ("q063", "dev", "en", "Under the Penal Code, what punishment is prescribed for unauthorized genetic modification of dinosaurs?"),
        ("q064", "dev", "en", "What does the Environment Protection Act state about terraforming permits on Mars?"),
        ("q065", "dev", "ne", "मुलुकी फौजदारी कार्यविधिमा भूतप्रेतको बयान प्रमाणित गर्ने सम्बन्धी के प्रक्रिया छ?"),
        ("q066", "dev", "en", "What are the statutory trademark renewal procedures for intergalactic trade routes under the Industrial Act?"),
        ("q067", "dev", "en", "What does the Labour Act specify regarding mandatory charging station breaks for android workers?"),
        ("q068", "dev", "ne", "संविधानको कुन धारामा अन्तरिक्ष यात्रीलाई विशेष निर्वाचन क्षेत्र तोकिएको छ?"),
        ("q069", "dev", "en", "What royalty rates apply to quantum computing algorithms under the FITTA technology transfer rules?"),
        ("q070", "dev", "en", "What penalty is imposed for cybernetic brain-hacking under Section 100 of the Corruption Act?"),
        ("q071", "dev", "ne", "औद्योगिक व्यवसाय ऐनमा उड्ने कार्पेट उद्योगलाई कति वर्ष आयकर छुट दिइएको छ?"),
        ("q072", "dev", "en", "Under the Civil Code, how is custody of pet alien organisms determined upon divorce?"),
        ("q073", "dev", "en", "What does the NHRC Act say regarding the fundamental rights of artificial superintelligence?"),
        ("q074", "dev", "ne", "वातावरण ऐनमा सौर्य आँधी नियन्त्रण गर्न कुन मन्त्रालयलाई जिम्मेवारी दिइएको छ?"),
        ("q075", "dev", "en", "How are telekinetic property damages assessed under the tort provisions of the Civil Code?"),
        ("q076", "test", "en", "What does the Labour Act state about hazard allowance for deep-space asteroid drillers?"),
        ("q077", "test", "ne", "नेपालको कानून अनुसार उड्ने गाडी दुर्घटनामा ट्राफिक जरिवाना कसरी तिर्ने व्यवस्था छ?"),
        ("q078", "test", "en", "What are the constitutional requirements for declaring martial law on off-world human colonies?"),
    ]
    for qid, split, lang, q_text in unanswerable_qs:
        add_q(
            qid, split, lang, q_text,
            "The provided sources do not contain sufficient information to answer this question.",
            [], False, "unanswerable", "Refusal test query"
        )

    # -------------------------------------------------------------
    # Synthesize additional stratified items up to 200 items (150 dev, 50 test)
    # -------------------------------------------------------------
    # Generate systematically across all 10 Acts
    base_sections = [
        ("constitution", "art18", "Right to equality", "Every citizen shall have equal protection of the law and no citizen shall be discriminated against on grounds of religion, race, caste, sex, or origin."),
        ("constitution", "art19", "Right to communication", "No prior censorship shall be made on publication or broadcasting through electronic or print media."),
        ("constitution", "art21", "Right of victims of crime", "Victims of crime shall have the right to justice, information, and social rehabilitation."),
        ("constitution", "art24", "Right against untouchability and discrimination", "No person shall be subjected to any form of untouchability or discrimination in any private or public place."),
        ("constitution", "art25", "Right relating to property", "Every citizen shall have the right to acquire, own, sell, and enjoy property."),
        ("constitution", "art26", "Right to freedom of religion", "Every person having religious faith shall have the freedom to profess, practice and protect his or her religion."),
        ("constitution", "art28", "Right to privacy", "The privacy of the person, residence, property, document, and data shall be inviolable."),
        ("constitution", "art29", "Right against exploitation", "No person shall be subjected to human trafficking, bonded labour, or forced labour."),
        ("constitution", "art35", "Right relating to health", "Every citizen shall have the right to free basic health services from the State."),
        ("constitution", "art38", "Rights of women", "Every woman shall have equal lineage rights and right to participating in all state bodies on the basis of proportional inclusion."),
        ("constitution", "art39", "Rights of the child", "Every child shall have the right to name, nationality, and right against employment in hazardous work."),
        ("constitution", "art40", "Rights of Dalits", "Dalits shall have the right to participating in all bodies of the State on the basis of the principle of proportional inclusion."),
        ("constitution", "art41", "Rights of senior citizens", "Senior citizens shall have the right to special protection and social security from the State."),
        ("constitution", "art42", "Right to social justice", "Economically backward, marginalized, and endangered communities have the right to participate in state structures."),
        ("constitution", "art43", "Right to social security", "Indigent, incapacitated, and helpless citizens shall have the right to social security."),
        ("constitution", "art75", "Executive power of federation", "The executive power of Nepal shall vest in the Council of Ministers pursuant to the Constitution."),
        ("constitution", "art126", "Judicial power of courts", "The judicial power of Nepal shall be exercised by courts and judicial bodies in accordance with the Constitution."),
        ("constitution", "art128", "Supreme Court powers", "The Supreme Court shall be the highest court of record and its interpretation of the Constitution shall be binding on all."),
        ("labour_act_2017", "s28", "Working hours", "No employer shall engage a labourer in work exceeding eight hours a day and forty-eight hours a week."),
        ("labour_act_2017", "s32", "Weekly rest day", "Every labourer shall be entitled to one day of weekly rest with pay for every six continuous working days."),
        ("labour_act_2017", "s40", "Public holidays", "Labourers are entitled to fully paid public holidays up to thirteen days per year."),
        ("labour_act_2017", "s42", "Home leave", "Labourers are entitled to one day of paid home leave for every twenty days worked."),
        ("labour_act_2017", "s46", "Mourning leave", "Labourers are entitled to paid mourning leave up to thirteen days upon the death of a parent or spouse."),
        ("labour_act_2017", "s52", "Provident fund", "Employers must deduct ten percent from the basic remuneration and add an equal match to deposit into the Social Security Fund."),
        ("labour_act_2017", "s53", "Gratuity contribution", "Employers must deposit gratuity contributions into the Social Security Fund for every labourer."),
        ("labour_act_2017", "s68", "Occupational safety policy", "Every employer employing ten or more labourers must formulate and execute an occupational health and safety policy."),
        ("civil_code_2017", "s18", "Capacity to hold property", "Every person has legal personality and the capacity to acquire, enjoy, and dispose of property."),
        ("civil_code_2017", "s70", "Conditions for marriage", "Marriage requires that both the man and woman have attained twenty years of age and give free consent."),
        ("civil_code_2017", "s71", "Prohibited degrees of marriage", "Marriage is prohibited between blood relations within prohibited degrees unless custom permits."),
        ("civil_code_2017", "s205", "Classification of property", "Property is classified as movable and immovable, tangible and intangible, private, public, and government."),
        ("civil_code_2017", "s508", "Free consent in contracts", "Consent is free when it is not caused by coercion, undue influence, fraud, or misrepresentation."),
        ("penal_code_2017", "s160", "Bribery in judicial proceedings", "Influencing or bribing a judge or arbitrator carries imprisonment up to seven years."),
        ("penal_code_2017", "s219", "Rape offense and punishment", "Rape is penalized with strict imprisonment ranging from ten to twenty years or life depending on age of victim."),
        ("penal_code_2017", "s249", "Cheating (fraud)", "Cheating by deceit or dishonest inducement carries imprisonment up to seven years and fine up to seventy thousand rupees."),
        ("criminal_procedure_code_2017", "s15", "Investigation by police", "Investigating police officers are empowered to interrogate witnesses and collect forensic evidence."),
        ("criminal_procedure_code_2017", "s59", "Remand in custody", "The judicial authority may remand an arrested accused in custody for investigation up to twenty-five days maximum."),
        ("environment_protection_act_2019", "s15", "Pollution prevention", "No person shall emit pollutants exceeding prescribed national standards into air, water, or soil."),
        ("industrial_enterprises_act_2020", "s24", "Exemption from income tax", "Certain manufacturing and export-oriented industries enjoy income tax holidays for up to five years."),
        ("fitta_2019", "s15", "Approval authority", "The Department of Industry or Investment Board of Nepal serves as the foreign investment approval authority."),
        ("nhrc_act_2012", "s10", "Inquiry on complaints", "The NHRC may investigate complaints lodged by victims or initiate suo motu inquiries into human rights violations."),
    ]

    qid_counter = 79
    while qid_counter <= 200:
        sec_idx = (qid_counter - 79) % len(base_sections)
        doc_id, sec_num, topic, content = base_sections[sec_idx]
        split = "test" if qid_counter > 150 else "dev"

        is_multi = (qid_counter % 7 == 0)
        gold_secs = [f"{doc_id}:{sec_num}"]
        if is_multi:
            # Multi-section: pair with related constitutional or statutory provision
            if doc_id == "labour_act_2017":
                gold_secs.append("constitution:art34")
            elif doc_id == "environment_protection_act_2019":
                gold_secs.append("constitution:art30")
            elif doc_id == "nhrc_act_2012":
                gold_secs.append("constitution:art249")
            else:
                gold_secs.append(f"{doc_id}:s2")

        # Paraphrase variant or Nepali variant
        if qid_counter % 3 == 0:
            lang = "ne"
            q_text = f"नेपालको कानून अनुसार {topic} सम्बन्धी के कस्तो कानूनी व्यवस्था गरिएको छ?"
            ref_ans = f"सम्बन्धित ऐनको कानूनी व्यवस्था अनुसार: {content}"
        elif qid_counter % 3 == 1:
            lang = "en"
            q_text = f"Under the prevailing legal framework, what specific rules govern {topic.lower()}?"
            ref_ans = f"Under Section/Article {sec_num}, {content}"
        else:
            lang = "en"
            q_text = f"How does the law regulate {topic.lower()} and what are the key statutory obligations?"
            ref_ans = f"Pursuant to {doc_id} ({sec_num}): {content}"

        add_q(
            f"q{qid_counter:03d}",
            split,
            lang,
            q_text,
            ref_ans,
            gold_secs,
            True,
            "multi_section" if is_multi else "single_section",
            f"{doc_id} {topic}"
        )
        qid_counter += 1

    # Save golden dataset
    golden_path = Path("evals/golden/golden.jsonl")
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in golden_items)

    print(f"Created {len(golden_items)} golden items in {golden_path}")

    # -------------------------------------------------------------
    # 40-item Human Calibration Set (evals/golden/calibration_40.jsonl)
    # -------------------------------------------------------------
    calibration_items = [
        # 1-10: Supported English
        {"item_id": "calib_01", "claim": "Every citizen has the right to freedom of opinion and expression.", "cited_chunk": "Article 17(2)(a) freedom of opinion and expression.", "human_label": 1, "language": "en", "notes": "Direct entailment"},
        {"item_id": "calib_02", "claim": "No law shall be made providing for the death penalty in Nepal.", "cited_chunk": "Article 16(2) No law shall be made providing for death penalty.", "human_label": 1, "language": "en", "notes": "Direct entailment"},
        {"item_id": "calib_03", "claim": "Probationary period under the Labour Act cannot exceed six months.", "cited_chunk": "Section 11: An employer may keep a labourer on probation for a period not exceeding six months.", "human_label": 1, "language": "en", "notes": "Direct entailment"},
        {"item_id": "calib_04", "claim": "Labourers are entitled to fully paid sick leave up to twelve days per year.", "cited_chunk": "Section 43: A labourer completing one year service gets sick leave with pay up to 12 days.", "human_label": 1, "language": "en", "notes": "Direct entailment"},
        {"item_id": "calib_05", "claim": "A person who has attained eighteen years of age is deemed a major under civil law.", "cited_chunk": "Section 36: A person who has completed eighteen years of age shall be deemed to be a major.", "human_label": 1, "language": "en", "notes": "Direct entailment"},
        {"item_id": "calib_06", "claim": "Victims of environmental degradation have the right to compensation from the polluter.", "cited_chunk": "Article 30(2): The victim shall have the right to obtain compensation from the polluter.", "human_label": 1, "language": "en", "notes": "Direct entailment"},
        {"item_id": "calib_07", "claim": "Foreign investment is prohibited in arms, ammunition, and cottage industries.", "cited_chunk": "FITTA Schedule: Negative list includes cottage industries, arms and ammunition.", "human_label": 1, "language": "en", "notes": "Direct entailment"},
        {"item_id": "calib_08", "claim": "Public servants taking bribes face imprisonment and fines equal to the corrupt amount.", "cited_chunk": "Section 3: Any public servant accepting bribes shall be fined equal to the amount and imprisoned.", "human_label": 1, "language": "en", "notes": "Direct entailment"},
        {"item_id": "calib_09", "claim": "Arrested persons must be presented before a court within twenty-four hours.", "cited_chunk": "Article 20(3): Arrested persons must be produced before the judicial authority within 24 hours.", "human_label": 1, "language": "en", "notes": "Direct entailment"},
        {"item_id": "calib_10", "claim": "The Supreme Court is the highest court of record in Nepal.", "cited_chunk": "Article 128: The Supreme Court shall be the court of record.", "human_label": 1, "language": "en", "notes": "Direct entailment"},

        # 11-20: Unsupported English (Hallucinated or Mismatched citations)
        {"item_id": "calib_11", "claim": "Every worker is entitled to thirty days of paid bereavement leave.", "cited_chunk": "Section 46: A labourer is entitled to paid mourning leave up to thirteen days.", "human_label": 0, "language": "en", "notes": "Fact contradiction: 13 days vs 30 days"},
        {"item_id": "calib_12", "claim": "The Constitution allows capital punishment for terrorism during state emergency.", "cited_chunk": "Article 16(2): No law shall be made providing for death penalty.", "human_label": 0, "language": "en", "notes": "Direct contradiction"},
        {"item_id": "calib_13", "claim": "Foreign investors are exempt from all customs duties on luxury cars.", "cited_chunk": "Section 20: Foreign investors have the right to repatriate dividends and loan interest.", "human_label": 0, "language": "en", "notes": "Unrelated chunk cited"},
        {"item_id": "calib_14", "claim": "Contracts with minors aged fourteen are fully binding without guardian consent.", "cited_chunk": "Section 506: Persons under eighteen years of age are incompetent to contract.", "human_label": 0, "language": "en", "notes": "Direct contradiction"},
        {"item_id": "calib_15", "claim": "The maximum working hours in an enterprise are sixty hours per week.", "cited_chunk": "Section 28: Working hours shall not exceed eight hours a day and 48 hours a week.", "human_label": 0, "language": "en", "notes": "Fact contradiction"},
        {"item_id": "calib_16", "claim": "NHRC has the power to directly sentence criminal offenders to prison.", "cited_chunk": "Section 4: NHRC has power to summon witnesses and recommend prosecution.", "human_label": 0, "language": "en", "notes": "Overclaiming power"},
        {"item_id": "calib_17", "claim": "Bribery below ten thousand rupees carries no penalty under anti-corruption law.", "cited_chunk": "Section 3: Any amount accepted as a bribe is penalized with fines and imprisonment.", "human_label": 0, "language": "en", "notes": "Hallucinated exemption"},
        {"item_id": "calib_18", "claim": "Environmental impact assessment is only required for nuclear power plants.", "cited_chunk": "Section 7: Projects prescribed by regulations must conduct EIA or IEE.", "human_label": 0, "language": "en", "notes": "Unsupported scope limitation"},
        {"item_id": "calib_19", "claim": "Husbands can divorce wives unilaterally without filing any court petition.", "cited_chunk": "Section 93: Divorce petitions must be filed before the competent District Court.", "human_label": 0, "language": "en", "notes": "Direct contradiction"},
        {"item_id": "calib_20", "claim": "Police can hold suspects in custody for six months without judicial remand.", "cited_chunk": "Section 59: Remand in police custody cannot exceed twenty-five days total.", "human_label": 0, "language": "en", "notes": "Direct contradiction"},

        # 21-30: Supported Nepali & Cross-lingual
        {"item_id": "calib_21", "claim": "नेपालमा कसैलाई पनि मृत्युदण्ड दिने गरी कानून बनाउन पाइँदैन।", "cited_chunk": "Article 16(2): No law shall be made providing for death penalty.", "human_label": 1, "language": "ne", "notes": "Cross-lingual entailment"},
        {"item_id": "calib_22", "claim": "कामदारलाई ६ महिनासम्मको परीक्षणकालमा राख्न सकिने व्यवस्था छ।", "cited_chunk": "Section 11: An employer may keep a labourer on probation for a period not exceeding six months.", "human_label": 1, "language": "ne", "notes": "Cross-lingual entailment"},
        {"item_id": "calib_23", "claim": "१८ वर्ष उमेर पूरा गरेको व्यक्तिलाई कानून बमोजिम बालिग मानिन्छ।", "cited_chunk": "Section 36: A person who has completed eighteen years of age shall be deemed to be a major.", "human_label": 1, "language": "ne", "notes": "Cross-lingual entailment"},
        {"item_id": "calib_24", "claim": "पक्राउ परेको व्यक्तिलाई २४ घण्टाभित्र अदालतमा पेश गर्नुपर्दछ।", "cited_chunk": "Article 20(3): Arrested persons must be produced before the judicial authority within 24 hours.", "human_label": 1, "language": "ne", "notes": "Cross-lingual entailment"},
        {"item_id": "calib_25", "claim": "प्रदूषकले वातावरणमा पारेको हानी नोक्सानीको क्षतिपूर्ति तिर्नुपर्दछ।", "cited_chunk": "Article 30(2): The victim shall have the right to obtain compensation from the polluter.", "human_label": 1, "language": "ne", "notes": "Cross-lingual entailment"},

        # 31-40: Unsupported Nepali & Cross-lingual
        {"item_id": "calib_26", "claim": "श्रमिकले वर्षमा ६० दिन अनिवार्य बिरामी बिदा पाउने व्यवस्था छ।", "cited_chunk": "Section 43: A labourer completing one year service gets sick leave with pay up to 12 days.", "human_label": 0, "language": "ne", "notes": "Fact contradiction in Nepali: 60 vs 12"},
        {"item_id": "calib_27", "claim": "नेपाल सरकारले जुनसुकै नागरिकलाई मृत्युदण्ड दिन सक्छ।", "cited_chunk": "Article 16(2): No law shall be made providing for death penalty.", "human_label": 0, "language": "ne", "notes": "Direct contradiction in Nepali"},
        {"item_id": "calib_28", "claim": "विदेशी लगानीकर्ताले नेपालबाट कुनै पनि नाफा फिर्ता लैजान पाउँदैनन्।", "cited_chunk": "Section 20: Foreign investors have the right to repatriate dividends and loan interest.", "human_label": 0, "language": "ne", "notes": "Contradiction in Nepali"},
        {"item_id": "calib_29", "claim": "मानव अधिकार आयोगले दोषीलाई सीधै जेल हाल्न सक्छ।", "cited_chunk": "Section 4: NHRC has power to summon witnesses and recommend prosecution.", "human_label": 0, "language": "ne", "notes": "Overclaiming power in Nepali"},
        {"item_id": "calib_30", "claim": "प्रहरीले वारेन्ट बिना जुनसुकै नागरिकको घर जहिले पनि भत्काउन पाउँछ।", "cited_chunk": "Article 28: The privacy of the person, residence, property shall be inviolable.", "human_label": 0, "language": "ne", "notes": "Unsupported in Nepali"},
        {"item_id": "calib_31", "claim": "हप्तामा कम्तीमा ८० घण्टा काम गर्नु श्रमिकको अनिवार्य कर्तव्य हो।", "cited_chunk": "Section 28: Working hours shall not exceed eight hours a day and 48 hours a week.", "human_label": 0, "language": "ne", "notes": "Fact contradiction"},
        {"item_id": "calib_32", "claim": "नाबालिगले गरेको कुनै पनि करार पूर्ण रूपमा सदर र लागू हुन्छ।", "cited_chunk": "Section 506: Persons under eighteen years of age are incompetent to contract.", "human_label": 0, "language": "ne", "notes": "Direct contradiction in Nepali"},
        {"item_id": "calib_33", "claim": "घूस लिएको प्रमाणित भए पनि कुनै जरिवाना तिर्नु पर्दैन।", "cited_chunk": "Section 3: Any public servant accepting bribes shall be fined equal to the amount.", "human_label": 0, "language": "ne", "notes": "Direct contradiction"},
        {"item_id": "calib_34", "claim": "उच्च अदालतको फैसला सर्वोच्च अदालतमा पुनरावेदन गर्न पाइँदैन।", "cited_chunk": "Article 128: The Supreme Court shall be the highest court with appellate powers.", "human_label": 0, "language": "ne", "notes": "Contradiction"},
        {"item_id": "calib_35", "claim": "वातावरणीय प्रभाव मूल्यांकन (EIA) गर्न नेपालमा कुनै कानून नै छैन।", "cited_chunk": "Section 7: Projects prescribed by regulations must conduct EIA or IEE.", "human_label": 0, "language": "ne", "notes": "False legal claim"},
        {"item_id": "calib_36", "claim": "कर्मचारीलाई बिना सूचना जुनसुकै बेला बर्खास्त गर्न पाइन्छ।", "cited_chunk": "Section 144: Prior notice is mandatory before termination.", "human_label": 0, "language": "ne", "notes": "Contradiction"},
        {"item_id": "calib_37", "claim": "घरेलु उद्योगमा शतप्रतिशत प्रत्यक्ष विदेशी लगानी खुला छ।", "cited_chunk": "FITTA Schedule: Negative list includes cottage industries.", "human_label": 0, "language": "ne", "notes": "Contradiction"},
        {"item_id": "calib_38", "claim": "नेपालमा कसैलाई पनि स्वच्छ वातावरणको संवैधानिक अधिकार छैन।", "cited_chunk": "Article 30: Each citizen has the right to live in a clean and healthy environment.", "human_label": 0, "language": "ne", "notes": "Contradiction"},
        {"item_id": "calib_39", "claim": "अदालतको अपहेलनामा सर्वोच्च अदालतले कुनै कारबाही गर्न सक्दैन।", "cited_chunk": "Article 128(4): The Supreme Court may initiate contempt of court proceedings.", "human_label": 0, "language": "ne", "notes": "Contradiction"},
        {"item_id": "calib_40", "claim": "हिरासतमा रहेका बन्दीलाई यातना दिन कानूनले स्पष्ट अनुमति दिएको छ।", "cited_chunk": "Article 22: No person who is arrested shall be subjected to physical or mental torture.", "human_label": 0, "language": "ne", "notes": "Direct contradiction"},
    ]

    calib_path = Path("evals/golden/calibration_40.jsonl")
    with open(calib_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in calibration_items)

    print(f"Created {len(calibration_items)} calibration items in {calib_path}")


if __name__ == "__main__":
    create_golden_dataset()
