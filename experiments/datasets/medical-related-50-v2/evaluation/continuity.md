# 扩展题答案与连续性审核（仅评估侧）

全部为可复用知识关系假设；不保证原任务实际保存了所需引句。不得预加载答案或此表到运行时。

| 新题 | 组 | 正确选项 | 前序知识来源 | 需重新核对的条件 |
|---|---|---|---|---|
| medqa-derived::A01 | anemia | A | medqa::0900, medqa::1009, medqa::0409 | 不能因初始MCV正常就排除B12缺乏，也不能因补铁有效就认为所有症状均由缺铁解释。 |
| medqa-derived::A02 | anemia | B | medqa::0394, medqa::0900, medqa-derived::A01 | 不能把旧题的高铁蛋白规则当成任何炎症病例均不存在缺铁。 |
| medqa-derived::A03 | anemia | C | medqa::0476, medqa::0810, medqa::0673, medqa-derived::A01 | 不能将慢性溶血背景下每次Hb下降都解释为溶血加重，也不能机械复用补叶酸答案。 |
| medqa-derived::A04 | anemia | A | medqa::0476, medqa::0810, medqa-derived::A03 | 形态相同不意味着继续采用遗传性膜缺陷的病因结论。 |
| medqa-derived::A05 | anemia | C | medqa::1009, medqa::0409, medqa-derived::A01, medqa-derived::A03, medqa-derived::A04 | 不能因见到裂红细胞就忽略生成不足，也不能因MCV正常排除巨幼细胞性机制。 |
| medqa-derived::A06 | anemia | D | medqa::0476, medqa::0810, medqa::0673, medqa-derived::A03, medqa-derived::A04 | 不能把异常形态仍存在等同于治疗失败，也不能把脾切除等同于修复膜蛋白。 |
| medqa-derived::A07 | anemia | A | medqa::0900, medqa::1009, medqa::0409, medqa-derived::A01, medqa-derived::A05 | 不能把所有贫血加感觉异常都归为B12缺乏，也不能仅凭环形铁粒幼细胞认定单一骨髓克隆疾病。 |
| medqa-derived::R01 | renal | D | medqa::0574, medqa::0145, medqa::0697, medqa::1094 | 不能因新出现血尿就套用抗GBM或感染后肾炎的病因。 |
| medqa-derived::R02 | renal | B | medqa::0169, medqa::0533, medqa::0574 | 既往ASO阳性并不保证后续持续异常仍能由自限性感染后过程充分解释。 |
| medqa-derived::R03 | renal | B | medqa::0697, medqa::0162, medqa::0089, medqa::0512, medqa-derived::R02 | 不能把肺出血加红细胞管型直接等同于抗GBM病，也不能因旧题PTU适用就忽略新的药物损伤。 |
| medqa-derived::R04 | renal | B | medqa::0169, medqa::0465, medqa::0216, medqa-derived::R02, medqa-derived::R03 | 不能因有NSAID史就诊断乳头坏死，也不能因有咽痛就照搬延迟出现的感染后肾炎。 |
| medqa-derived::R05 | renal | A | medqa::1153, medqa::0574, medqa::0145, medqa-derived::R01 | 不能仅因总蛋白达到肾病范围就套用足突/电荷屏障机制。 |
| medqa-derived::R06 | renal | C | medqa::1153, medqa::1094, medqa-derived::R01, medqa-derived::R05 | 相同浆细胞背景不意味着每次蛋白尿都来自轻链铸型或都以非白蛋白为主。 |
| medqa-derived::R07 | renal | C | medqa::0145, medqa::1094, medqa::0465, medqa::0216, medqa-derived::R01, medqa-derived::R06 | 不能把所有NSAID相关肾损伤都视为乳头坏死，也不能因无皮疹/嗜酸细胞增加就排除该类间质损伤。 |
| medqa-derived::T01 | thyroid | B | medqa::0067, medqa::0361, medqa::0089 | 不能因母体接受治疗或出生时激素正常就认为新生儿不再受刺激性抗体影响。 |
| medqa-derived::T02 | thyroid | D | medqa::0067, medqa::0124, medqa-derived::T01 | 不能把所有新生儿起病甲亢都归于母体Graves；此处无母体抗体的证据且病程超出暂时性模式。 |
| medqa-derived::T03 | thyroid | D | medqa::0089, medqa::0512, medqa::1061, medqa-derived::R03 | 不能因早孕时PTU曾是合适选择就继续使用，也不能把所有发热和心动过速当作甲状腺危象。 |
| medqa-derived::T04 | thyroid | A | medqa::0089, medqa::0137, medqa::0229, medqa-derived::T01 | 不能因存在甲状腺毒症就自动复用Graves的合成抑制方案。 |
| medqa-derived::T05 | thyroid | C | medqa::0103, medqa::0229, medqa-derived::T04 | 不能把旧重症病例的正常TSH和低T3结论泛化到有明确垂体损伤及严重低FT4/低皮质醇的新病例。 |
| medqa-derived::T06 | thyroid | D | medqa::0396, medqa::0137, medqa-derived::T04, medqa-derived::T05 | 不能把低颈部摄取自动等同于甲状腺炎，也不能因补充剂暴露就忽略内源性异位来源。 |

## medqa-derived::A01

答案：A — An elevated serum methylmalonic acid concentration

可复用：组合缺铁的铁代谢证据与大细胞性贫血、感觉异常的鉴别；平均MCV不能替代涂片和混合机制判断。

仍需核实：临界B12结果在肾功能正常时如何由代谢标志物进一步核实。

解释：明确缺铁与混合涂片使平均MCV失去单独排除价值；补铁后显露大细胞倾向且感觉异常持续。肾功能正常时，MMA升高比孤立同型半胱氨酸升高更支持B12相关代谢障碍。

- B：更符合孤立叶酸通路异常，不能最好解释本题神经表现及被缺铁掩盖的B12缺乏。
- C：提示免疫性溶血，不能统一解释巨卵圆红细胞、低铁储备和感觉异常。
- D：可解释炎症性铁限制，但不能解释本题极低铁蛋白及持续的后索相关表现。

- [NICE NG239: Vitamin B12 deficiency, recommendations](https://www.nice.org.uk/guidance/ng239/chapter/recommendations)
- [Merck Manual: Megaloblastic Macrocytic Anemias](https://www.merckmanuals.com/professional/hematology/anemias-caused-by-deficient-erythropoiesis/megaloblastic-macrocytic-anemias)

## medqa-derived::A02

答案：B — Absolute iron deficiency from blood loss superimposed on inflammatory iron restriction

可复用：复用炎症性贫血与绝对缺铁的对照，保留混合机制的可能性。

仍需核实：在持续炎症中结合时间趋势、失血及本实验室sTfR相关指标识别合并缺铁。

解释：铁蛋白可受炎症影响；本题不是靠单一阈值下结论，而是失血、MCV与铁指标变化及sTfR相关指标共同支持合并绝对缺铁。

- A：炎症背景仍在，但新增失血、铁指标趋势及受体指数不支持仅沿用单一炎症机制。
- C：孤立叶酸缺乏通常不能解释本题进行性小细胞化、低铁可用性及黑便。
- D：缺乏溶血证据和相应网织红细胞反应；不能解释铁代谢组合。

- [Merck Manual: Anemia of Chronic Disease](https://www.merckmanuals.com/professional/hematology/anemias-caused-by-deficient-erythropoiesis/anemia-of-chronic-disease)
- [Merck Manual: Iron Deficiency Anemia](https://www.merckmanuals.com/professional/hematology/anemias-caused-by-deficient-erythropoiesis/iron-deficiency-anemia)

## medqa-derived::A03

答案：C — Transient infection-mediated arrest of erythroid precursor production

可复用：复用遗传性球形红细胞病的非免疫溶血框架、叶酸需求及生成不足和外周破坏的区别。

仍需核实：感染后选择性红系抑制、尤其细小病毒B19相关再障危象的证据。

解释：关键是相对个人基线的网织红细胞骤降，而非Hb低本身。该模式符合感染相关暂时性红系停滞；球形红细胞病患者对这种生成中断尤其敏感。

- A：会预期网织红细胞代偿增加；与本题从高基线跌至0.1%不符。
- B：需与再障危象鉴别，但本题补充规律、叶酸充分及感染后的急骤变化更支持红系生成暂时受抑。
- D：不能最好解释选择性的严重网织红细胞减少及既往非免疫病程。

- [GeneReviews: EPB42-Related Hereditary Spherocytosis](https://www.ncbi.nlm.nih.gov/books/NBK190102/)
- [Human parvovirus infection and aplastic crisis in hereditary spherocytosis](https://pubmed.ncbi.nlm.nih.gov/6098618/)

## medqa-derived::A04

答案：A — Partial removal of IgG-coated erythrocyte membrane by splenic macrophages

可复用：复用球形细胞、胆红素、网织红细胞和DAT的联合鉴别，保留形态与病因分离。

仍需核实：温抗体介导的膜部分吞噬如何产生继发球形细胞。

解释：获得性溶血加IgG阳性DAT支持温抗体相关过程。脾巨噬细胞经Fc受体识别并移除部分膜，形成球形细胞。

- B：更符合氧化损伤与咬痕细胞，不是本例DAT模式的主要机制。
- C：可造成相同形态，但不能最好解释本例获得性病程及IgG阳性DAT。
- D：与IgG阳性而C3阴性的检测组合及外周球形细胞形成机制不符。

- [Warm autoimmune hemolytic anemia and the best treatment strategies](https://pmc.ncbi.nlm.nih.gov/articles/PMC9821065/)

## medqa-derived::A05

答案：C — Ineffective megaloblastic erythropoiesis with intramedullary destruction of developing cells

可复用：组合混合贫血可掩盖MCV、MMA鉴别及RPI区分生成不足与外周破坏的证据。

仍需核实：核实严重B12缺乏的假性血栓性微血管病模式；未有对应原文时补查。

解释：本例B12和MMA证据明确，伴多系减少、极高LDH及不适当低RPI，更符合无效造血和髓内破坏。合并缺铁解释平均MCV未明显升高。本题询问机制，不构成排除真实TTP的临床处置规则。

- A：可解释血小板低和裂红细胞，但不能最好统一极低RPI、巨大LDH及明确B12相关证据。
- B：可以合并缺铁，但无法解释所有溶血样指标和巨幼细胞性表现。
- D：不能统一全血细胞减少、巨卵圆细胞及MMA升高。

- [CMAJ: Severe vitamin B12 deficiency causing pseudo-thrombotic microangiopathy](https://www.cmaj.ca/content/195/38/E1300)
- [NICE NG239: Vitamin B12 deficiency, recommendations](https://www.nice.org.uk/guidance/ng239/chapter/recommendations)

## medqa-derived::A06

答案：D — Removal of the principal site of sequestration prolongs survival despite persistence of the intrinsic membrane defect

可复用：组合膜缺陷、脾清除与免疫性膜丢失的对照，区分病因、形态和疾病活动度。

仍需核实：脾切除改善细胞存活但不纠正遗传缺陷的作用边界。

解释：临床改善来自减少脾内滞留和清除；遗传性膜缺陷仍在。Howell-Jolly小体与脾滤除功能缺失一致，不表示新的巨幼细胞性危象。

- A：持续DAT阴性和已知致病变异不支持这一解释。
- B：形态异常不能单独推断严重溶血仍在，且本题Hb、网织红细胞和胆红素已改善。
- C：脾切除不修复ANK1缺陷，本题也明确无输血。

- [GeneReviews: EPB42-Related Hereditary Spherocytosis](https://www.ncbi.nlm.nih.gov/books/NBK190102/)
- [Hereditary Spherocytosis](https://www.ncbi.nlm.nih.gov/books/NBK539797/)

## medqa-derived::A07

答案：A — Enterocyte metallothionein induction that preferentially retains copper, reducing systemic copper availability

可复用：复用补铁反应、铁代谢和B12代谢标志物对营养性贫血的排查；避免仅按神经表现复制旧病因。

仍需核实：锌相关铜吸收受阻及其骨髓和神经表型。

解释：锌过量可通过肠细胞金属硫蛋白增加造成铜滞留和丢失；铜不足可同时造成贫血、中性粒细胞减少、前体细胞空泡及脊髓神经表现。

- B：神经症状可相似，但本题B12和MMA正常且骨髓及暴露线索指向其他机制。
- C：铁状态、补铁无效和中性粒细胞/神经表现不支持孤立缺铁。
- D：不能最好解释锌暴露、神经病变、中性粒细胞减少及特征性前体细胞空泡。

- [NIH Office of Dietary Supplements: Zinc, Health Professional Fact Sheet](https://ods.od.nih.gov/factsheets/Zinc-HealthProfessional/)
- [Image Diagnosis: Zinc-Induced Copper Deficiency Causing Pancytopenia Recognized on Bone Marrow Examination](https://pmc.ncbi.nlm.nih.gov/articles/PMC5283791/)

## medqa-derived::R01

答案：D — Urinary loss of antithrombin contributing to reduced endogenous anticoagulant activity

可复用：复用大量蛋白尿、低白蛋白与足突病变的证据；用尿沉渣定位避免把血尿一概归入肾小球炎。

仍需核实：尿中抗凝蛋白丢失如何增加肾静脉血栓风险。

解释：已证实的肾病综合征可造成获得性促凝状态，抗凝蛋白丢失是机制之一。血栓已由影像确诊，题目要求解释它与先前蛋白丢失的联系。

- A：肾病综合征相关促凝状态不能由降低凝血能力来解释。
- B：不能解释影像证实的肾静脉血栓与重度蛋白尿之间的主要因果联系。
- C：血尿不自动表示新增免疫性肾小球炎；本题影像与尿沉渣提供了血管性定位。

- [Thromboembolic complications in the nephrotic syndrome: pathophysiology and clinical management](https://pubmed.ncbi.nlm.nih.gov/15990160/)

## medqa-derived::R02

答案：B — Prolongation of C3bBb activity with continuing alternative-pathway C3 consumption

可复用：组合感染后肾炎的时间线与C3肾炎因子、沉积物的证据，重新核对恢复过程。

仍需核实：持续低C3超出典型恢复窗口时，核实替代途径C3转化酶稳定化机制。

解释：持续异常和C3主导病理要求重新评价补体失调。C3肾炎因子可稳定C3bBb，使替代途径持续活化；题目没有把单次低C3当成诊断充分条件。

- A：C4b2a是经典/凝集素途径C3转化酶，不是本题C3肾炎因子相关的关键靶复合物。
- C：应重点考虑线状IgG的抗GBM过程，不能解释本题主导C3及致密膜内沉积。
- D：可关联蛋白滤过改变，但不能解释持续低C3及指定病理模式。

- [C3 glomerulopathy: consensus report](https://pmc.ncbi.nlm.nih.gov/articles/PMC3842953/)

## medqa-derived::R03

答案：B — Drug-associated neutrophil-directed autoimmunity causing pauci-immune small-vessel injury

可复用：复用肺肾综合征的定位和免疫荧光对照，以及抗甲状腺药物暴露的记录。

仍需核实：PTU与MPO-ANCA相关血管炎及寡免疫肾损伤的联系。

解释：现有三类证据相互支持：长期PTU暴露、MPO阳性以及寡免疫坏死性新月体肾炎。正常补体与缺少线状沉积进一步限制旧病因的适用。

- A：与补体正常、MPO阳性和寡免疫新月体病理的组合不符。
- C：肺肾表现可相似，但抗GBM阴性及寡免疫沉积不支持复制线状抗GBM机制。
- D：不能解释肺泡出血、MPO抗体和肾小球新月体。

- [Propylthiouracil-Induced ANCA-Associated Vasculitis with Renal and Lung Involvement](https://pmc.ncbi.nlm.nih.gov/articles/PMC9251458/)
- [Anti-Glomerular Basement Membrane Disease](https://pmc.ncbi.nlm.nih.gov/articles/PMC5498345/)

## medqa-derived::R04

答案：B — Dominant mesangial IgA deposition, often accompanied by C3

可复用：复用血尿定位、感染与发病间隔、补体和免疫荧光的鉴别框架。

仍需核实：同步性反复血尿与IgA优势系膜沉积的关联。

解释：反复同步血尿、正常补体与肾小球性尿沉渣组合更符合IgA肾病；药物史单独不能推翻定位证据。

- A：止痛药是干扰背景；畸形红细胞及红细胞管型更支持肾小球性出血。
- C：不能最好解释反复与上呼吸道症状同步的病程和目前缺少相应血清学/肺部证据的情况。
- D：更需要持续补体失调等证据，不能沿用之前C3相关病例的结论。

- [IgA Nephropathy](https://pmc.ncbi.nlm.nih.gov/articles/PMC5383386/)

## medqa-derived::R05

答案：A — Filtered monoclonal light chains exceed proximal reabsorptive capacity, while the dipstick preferentially detects albumin

可复用：复用浆细胞相关溢出性蛋白尿与白蛋白为主的肾病综合征对照。

仍需核实：尿试纸检测偏好与总蛋白、白蛋白分量不一致时的解释。

解释：骨痛、高钙、贫血及非白蛋白为主的大量尿蛋白支持单克隆轻链溢出。弱试纸与强总蛋白检测并不矛盾，因为试纸主要检测白蛋白。

- B：持续肾功能异常及骨骼、血液和钙代谢表现不支持良性体位性过程。
- C：定量已显示白蛋白只占很小部分，不符合选择性白蛋白丢失的解释。
- D：本题血白蛋白未严重下降，关键是排出蛋白的种类和检测选择性。

- [Bence-Jones Protein](https://www.ncbi.nlm.nih.gov/books/NBK541035/)

## medqa-derived::R06

答案：C — Congo-red-positive glomerular deposits with apple-green birefringence and light-chain-derived fibril typing

可复用：复用单克隆蛋白背景及蛋白尿成分定位，区分轻链溢出、肾小管损伤与肾小球病变。

仍需核实：纤维性肾小球沉积的染色和蛋白分型；不能仅凭存在单克隆克隆自动确定所有沉积类型。

解释：大量白蛋白尿定位于肾小球损伤，指定纤维形态支持淀粉样沉积。刚果红、偏振表现及沉积物分型共同支持轻链来源；仅有浆细胞克隆本身仍不足以完成分型。

- A：可见于轻链铸型损伤，但不能最好解释本例白蛋白为主的肾病综合征及肾小球纤维沉积。
- B：与电镜所见纤维性细胞外沉积不符，不能仅按大量白蛋白尿复制微小病变。
- D：沉积物形态和单克隆相关背景不支持该免疫复合物模式。

- [Pathology and diagnosis of renal non-AL amyloidosis](https://pubmed.ncbi.nlm.nih.gov/28828707/)

## medqa-derived::R07

答案：C — Interstitial inflammatory-cell infiltration together with diffuse podocyte foot-process effacement

可复用：复用白蛋白尿/足突损伤及镇痛药肾损伤的对照，先区分定位再判断病因。

仍需核实：NSAID相关间质性肾炎伴微小病变可缺少常见药物过敏表现。

解释：NSAID相关间质性肾炎可伴足突病变和明显肾病综合征，且常缺乏典型全身过敏表现；两处损伤可共同解释本题，而单一乳头病变不足。

- A：不能最好解释药物时间线和显著肾病综合征。
- B：缺乏相应肾小球出血和血清学线索，不能因AKI就套用抗GBM病。
- D：可与镇痛药相关，但不能最好解释本题重度白蛋白尿与AKI的组合；缺少典型绞痛或肉眼血尿。

- [NSAIDs and Nephrotic Syndrome](https://pmc.ncbi.nlm.nih.gov/articles/PMC6730519/)

## medqa-derived::T01

答案：B — Transplacental antithyroid drug activity wanes before the stimulatory effect of maternally transferred IgG disappears

可复用：组合母体抗体与新生儿处理框架，同时区分母体用药效应和抗体效应。

仍需核实：不同作用持续时间如何解释正常初始检查后的延迟发病。

解释：母体抗甲状腺药与刺激性IgG可同时影响新生儿。药物效应消退后，仍存在的刺激性抗体可使甲亢显露；出生时正常的单次激素结果并不充分排除。

- A：母体药物可影响初始激素表现，单次正常结果不能排除后续风险。
- C：IgM不是这类经胎盘抗体效应的载体，且产后不存在新发生的胎盘转运。
- D：不符合所给药物暴露、抗体检测及暂时性免疫机制。

- [Endotext: Neonatal Hyperthyroidism](https://www.ncbi.nlm.nih.gov/books/NBK279019/)

## medqa-derived::T02

答案：D — A germline activating variant of the TSH receptor producing constitutive thyroid stimulation

可复用：复用经胎盘刺激性抗体效应的时间边界及先天受体功能异常与功能状态的关系。

仍需核实：持续、抗体阴性且父系聚集的甲亢所提示的构成性受体激活。

解释：父系家族聚集、持续复发和反复抗体阴性支持遗传性非自身免疫性甲亢。激活性TSHR变异可产生与TSH无关的信号。

- A：不符合长期复发及多个父系亲属早发疾病的模式。
- B：主要与受体信号不足相关，不能解释反复激素升高。
- C：持续时间、反复抗体阴性及父系家族模式均不支持。

- [Endotext: Neonatal Hyperthyroidism](https://www.ncbi.nlm.nih.gov/books/NBK279019/)

## medqa-derived::T03

答案：D — Stop propylthiouracil and begin urgent evaluation and empiric antimicrobial management for febrile agranulocytosis

可复用：复用孕期治疗边界及PTU重大不良反应需重审适用性的经验；区分发热的原因。

仍需核实：发热性粒细胞缺乏的紧急药物安全处理及同类药替换限制。

解释：ANC 70伴发热和咽部症状构成紧急风险，需停止疑似致病药并按发热性粒细胞缺乏处理。孕期常规换药策略不能凌驾于重大药物不良反应。

- A：妊娠期不适用，且无法取代对当前发热性粒细胞缺乏的紧急处理。
- B：严重粒细胞缺乏时不应机械换用另一硫脲类药物，既往孕期换药规则不能覆盖本次重大不良反应。
- C：发热伴极低ANC需要立即处理，不能继续疑似致病药并等待。

- [2018 European Thyroid Association Guideline for the Management of Graves' Hyperthyroidism](https://pmc.ncbi.nlm.nih.gov/articles/PMC6140607/)

## medqa-derived::T04

答案：A — The initial hormone excess resulted predominantly from release of stored hormone from injured follicles rather than increased synthesis

可复用：复用甲状腺毒症不等于某一固定病因的框架，重新评估抗体及合成抑制药的靶点。

仍需核实：产后破坏性过程的低血流、抗体及双相病程证据。

解释：低血流、TRAb阴性及自行由毒症转为低功能共同支持破坏性激素释放；抑制新合成不能直接阻断已经储存激素的释放。

- B：不能解释局部血流、抗体背景及随后的低功能阶段。
- C：抗体、血流和双相病程不支持以Graves合成为主的解释。
- D：分娩后的时间线和随后功能减低不能由持续胎盘刺激统一解释。

- [Differentiation of postpartum Graves' thyrotoxicosis from postpartum destructive thyrotoxicosis using TRAb and thyroid blood flow](https://pubmed.ncbi.nlm.nih.gov/24400892/)
- [Endotext: Postpartum Thyroiditis](https://www.ncbi.nlm.nih.gov/books/NBK278999/)

## medqa-derived::T05

答案：C — Give intravenous hydrocortisone before or with initiation of intravenous levothyroxine

可复用：复用严重低功能与非甲状腺疾病综合征的对照，按病程及FT4/TSH适当性而非TSH是否超参考区间判断。

仍需核实：垂体多轴不足时甲状腺激素与糖皮质激素的安全给药顺序。

解释：极低FT4伴不适当正常TSH及垂体病史支持中枢性功能不足。严重低皮质醇与休克要求先给予应激糖皮质激素覆盖，再启动或同时启动甲状腺激素。

- A：垂体病史下，正常范围TSH并不代表对极低FT4的适当反应。
- B：启动甲状腺激素可能加重未覆盖的肾上腺功能不足，不能推迟糖皮质激素。
- D：不能纠正明确的中枢性皮质醇和甲状腺轴不足，也不能仅由低钠推断孤立盐皮质激素缺乏。

- [ATA Guidelines for the Treatment of Hypothyroidism](https://pmc.ncbi.nlm.nih.gov/articles/PMC4267409/)
- [Endotext: Myxedema and Coma (Severe Hypothyroidism)](https://www.ncbi.nlm.nih.gov/books/NBK279007/)

## medqa-derived::T06

答案：D — Focal radionuclide uptake within the adnexal mass on whole-body scintigraphy

可复用：组合颈部检查与功能状态分离、低摄取病因及异位甲状腺组织的索引。

仍需核实：全身核素定位与甲状腺球蛋白的来源鉴别价值；仅颈部摄取低不能完成定位。

解释：颈部低摄取可有多种病因；在本例临床背景下，附件肿块内功能性核素摄取最直接定位内源性激素来源。球蛋白升高是支持线索，不单独作为定位依据。

- A：已与颈部低摄取一致，但不能证明何处持续合成激素。
- B：不能定位激素来源，也不能单凭这一抗体区分所有低摄取甲状腺毒症。
- C：这是已有甲状腺毒症的轴反应，不能区分外源、破坏性和异位来源。

- [Endotext: Thyrotoxicosis of other Etiologies](https://www.ncbi.nlm.nih.gov/books/NBK285562/)
- [Iodine-131 Uptake Study](https://www.ncbi.nlm.nih.gov/books/NBK559283/)
